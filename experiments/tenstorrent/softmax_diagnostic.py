"""Compare stable FP32 softmax formulations on identical observed TT scores.

Native PJRT execution with explicit non-dry-run options. IR is dumped by the
native compiler; codegen is never invoked. Operator timings include output D2H
from resident inputs and are not whole-attention/pipeline speed measurements.
"""
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class SoftmaxMath(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('builtin', 'renormalized', 'explicit_exp', 'explicit_exp2', 'explicit_reciprocal', 'refined_reciprocal'):
            raise ValueError('Unknown softmax mode')
        self.mode = mode

    def forward(self, x):
        if self.mode in ('builtin', 'renormalized'):
            probabilities = torch.softmax(x, dim=-1)
            if self.mode == 'builtin':
                return probabilities
            return probabilities / probabilities.sum(dim=-1, keepdim=True)
        shifted = x - x.amax(dim=-1, keepdim=True)
        exponentials = torch.exp2(shifted * (1 / math.log(2))) if self.mode == 'explicit_exp2' else torch.exp(shifted)
        total = exponentials.sum(dim=-1, keepdim=True)
        if self.mode in ('explicit_reciprocal', 'refined_reciprocal'):
            inverse = torch.reciprocal(total)
            if self.mode == 'refined_reciprocal':
                inverse = inverse * (2 - total * inverse)
            return exponentials * inverse
        return exponentials / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-directory', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--modes', nargs='+', choices=('builtin', 'renormalized', 'explicit_exp', 'explicit_exp2',
                                                'explicit_reciprocal', 'refined_reciprocal'),
                   default=['builtin', 'renormalized', 'explicit_exp', 'explicit_reciprocal', 'refined_reciprocal'])
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        os.environ.pop(key, None)
    os.environ['PJRT_DEVICE'] = 'TT'
    report = {'status': 'running', 'scope': __doc__,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    def phase(name):
        report['phase'] = name
        (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, flush=True)
    phase('imports')
    try:
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as xr
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.validation import compare_tensors
        from experiments.tenstorrent.trajectory import tensor_digest
        torch.set_num_threads(args.threads)
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['script_sha256'] = sha256(Path(__file__))
        origin = json.loads((args.stage_directory / 'result.json').read_text())
        if origin['status'] != 'completed_stage_diagnostic':
            raise ValueError('Source stage observation was not accepted')
        path = args.stage_directory / 'states.pt'
        if sha256(path) != origin['states_sha256']:
            raise ValueError('Source state artifact changed')
        states = torch.load(path, weights_only=True, map_location='cpu')
        inputs = {name: value['tt'][2].contiguous() for name, value in states.items()}
        report['source_states_sha256'] = origin['states_sha256']
        report['input_digests'] = {k: tensor_digest(v) for k, v in inputs.items()}
        report['cases'] = {}
        outputs = {}
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert xr.device_type() == 'TT'
        os.chdir(args.output_dir)
        for name, values in inputs.items():
            report['cases'][name] = {}
            device_input = values.to(device)
            reference = torch.softmax(values, dim=-1)
            for mode in args.modes:
                label = name + '_' + mode
                phase(label)
                options = {'optimization_level': 0, 'dry_run': 'false',
                           'export_path': str(args.output_dir / (label + '-native-ir'))}
                # Each case is an independent specialization. Do not exhaust
                # Dynamo's per-code-object guard limit across shape/mode trials.
                torch.compiler.reset()
                torch_xla.set_custom_compile_options(options)
                module = SoftmaxMath(mode).eval()
                with torch.no_grad(): cpu = module(values)
                compiled = torch.compile(module, backend='tt', fullgraph=True)
                metrics.clear_all()
                with torch.no_grad():
                    actual = compiled(device_input).cpu()
                    compiled(device_input).cpu()
                    timings = []
                    for _ in range(3):
                        start = time.perf_counter()
                        repeated = compiled(device_input).cpu()
                        timings.append(time.perf_counter() - start)
                        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
                result = {'requested_compile_options': options,
                          'cpu_vs_builtin': compare_tensors(cpu, reference),
                          'tt_vs_builtin_cpu': compare_tensors(actual, reference),
                          'tt_vs_same_cpu_formula': compare_tensors(actual, cpu),
                          'max_row_sum_error': float((actual.sum(-1) - 1).abs().max()),
                          'warm_seconds': timings,
                          'fallback_ops': list(metrics.executed_fallback_ops()),
                          'aten_counters': {key: metrics.counter_value(key) for key in metrics.counter_names() if key.startswith('aten::')}}
                report['cases'][name][mode] = result
                outputs[label] = actual
                (args.output_dir / (label + '-metrics.txt')).write_text(metrics.metrics_report())
                if result['fallback_ops'] or result['aten_counters']:
                    raise ValueError('Unexpected CPU fallback')
        torch.save(outputs, args.output_dir / 'outputs.pt')
        report['outputs_sha256'] = sha256(args.output_dir / 'outputs.pt')
        report['status'] = 'completed_softmax_diagnostic'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
