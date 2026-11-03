"""Measure attention precision boundaries on sealed pretrained Q/K/V.

Resident inputs, output D2H and synchronization included. Native IR dumps are
ordinary execution with dry_run=false; no codegen or full-flow quality claim.
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

from experiments.tenstorrent.inference_attention import ATTENTION_MODES, reciprocal_attention


class PrecisionAttention(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, q, k, v):
        if self.mode == 'legacy':
            # Exact expression order from the accepted full-flow helper before
            # extracting its arithmetic into reciprocal_attention().
            a, b, c = [value.float().permute(0, 2, 1, 3) for value in (q, k, v)]
            scores = (a @ b.transpose(-2, -1)) * (1 / math.sqrt(q.shape[-1]))
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            exponentials = torch.exp(shifted)
            probabilities = exponentials * torch.reciprocal(exponentials.sum(dim=-1, keepdim=True))
            return (probabilities @ c).permute(0, 2, 1, 3).to(q.dtype)
        return reciprocal_attention(q, k, v, self.mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-directory', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--runs', type=int, default=5)
    args = p.parse_args()
    if min(args.threads, args.runs) < 1:
        p.error('threads and runs must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
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
        from experiments.tenstorrent.attention_diagnostic import AttentionMath
        from experiments.tenstorrent.precision_reference import PipelinePrecisionMode
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(args.threads)
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name('inference_attention.py'))}
        origin = json.loads((args.fixture_directory / 'result.json').read_text())
        path = args.fixture_directory / 'fixtures.pt'
        if (origin['status'] != 'completed_isolated_attention_diagnostic' or
                sha256(path) != origin['artifacts_sha256']['fixtures.pt']):
            raise ValueError('Source QKV fixtures were not sealed by an accepted experiment')
        fixtures = torch.load(path, weights_only=True, map_location='cpu')
        digests = {name: [tensor_digest(v) for v in values] for name, values in fixtures.items()}
        if digests != origin['fixture_digests']:
            raise ValueError('Fixture tensor digests changed')
        report.update(fixture_sha256=sha256(path), fixture_digests=digests,
                      fixture_model=origin['model'], cases={})
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert xr.device_type() == 'TT'
        os.chdir(args.output_dir)
        outputs = {}
        for name, values in fixtures.items():
            with torch.no_grad():
                reference = AttentionMath(fp32=True)(*values)
            device_inputs = tuple(v.to(device) for v in values)
            report['cases'][name] = {}
            for mode in ('legacy', *ATTENTION_MODES):
                label = name + '_' + mode
                phase(label)
                torch.compiler.reset()
                options = {'optimization_level': 0, 'dry_run': 'false',
                           'export_path': str(args.output_dir / (label + '-native-ir'))}
                torch_xla.set_custom_compile_options(options)
                module = PrecisionAttention(mode).eval()
                with torch.no_grad(), PipelinePrecisionMode():
                    cpu = module(*values)
                compiled = torch.compile(module, backend='tt', fullgraph=True)
                metrics.clear_all()
                with torch.no_grad():
                    actual = compiled(*device_inputs).cpu()
                    compiled(*device_inputs).cpu()
                    times = []
                    for _ in range(args.runs):
                        start = time.perf_counter()
                        repeated = compiled(*device_inputs).cpu()
                        times.append(time.perf_counter() - start)
                        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
                result = {'requested_compile_options': options, 'warm_seconds': times,
                          'tt_vs_cpu_reference': compare_tensors(actual, reference),
                          'tt_vs_same_precision_cpu': compare_tensors(actual, cpu),
                          'cpu_precision_vs_reference': compare_tensors(cpu, reference),
                          'fallback_ops': list(metrics.executed_fallback_ops()),
                          'aten_counters': {k: metrics.counter_value(k) for k in metrics.counter_names() if k.startswith('aten::')}}
                report['cases'][name][mode] = result
                outputs[label] = actual
                (args.output_dir / (label + '-metrics.txt')).write_text(metrics.metrics_report())
                if result['fallback_ops'] or result['aten_counters']:
                    raise ValueError('Unexpected CPU fallback')
            exact = torch.equal(outputs[name + '_legacy'], outputs[name + '_fp32_reciprocal'])
            report['cases'][name]['refactored_baseline_bitwise'] = exact
            if not exact:
                raise ValueError('Arithmetic extraction changed native baseline output')
        torch.save(outputs, args.output_dir / 'outputs.pt')
        report.update(outputs_sha256=sha256(args.output_dir / 'outputs.pt'),
                      status='completed_attention_precision_diagnostic')
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
