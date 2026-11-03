"""Check actual native RoPE layouts before full-model experiments.

Seeded synthetic BF16 values, original adjacent-pair rotation, BHSD output.
Resident inputs/phases; output D2H and synchronization included in timings.
This kernel diagnostic establishes neither full-flow speed nor scene quality.
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
from experiments.tenstorrent.inference_rope import RealPairRoPE


class RoPELayoutMath(nn.Module):
    def __init__(self, phases, mode):
        super().__init__()
        self.mode = mode
        self.rope = RealPairRoPE(phases,
            head_first=mode in ('contiguous_head_first', 'flat_head_first', 'integrated_head_first'),
            flatten=mode == 'integrated_head_first')

    def forward(self, x):
        if self.mode == 'token_first':
            return self.rope(x).permute(0, 2, 1, 3)
        if self.mode != 'flat_token_first':
            x = x.permute(0, 2, 1, 3).contiguous()
        if self.mode == 'integrated_head_first':
            return self.rope(x)
        if self.mode == 'contiguous_head_first':
            partner = torch.index_select(x, -1, self.rope.partner_ids).float()
            return (x.float()*self.rope.cosine + partner*self.rope.signed_sine).to(x.dtype)
        if self.mode not in ('flat_head_first', 'flat_token_first'):
            raise ValueError('Unknown RoPE layout')
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        cosine = self.rope.cosine.expand(shape).reshape(-1, shape[-1])
        sine = self.rope.signed_sine.expand(shape).reshape(-1, shape[-1])
        partner = torch.index_select(flat, -1, self.rope.partner_ids).float()
        result = (flat.float()*cosine + partner*sine).to(x.dtype).reshape(shape)
        return result.permute(0, 2, 1, 3) if self.mode == 'flat_token_first' else result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, nargs='+', default=[512])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    args = p.parse_args()
    if args.runs < 1 or min(args.tokens) < 1:
        p.error('tokens and runs must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'seed': 57,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'cases': {}}

    def phase(name):
        report['phase'] = name
        (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, flush=True)

    phase('imports')
    try:
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from genrecon.modules.attention import RotaryPositionEmbedder
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(4)
        torch.manual_seed(report['seed'])
        report['versions'] = {k: importlib.metadata.version(k)
                              for k in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(__file__), Path(__file__).with_name('inference_rope.py')]}
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        outputs = {}
        for tokens in args.tokens:
            side = math.ceil(tokens**(1/3))
            ids = torch.arange(tokens)
            coords = torch.stack((ids//side**2, ids//side % side, ids % side), -1)
            phases = RotaryPositionEmbedder(128, 3)(coords)
            values = torch.randn(1, tokens, 12, 128).bfloat16()
            reference = RoPELayoutMath(phases, 'token_first')(values)
            report['cases'][str(tokens)] = {'input_digest': tensor_digest(values), 'phase_digest': tensor_digest(phases), 'modes': {}}
            device_input = values.to(device)
            for mode in ('token_first', 'contiguous_head_first', 'flat_head_first', 'flat_token_first', 'integrated_head_first'):
                label = str(tokens)+'_'+mode
                phase(label)
                module = RoPELayoutMath(phases, mode).eval()
                torch.testing.assert_close(module(values), reference, rtol=0, atol=0)
                torch.compiler.reset()
                options = {'optimization_level': 0, 'dry_run': 'false',
                           'export_path': str(args.output_dir / (label+'-native-ir'))}
                torch_xla.set_custom_compile_options(options)
                module.to(device)
                compiled = torch.compile(module, backend='tt', fullgraph=True)
                metrics.clear_all()
                with torch.no_grad():
                    actual = compiled(device_input).cpu()
                    compiled(device_input).cpu()
                    times = []
                    for _ in range(args.runs):
                        start = time.perf_counter()
                        repeated = compiled(device_input).cpu()
                        times.append(time.perf_counter()-start)
                        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
                result = {'warm_seconds': times, 'requested_compile_options': options,
                          'cpu_formula_bitwise': True, 'tt_vs_cpu': compare_tensors(actual, reference),
                          'fallback_ops': list(metrics.executed_fallback_ops()),
                          'aten_counters': {k: metrics.counter_value(k) for k in metrics.counter_names() if k.startswith('aten::')}}
                outputs[label] = actual
                if mode != 'token_first':
                    result['tt_vs_token_first'] = compare_tensors(actual, outputs[str(tokens)+'_token_first'])
                    result['tt_bitwise_token_first'] = torch.equal(actual, outputs[str(tokens)+'_token_first'])
                if mode == 'integrated_head_first':
                    result['integrated_bitwise_prototype'] = torch.equal(actual, outputs[str(tokens)+'_flat_head_first'])
                    if not result['integrated_bitwise_prototype']:
                        raise ValueError('Integrated helper differs from prototype')
                report['cases'][str(tokens)]['modes'][mode] = result
                (args.output_dir/(label+'-metrics.txt')).write_text(metrics.metrics_report())
                if result['fallback_ops'] or result['aten_counters']:
                    raise ValueError('Unexpected CPU fallback')
        torch.save(outputs, args.output_dir/'outputs.pt')
        report.update(status='completed_rope_layout_probe', outputs_sha256=sha256(args.output_dir/'outputs.pt'))
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
