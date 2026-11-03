"""Observe FP32 attention stages and compare each on identical input values.

Capturing intermediates is accepted only if final TT output matches ordinary
FP32 postscale attention bitwise. Teacher-forced CPU suboperations consume TT
intermediates to separate local arithmetic error from propagation. No latency,
CUDA equivalence or full-flow quality claim.
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
import traceback

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class AttentionStages(nn.Module):
    def forward(self, q, k, v):
        a, b, c = [x.float().permute(0, 2, 1, 3) for x in (q, k, v)]
        product = a @ b.transpose(-2, -1)
        scaled = product * (1 / math.sqrt(q.shape[-1]))
        probabilities = torch.softmax(scaled, dim=-1)
        raw = probabilities @ c
        return raw.permute(0, 2, 1, 3).to(q.dtype), product, scaled, probabilities, raw


def mantissa_stats(value):
    if value.dtype != torch.float32 or value.device.type != 'cpu':
        raise ValueError('Expected FP32 CPU intermediate')
    mask = torch.isfinite(value) & (value != 0)
    bits = value.contiguous().view(torch.int32)[mask]
    return {'nonzero_finite': bits.numel(), 'finite': bool(torch.isfinite(value).all()),
            **{f'low_{count}_bits_zero_fraction': ((bits & ((1 << count) - 1)) == 0).float().mean().item()
               if bits.numel() else None for count in (13, 16)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-directory', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--math-fidelity', choices=('hifi3', 'hifi4'), default='hifi4')
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'requested_compile_options': {
        'optimization_level': 0, 'math_fidelity': args.math_fidelity},
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
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(args.threads)
        report['versions'] = {key: importlib.metadata.version(key) for key in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name('attention_diagnostic.py'),
            ROOT / 'genrecon/modules/attention/full_attn.py')}
        origin = json.loads((args.fixture_directory / 'result.json').read_text())
        path = args.fixture_directory / 'fixtures.pt'
        if sha256(path) != origin['artifacts_sha256']['fixtures.pt']:
            raise ValueError('Fixture artifact changed')
        fixtures = torch.load(path, weights_only=True, map_location='cpu')
        if {k: [tensor_digest(x) for x in v] for k, v in fixtures.items()} != origin['fixture_digests']:
            raise ValueError('Fixture tensor identity changed')
        report['fixture_sha256'] = sha256(path)
        report['fixture_report_sha256'] = sha256(args.fixture_directory / 'result.json')
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert xr.device_type() == 'TT'
        torch_xla.set_custom_compile_options(report['requested_compile_options'])
        os.chdir(args.output_dir)
        report['cases'], bundles = {}, {}
        names = ('final', 'qk_product', 'scaled_scores', 'probabilities', 'av_product')
        for name, qkv in fixtures.items():
            phase(name)
            device_inputs = tuple(x.to(device) for x in qkv)
            metrics.clear_all()
            plain = torch.compile(AttentionMath(True, True).eval(), backend='tt', fullgraph=True)
            observer = torch.compile(AttentionStages().eval(), backend='tt', fullgraph=True)
            with torch.no_grad():
                cpu = AttentionStages()(*qkv)
                original = AttentionMath(True, True)(*qkv)
                torch.testing.assert_close(cpu[0], original, rtol=0, atol=0)
                baseline = plain(*device_inputs).cpu()
                actual = tuple(x.cpu() for x in observer(*device_inputs))
                repeated = tuple(x.cpu() for x in observer(*device_inputs))
                teacher = (None, cpu[1], actual[1] * (1 / math.sqrt(qkv[0].shape[-1])),
                           torch.softmax(actual[2], dim=-1), actual[3] @ qkv[2].float().permute(0, 2, 1, 3))
            result = {'captured_final_exact_to_ordinary': torch.equal(actual[0], baseline),
                      'all_repeated_states_exact': all(torch.equal(a, b) for a, b in zip(actual, repeated, strict=True)),
                      'vs_cpu': {key: compare_tensors(a, b) for key, a, b in zip(names, actual, cpu, strict=True)},
                      'local_same_input_error': {key: compare_tensors(actual[i], teacher[i]) for i, key in enumerate(names) if i},
                      'mantissa_statistics': {key: {'tt': mantissa_stats(actual[i]), 'cpu': mantissa_stats(cpu[i])}
                                              for i, key in enumerate(names) if i},
                      'fallback_ops': list(metrics.executed_fallback_ops()),
                      'aten_counters': {key: metrics.counter_value(key) for key in metrics.counter_names() if key.startswith('aten::')}}
            report['cases'][name] = result
            bundles[name] = {'tt': actual, 'cpu': cpu}
            (args.output_dir / (name + '-metrics.txt')).write_text(metrics.metrics_report())
        torch.save(bundles, args.output_dir / 'states.pt')
        report['states_sha256'] = sha256(args.output_dir / 'states.pt')
        accepted = all(c['captured_final_exact_to_ordinary'] and c['all_repeated_states_exact'] and
                       not c['fallback_ops'] and not c['aten_counters'] for c in report['cases'].values())
        report['status'] = 'completed_stage_diagnostic' if accepted else 'rejected_observation_changed_execution'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
