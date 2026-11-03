"""Measure FP32 matrix-reduction error with exactly BF16-representable inputs.

Seed62 matches the earlier matrix probe. All native storage remains FP32;
only the reduction length changes, using prefixes of the same original data.
This is a synthetic mechanism diagnostic, not learned-model attribution.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'seed': 62, 'cases': {},
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(value):
        report['phase'] = value
        (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(value, flush=True)

    phase('imports')
    try:
        import torch
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from experiments.tenstorrent.fp32_matmul_probe import MatrixProduct
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), Path(__file__).with_name('fp32_matmul_probe.py'))}
        torch.set_num_threads(4)
        torch.manual_seed(62)
        left = torch.randn(512, 128).bfloat16().float()
        right = torch.randn(128, 512).bfloat16().float()
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        outputs = {}
        for length in (1, 2, 4, 8, 16, 32, 64, 128):
            phase('k'+str(length))
            a, b = left[:, :length].contiguous(), right[:length].contiguous()
            torch.testing.assert_close(a, a.bfloat16().float(), rtol=0, atol=0)
            torch.testing.assert_close(b, b.bfloat16().float(), rtol=0, atol=0)
            expected = a@b
            exact = (a.double()@b.double()).float()
            torch.compiler.reset()
            directory = args.output_dir/('k'+str(length)+'-native-ir')
            torch_xla.set_custom_compile_options({'optimization_level': 0, 'math_fidelity': 'hifi4',
                                                  'dry_run': 'false', 'export_path': str(directory)})
            compiled = torch.compile(MatrixProduct().eval(), backend='tt', fullgraph=True)
            resident = a.to(device), b.to(device)
            if not all(torch.equal(value.cpu(), original) for value, original in zip(resident, (a, b), strict=True)):
                raise ValueError('Input roundtrip altered exact BF16-representable values')
            metrics.clear_all()
            with torch.no_grad():
                value = compiled(*resident).cpu()
                latencies = []
                for _ in range(3):
                    start = time.perf_counter()
                    repeat = compiled(*resident).cpu()
                    latencies.append(time.perf_counter()-start)
                    torch.testing.assert_close(value, repeat, rtol=0, atol=0)
            path = next((directory/'irs').glob('ttnn_runtime_[0-9]*.mlir'))
            matrix_ops = [line.strip() for line in path.read_text().splitlines() if re.search(r'"ttnn\.(matmul|linear)"', line)]
            row = {'input_shapes': [list(a.shape), list(b.shape)], 'input_digests': [tensor_digest(a), tensor_digest(b)],
                   'native_vs_cpu_fp32': compare_tensors(value, expected), 'native_vs_cpu_float64_rounded': compare_tensors(value, exact),
                   'cpu_fp32_vs_float64_rounded': compare_tensors(expected, exact),
                   'native_bitwise_cpu_fp32': torch.equal(value, expected), 'native_bitwise_float64_rounded': torch.equal(value, exact),
                   'output_dtype': str(value.dtype), 'warm_seconds': latencies,
                   'input_roundtrip_bitwise': True, 'native_repeats_bitwise': True,
                   'matrix_ops': matrix_ops, 'runtime_ir_sha256': sha256(path),
                   'executed_fallback_ops': list(metrics.executed_fallback_ops()),
                   'aten_counters': {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}}
            report['cases'][str(length)] = row
            if row['executed_fallback_ops'] or row['aten_counters']:
                raise ValueError('Reduction diagnostic used fallback')
            if any('math_fidelity = hifi4, fp32_dest_acc_en = true' not in op or 'xbf16' in op for op in matrix_ops):
                raise ValueError('Native matrix arithmetic does not retain the requested FP32/HiFi4 configuration')
            outputs[str(length)] = value
            (args.output_dir/('k'+str(length)+'-metrics.txt')).write_text(metrics.metrics_report())
        torch.save(outputs, args.output_dir/'outputs.pt')
        report['outputs_sha256'] = sha256(args.output_dir/'outputs.pt')
        report['status'] = 'completed_native_reduction_diagnostic'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise


if __name__ == '__main__':
    main()
