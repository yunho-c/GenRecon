"""Measure FP32 matrix-reduction error with exactly BF16-representable inputs.

Seed62 matches the earlier matrix probe. All native storage remains FP32;
Reduction mode varies dot length using prefixes of the same original data.
Active-terms mode retains one K2 matrix graph and changes only input values,
isolating the two products and reversing their order without recompilation.
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
    parser.add_argument('--mode', choices=('reduction', 'active_terms'), default='reduction')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'seed': 62, 'mode': args.mode, 'cases': {},
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
        if args.mode == 'reduction':
            cases = {str(length): (left[:, :length].contiguous(), right[:length].contiguous())
                     for length in (1, 2, 4, 8, 16, 32, 64, 128)}
        else:
            a, b = left[:, :2].contiguous(), right[:2].contiguous()
            a0, a1, b0, b1 = a.clone(), a.clone(), b.clone(), b.clone()
            a0[:, 1] = 0; a1[:, 0] = 0; b0[1] = 0; b1[0] = 0
            cases = {'full_k2': (a, b), 'first_only': (a, b0), 'second_only': (a, b1),
                     'first_only_zero_a': (a0, b), 'second_only_zero_a': (a1, b),
                     'reversed_k2': (a.flip(-1).contiguous(), b.flip(0).contiguous()),
                     'positive_products': (a.abs(), b.abs()),
                     'integer_values': ((a*16).round(), (b*16).round())}
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        outputs = {}
        compiled = None
        for name, (a, b) in cases.items():
            label = 'k'+name if args.mode == 'reduction' else name
            phase(label)
            torch.testing.assert_close(a, a.bfloat16().float(), rtol=0, atol=0)
            torch.testing.assert_close(b, b.bfloat16().float(), rtol=0, atol=0)
            expected = a@b
            exact = (a.double()@b.double()).float()
            if compiled is None or args.mode == 'reduction':
                torch.compiler.reset()
                directory = args.output_dir/(label+'-native-ir')
                torch_xla.set_custom_compile_options({'optimization_level': 0, 'math_fidelity': 'hifi4',
                                                      'dry_run': 'false', 'export_path': str(directory)})
                compiled = torch.compile(MatrixProduct().eval(), backend='tt', fullgraph=True)
            resident = a.to(device), b.to(device)
            if not all(torch.equal(value.cpu(), original) for value, original in zip(resident, (a, b), strict=True)):
                raise ValueError('Input roundtrip altered exact BF16-representable values')
            metrics.clear_all()
            before = dict(torch._dynamo.utils.counters['stats'])
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
            row['dynamo_before'] = before
            row['dynamo_after'] = dict(torch._dynamo.utils.counters['stats'])
            row['xla_compile_samples'] = {key: metrics.metric_data(key)[0] for key in metrics.metric_names() if 'Compile' in key and 'Cached' not in key}
            report['cases'][name] = row
            if args.mode == 'active_terms' and name != 'full_k2' and (
                    row['dynamo_before'] != row['dynamo_after'] or any(row['xla_compile_samples'].values())):
                raise ValueError('Changing only active terms recompiled the matrix graph')
            if row['executed_fallback_ops'] or row['aten_counters']:
                raise ValueError('Reduction diagnostic used fallback')
            if any('math_fidelity = hifi4, fp32_dest_acc_en = true' not in op or 'xbf16' in op for op in matrix_ops):
                raise ValueError('Native matrix arithmetic does not retain the requested FP32/HiFi4 configuration')
            outputs[name] = value
            (args.output_dir/(label+'-metrics.txt')).write_text(metrics.metrics_report())
        if args.mode == 'active_terms':
            isolated = outputs['first_only']+outputs['second_only']
            report['controls'] = {
                'native_full_vs_cpu_sum_of_isolated_native_products': compare_tensors(outputs['full_k2'], isolated),
                'isolated_native_sum_vs_cpu_full': compare_tensors(isolated, cases['full_k2'][0]@cases['full_k2'][1]),
                'full_vs_reversed_native': compare_tensors(outputs['full_k2'], outputs['reversed_k2']),
                'first_zero_operand_bitwise': torch.equal(outputs['first_only'], outputs['first_only_zero_a']),
                'second_zero_operand_bitwise': torch.equal(outputs['second_only'], outputs['second_only_zero_a'])}
        torch.save(outputs, args.output_dir/'outputs.pt')
        report['outputs_sha256'] = sha256(args.output_dir/'outputs.pt')
        report['status'] = 'completed_native_reduction_diagnostic' if args.mode == 'reduction' else 'completed_native_active_term_control'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise


if __name__ == '__main__':
    main()
