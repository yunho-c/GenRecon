"""Measure FP32 matrix-reduction error with exactly BF16-representable inputs.

Seed62 matches the earlier matrix probe. Native storage defaults to FP32;
--matmul-mode tests BF16 sources with either ordinary BF16 or explicit FP32 output.
Reduction mode varies dot length using prefixes of the same original data.
Active-terms mode retains one K2 matrix graph and changes only input values,
isolating the two products and reversing their order without recompilation.
Mantissa-split mode separates each BF16-representable operand into a six-fraction-
bit high part and an exact residual on CPU, then measures each native product.
Power-two mode measures 1 +/- 2**(-s), s=0..31, through one unchanged K2 graph.
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
    from experiments.tenstorrent.native_environment import reexec_clean_native_process
    reexec_clean_native_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('reduction', 'active_terms', 'mantissa_split', 'power_two'), default='reduction')
    parser.add_argument('--matmul-mode', choices=('fp32', 'bf16', 'bf16_fp32'), default='fp32')
    parser.add_argument('--lengths', type=int, nargs='+', choices=(1, 2, 4, 8, 16, 32, 64, 128),
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'seed': 62, 'mode': args.mode,
              'matmul_mode': args.matmul_mode, 'cases': {},
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
        class Float32OutputProduct(torch.nn.Module):
            def forward(self, a, b):
                return torch.mm(a, b, out_dtype=torch.float32)
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), Path(__file__).with_name('fp32_matmul_probe.py'), Path(__file__).with_name('native_environment.py'))}
        torch.set_num_threads(4)
        torch.manual_seed(62)
        left = torch.randn(512, 128).bfloat16().float()
        right = torch.randn(128, 512).bfloat16().float()
        if args.mode == 'reduction':
            cases = {str(length): (left[:, :length].contiguous(), right[:length].contiguous())
                     for length in args.lengths}
        elif args.mode == 'active_terms':
            a, b = left[:, :2].contiguous(), right[:2].contiguous()
            a0, a1, b0, b1 = a.clone(), a.clone(), b.clone(), b.clone()
            a0[:, 1] = 0; a1[:, 0] = 0; b0[1] = 0; b1[0] = 0
            cases = {'full_k2': (a, b), 'first_only': (a, b0), 'second_only': (a, b1),
                     'first_only_zero_a': (a0, b), 'second_only_zero_a': (a1, b),
                     'reversed_k2': (a.flip(-1).contiguous(), b.flip(0).contiguous()),
                     'positive_products': (a.abs(), b.abs()),
                     'integer_values': ((a*16).round(), (b*16).round())}
        elif args.mode == 'power_two':
            shifts = torch.arange(512) % 32
            a = torch.stack((torch.ones(512), torch.pow(2., -shifts.float())), dim=1)
            b = torch.stack((torch.ones(512), torch.cat((torch.ones(256), -torch.ones(256)))))
            first, second = b.clone(), b.clone()
            first[1] = 0; second[0] = 0
            negative = b.clone(); negative[0] = -1
            cases = {'full_k2': (a, b), 'first_only': (a, first), 'second_only': (a, second),
                     'reversed_k2': (a.flip(-1).contiguous(), b.flip(0).contiguous()),
                     'negative_base': (a, negative), 'scaled_up': (a*65536, b), 'scaled_down': (a/65536, b)}
        else:
            cases = {}
            for length in args.lengths:
                a, b = left[:, :length].contiguous(), right[:length].contiguous()
                ah = (a.view(torch.int32) & -131072).view(torch.float32)
                bh = (b.view(torch.int32) & -131072).view(torch.float32)
                al, bl = a-ah, b-bh
                torch.testing.assert_close(ah+al, a, rtol=0, atol=0)
                torch.testing.assert_close(bh+bl, b, rtol=0, atol=0)
                prefix = 'k'+str(length)
                cases.update({prefix+'_full': (a, b), prefix+'_left_hi': (ah, b),
                              prefix+'_left_lo': (al, b), prefix+'_right_hi': (a, bh), prefix+'_right_lo': (a, bl)})
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        outputs = {}
        compiled = None
        compiled_shapes = None
        for name, (a, b) in cases.items():
            label = 'k'+name if args.mode == 'reduction' else name
            phase(label)
            torch.testing.assert_close(a, a.bfloat16().float(), rtol=0, atol=0)
            torch.testing.assert_close(b, b.bfloat16().float(), rtol=0, atol=0)
            expected = a@b
            exact = (a.double()@b.double()).float()
            shapes = (tuple(a.shape), tuple(b.shape))
            if compiled is None or args.mode == 'reduction' or shapes != compiled_shapes:
                torch.compiler.reset()
                directory = args.output_dir/(label+'-native-ir')
                torch_xla.set_custom_compile_options({'optimization_level': 0, 'math_fidelity': 'hifi4',
                                                      'dry_run': 'false', 'export_path': str(directory)})
                module = Float32OutputProduct() if args.matmul_mode == 'bf16_fp32' else MatrixProduct()
                compiled = torch.compile(module.eval(), backend='tt', fullgraph=True)
                compiled_shapes = shapes
            dtype = torch.float32 if args.matmul_mode == 'fp32' else torch.bfloat16
            resident = a.to(dtype).to(device), b.to(dtype).to(device)
            if not all(torch.equal(value.cpu().float(), original) for value, original in zip(resident, (a, b), strict=True)):
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
                   'native_input_dtypes': [str(item.dtype) for item in resident],
                   'input_roundtrip_bitwise': True, 'native_repeats_bitwise': True,
                   'matrix_ops': matrix_ops, 'runtime_ir_sha256': sha256(path),
                   'executed_fallback_ops': list(metrics.executed_fallback_ops()),
                   'aten_counters': {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}}
            row['dynamo_before'] = before
            row['dynamo_after'] = dict(torch._dynamo.utils.counters['stats'])
            row['xla_compile_samples'] = {key: metrics.metric_data(key)[0] for key in metrics.metric_names() if 'Compile' in key and 'Cached' not in key}
            report['cases'][name] = row
            value_only_case = (args.mode in ('active_terms', 'power_two') and name != 'full_k2') or (
                args.mode == 'mantissa_split' and not name.endswith('_full'))
            if value_only_case and (
                    row['dynamo_before'] != row['dynamo_after'] or any(row['xla_compile_samples'].values())):
                raise ValueError('Changing only active terms recompiled the matrix graph')
            if row['executed_fallback_ops'] or row['aten_counters']:
                raise ValueError('Reduction diagnostic used fallback')
            if any('math_fidelity = hifi4' not in op for op in matrix_ops):
                raise ValueError('Native matrix arithmetic does not retain requested HiFi4')
            if args.matmul_mode == 'fp32' and any('fp32_dest_acc_en = true' not in op or 'xbf16' in op for op in matrix_ops):
                raise ValueError('Native matrix arithmetic does not retain requested FP32 storage/accumulation')
            if args.matmul_mode == 'bf16_fp32' and (value.dtype != torch.float32 or any(
                    'fp32_dest_acc_en = true' not in op or op.count('xbf16') < 2 or
                    not re.search(r'-> tensor<[^>]*xf32', op) for op in matrix_ops)):
                raise ValueError('Explicit BF16-source/FP32-output policy was not retained in native IR')
            outputs[name] = value
            (args.output_dir/(label+'-metrics.txt')).write_text(metrics.metrics_report())
        if args.mode in ('active_terms', 'power_two'):
            isolated = outputs['first_only']+outputs['second_only']
            report['controls'] = {
                'native_full_vs_cpu_sum_of_isolated_native_products': compare_tensors(outputs['full_k2'], isolated),
                'isolated_native_sum_vs_cpu_full': compare_tensors(isolated, cases['full_k2'][0]@cases['full_k2'][1]),
                'full_vs_reversed_native': compare_tensors(outputs['full_k2'], outputs['reversed_k2'])}
            if args.mode == 'active_terms':
                report['controls'].update(
                    first_zero_operand_bitwise=torch.equal(outputs['first_only'], outputs['first_only_zero_a']),
                    second_zero_operand_bitwise=torch.equal(outputs['second_only'], outputs['second_only_zero_a']))
            else:
                report['controls']['common_scale_up_bitwise'] = torch.equal(outputs['scaled_up']/65536, outputs['full_k2'])
                report['controls']['common_scale_down_bitwise'] = torch.equal(outputs['scaled_down']*65536, outputs['full_k2'])
                report['exponent_grid'] = {}
                expected = cases['full_k2'][0]@cases['full_k2'][1]
                for shift in range(32):
                    mask = shifts == shift
                    report['exponent_grid'][str(shift)] = {
                        'cpu_positive': expected[shift, 0].item(), 'cpu_negative': expected[shift, 256].item(),
                        'native_positive': torch.unique(outputs['full_k2'][mask, :256]).tolist(),
                        'native_negative': torch.unique(outputs['full_k2'][mask, 256:]).tolist()}
        if args.mode == 'mantissa_split':
            report['controls'] = {}
            for length in args.lengths:
                prefix = 'k'+str(length)
                a, b = cases[prefix+'_full']
                expected = a@b
                report['controls'][str(length)] = {}
                for side in ('left', 'right'):
                    summed = outputs[prefix+'_'+side+'_hi']+outputs[prefix+'_'+side+'_lo']
                    high, low = cases[prefix+'_'+side+'_hi'], cases[prefix+'_'+side+'_lo']
                    cpu_formula = high[0]@high[1]+low[0]@low[1]
                    outputs[prefix+'_'+side+'_summed_on_cpu'] = summed
                    report['controls'][str(length)][side] = {
                        'native_partial_products_summed_on_cpu_vs_cpu_full': compare_tensors(summed, expected),
                        'cpu_partial_formula_vs_cpu_full': compare_tensors(cpu_formula, expected),
                        'native_partial_sum_vs_cpu_partial_formula': compare_tensors(summed, cpu_formula),
                        'native_partial_products_summed_on_cpu_bitwise_full': torch.equal(summed, expected)}
        torch.save(outputs, args.output_dir/'outputs.pt')
        report['outputs_sha256'] = sha256(args.output_dir/'outputs.pt')
        report['status'] = {'reduction': 'completed_native_reduction_diagnostic',
                            'active_terms': 'completed_native_active_term_control',
                            'mantissa_split': 'completed_native_mantissa_split_control',
                            'power_two': 'completed_native_power_two_control'}[args.mode]
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise


if __name__ == '__main__':
    main()
