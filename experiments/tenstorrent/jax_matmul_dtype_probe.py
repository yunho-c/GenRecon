"""Test BF16-source/FP32-result dot through the installed TT JAX PJRT frontend.

Same canonical native runtime, seed62 inputs and HiFi4 as the PyTorch probes.
This capability diagnostic explicitly verifies emitted native operand/result
types; JAX result dtype alone does not establish a mixed native matrix operation.
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
    parser.add_argument('--source-dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(JAX_PLATFORMS='tt', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'source_dtype': args.source_dtype,
              'seed': 62, 'cases': {}, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, flush=True)

    phase('imports_and_cpu_fixture')
    try:
        import torch
        import numpy as np
        import ml_dtypes
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        report['versions'] = {name: importlib.metadata.version(name) for name in
            ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt', 'jax', 'jaxlib')}
        report['script_sha256'] = sha256(Path(__file__))
        torch.set_num_threads(4)
        torch.manual_seed(62)
        left = torch.randn(512, 128).bfloat16().float()
        right = torch.randn(128, 512).bfloat16().float()
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        import jax
        import jax.numpy as jnp
        devices = jax.devices('tt')
        if len(devices) != 1:
            raise ValueError('Expected the exposed single-chip TT device')
        device = devices[0]
        report['device'] = str(device)
        report['device_platform'] = device.platform
        outputs = {}
        dtype = ml_dtypes.bfloat16 if args.source_dtype == 'bfloat16' else np.float32
        for length in (2, 32, 128):
            phase('k'+str(length))
            a, b = left[:, :length].contiguous(), right[:length].contiguous()
            expected = a@b
            exact = (a.double()@b.double()).float()
            resident = tuple(jax.device_put(np.asarray(value.numpy(), dtype=dtype), device) for value in (a, b))
            for value, original in zip(resident, (a, b), strict=True):
                if not np.array_equal(np.asarray(value).astype(np.float32), original.numpy()):
                    raise ValueError('JAX input roundtrip changed exact fixture values')
            options = {'optimization_level': 0, 'math_fidelity': 'hifi4', 'dry_run': 'false',
                       'export_path': str(args.output_dir/('k'+str(length)+'-native-ir'))}

            def dot(x, y):
                return jax.lax.dot_general(x, y, (((1,), (0,)), ((), ())),
                    precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

            compiled = jax.jit(dot, backend='tt', compiler_options=options)
            lowered = compiled.lower(*resident)
            (args.output_dir/('k'+str(length)+'-frontend-stablehlo.mlir')).write_text(lowered.as_text())
            executable = lowered.compile()
            result = executable(*resident)
            result.block_until_ready()
            if result.devices() != {device}:
                raise ValueError('JAX result did not execute on the selected TT device')
            actual = torch.from_numpy(np.asarray(result).astype(np.float32).copy())
            times = []
            for _ in range(3):
                start = time.perf_counter()
                repeat = executable(*resident)
                repeat.block_until_ready()
                readback = torch.from_numpy(np.asarray(repeat).astype(np.float32).copy())
                times.append(time.perf_counter()-start)
                torch.testing.assert_close(actual, readback, rtol=0, atol=0)
            path = next((Path(options['export_path'])/'irs').glob('ttnn_runtime_[0-9]*.mlir'))
            ops = [line.strip() for line in path.read_text().splitlines() if re.search(r'"ttnn\.(matmul|linear)"', line)]
            retained = len(ops) == 1 and str(result.dtype) == 'float32' and all(
                'math_fidelity = hifi4, fp32_dest_acc_en = true' in op and
                re.search(r'-> tensor<[^>]*xf32', op) and
                (op.count('xbf16') == 2 if args.source_dtype == 'bfloat16' else 'xbf16' not in op) for op in ops)
            report['cases'][str(length)] = {'input_shapes': [list(a.shape), list(b.shape)],
                'input_digests': [tensor_digest(a), tensor_digest(b)],
                'native_input_dtypes': [str(value.dtype) for value in resident], 'output_dtype': str(result.dtype),
                'input_roundtrip_bitwise': True, 'repeats_bitwise': True, 'warm_seconds': times,
                'native_vs_cpu_fp32': compare_tensors(actual, expected),
                'native_vs_float64_rounded': compare_tensors(actual, exact),
                'native_bitwise_cpu_fp32': torch.equal(actual, expected),
                'native_policy_retained': bool(retained), 'matrix_ops': ops,
                'runtime_ir_sha256': sha256(path), 'requested_compile_options': options}
            outputs[str(length)] = actual
        torch.save(outputs, args.output_dir/'outputs.pt')
        report['outputs_sha256'] = sha256(args.output_dir/'outputs.pt')
        report['status'] = 'completed_native_dtype_diagnostic' if all(
            row['native_policy_retained'] for row in report['cases'].values()) else 'native_dtype_policy_not_retained'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise


if __name__ == '__main__':
    main()
