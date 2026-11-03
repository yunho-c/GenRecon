"""Direct canonical-wheel TTNN kernel controls for the exact K2 arithmetic case.

Fresh process per variant. Native library paths and wheel RECORD hashes are
verified before opening the device. Inspector records and an isolated kernel
cache bind each result to generated descriptors, compiler arguments and ELFs.
This is a kernel capability diagnostic, not a sampler performance comparison.
"""
import argparse
import base64
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from experiments.tenstorrent.native_environment import reexec_clean_native_process
    reexec_clean_native_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('default', 'full_sync', 'packer', 'precise', 'hifi3', 'hifi2', 'fp16_dest'), default='default')
    parser.add_argument('--native-control-dir', type=Path, required=True)
    parser.add_argument('--profile-device', action='store_true', help='Probe device-only profiler on standalone canonical TTNN')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(TT_METAL_LOGS_PATH=str(args.output_dir/'metal-logs'),
        TT_METAL_CACHE=str(args.output_dir/'metal-cache'), TT_METAL_INSPECTOR='1',
        TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES='1', TT_METAL_INSPECTOR_CAPTURE_TENSOR_SPECS='1',
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS='1', ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    os.environ.pop('TT_METAL_DEVICE_PROFILER', None)
    if args.profile_device:
        os.environ.update(TT_METAL_DEVICE_PROFILER='1', TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY='1')
    report = {'status': 'running', 'variant': args.variant, 'scope': __doc__,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'clean_native_process_start': True, 'profile_device': args.profile_device}

    def phase(name):
        report['phase'] = name
        (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, flush=True)

    device = None
    phase('imports_and_fixture')
    try:
        import torch
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(4)
        shifts = torch.arange(512) % 32
        a = torch.stack((torch.ones(512), torch.pow(2., -shifts.float())), dim=1)
        b = torch.stack((torch.ones(512), torch.cat((torch.ones(256), -torch.ones(256)))))
        expected = a@b
        source = json.loads((args.native_control_dir/'result.json').read_text())
        if source.get('status') != 'completed_native_power_two_control':
            raise ValueError('Requires the completed sealed original power-two native control')
        path = args.native_control_dir/'outputs.pt'
        if sha256(path) != source['outputs_sha256']:
            raise ValueError('Native control output seal changed')
        if [tensor_digest(a), tensor_digest(b)] != source['cases']['full_k2']['input_digests']:
            raise ValueError('Exact native-control inputs changed')
        reference = torch.load(path, weights_only=True, map_location='cpu')['full_k2']
        report['native_control'] = {'directory': str(args.native_control_dir.resolve()),
            'outputs_sha256': sha256(path), 'inputs_identical': True}
        report['source_sha256'] = sha256(Path(__file__))
        report['native_environment_sha256'] = sha256(Path(__file__).with_name('native_environment.py'))
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        import ttnn
        distribution = importlib.metadata.distribution('pjrt-plugin-tt')
        plugin = Path(distribution.locate_file('pjrt_plugin_tt')).resolve()
        mapped = sorted({Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                         if '/' in line and any(key in line for key in ('libtt_metal.so', '/_ttnn.so', '/_ttnncpp.so'))})
        records = {Path(distribution.locate_file(item)).resolve(): item for item in distribution.files}
        report['native_library_audit'] = {}
        for path in mapped:
            if not path.is_relative_to(plugin) or path not in records or records[path].hash is None:
                raise ValueError('Native library does not belong to canonical plugin: '+str(path))
            digest = sha256(path)
            record = records[path].hash
            if record.mode != 'sha256' or base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=') != record.value:
                raise ValueError('Canonical native library RECORD mismatch: '+str(path))
            report['native_library_audit'][str(path)] = {'sha256': digest, 'matches_wheel_record': True}
        if len(mapped) < 3:
            raise ValueError('Required TTNN/native Metal libraries were not bound')
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'tt-forge', 'pjrt-plugin-tt')}
        settings = {'math_fidelity': 'HiFi4', 'math_approx_mode': True, 'fp32_dest_acc_en': True,
                    'packer_l1_acc': False, 'dst_full_sync_en': False}
        if args.variant == 'full_sync': settings['dst_full_sync_en'] = True
        if args.variant == 'packer': settings['packer_l1_acc'] = True
        if args.variant == 'precise': settings['math_approx_mode'] = False
        if args.variant == 'hifi3': settings['math_fidelity'] = 'HiFi3'
        if args.variant == 'hifi2': settings['math_fidelity'] = 'HiFi2'
        if args.variant == 'fp16_dest': settings['fp32_dest_acc_en'] = False
        report['requested_kernel_settings'] = settings
        config = ttnn.WormholeComputeKernelConfig(**{**settings,
            'math_fidelity': getattr(ttnn.MathFidelity, settings['math_fidelity'])})
        report['native_config'] = str(config)
        phase('open_canonical_device')
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        resident = tuple(ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in (a, b))
        if not all(torch.equal(ttnn.to_torch(value), original) for value, original in zip(resident, (a, b), strict=True)):
            raise ValueError('Canonical TTNN upload changed original values')
        phase('native_matmul')
        outputs = []
        for _ in range(3):
            native = ttnn.matmul(*resident, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 compute_kernel_config=config)
            actual = ttnn.to_torch(native)
            if native.dtype != ttnn.float32 or actual.dtype != torch.float32:
                raise ValueError('Native output is not FP32')
            outputs.append(actual)
            ttnn.deallocate(native)
        if not all(torch.equal(value, outputs[0]) for value in outputs):
            raise ValueError('Native repeat outputs changed')
        actual = outputs[0]
        report['input_roundtrip_bitwise'] = True
        report['repeats_bitwise'] = True
        report['native_vs_cpu'] = compare_tensors(actual, expected)
        report['native_vs_pjrt_control'] = compare_tensors(actual, reference)
        report['native_bitwise_cpu'] = torch.equal(actual, expected)
        report['native_bitwise_pjrt_control'] = torch.equal(actual, reference)
        torch.save(actual, args.output_dir/'output.pt')
        report['output_sha256'] = sha256(args.output_dir/'output.pt')
        report['exponent_grid'] = {str(shift): {'cpu_positive': expected[shift, 0].item(),
            'cpu_negative': expected[shift, 256].item(),
            'native_positive': torch.unique(actual[shifts == shift, :256]).tolist(),
            'native_negative': torch.unique(actual[shifts == shift, 256:]).tolist()} for shift in range(32)}
        if args.profile_device:
            phase('flush_device_profiler')
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
        ttnn.close_mesh_device(device)
        device = None
        report['status'] = 'completed_canonical_kernel_control'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == '__main__':
    main()
