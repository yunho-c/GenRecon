"""Validate an isolated profiling SDK against the sealed canonical K2 control.

The SDK's interpreter and native search paths are installed by process re-exec.
Only libraries under the recorded SDK workspace are accepted. Device-profile
success requires actual numeric CSV rows, not merely a profiler API or flag.
This is a one-operation capability control, not full-model performance evidence.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-workspace', type=Path, required=True)
    parser.add_argument('--native-control-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile-device', action='store_true')
    args = parser.parse_args()
    sdk = args.sdk_workspace.resolve()
    source, build, venv = sdk/'source', sdk/'build', sdk/'venv'
    libraries = str(build/'lib')+os.pathsep+str(build/'lib64')
    python_paths = os.pathsep.join(map(str, (source/'ttnn', source, source/'tools')))
    from experiments.tenstorrent.native_environment import clean_native_environment
    if (Path(sys.prefix).resolve() != venv or os.environ.get('LD_LIBRARY_PATH') != libraries or
            os.environ.get('PYTHONPATH') != python_paths or os.environ.get('TT_METAL_HOME') != str(source)):
        environment = clean_native_environment()
        environment.update(LD_LIBRARY_PATH=libraries, PYTHONPATH=python_paths,
                           TT_METAL_HOME=str(source), TT_METAL_RUNTIME_ROOT=str(source), VIRTUAL_ENV=str(venv))
        os.execve(str(venv/'bin/python'), [str(venv/'bin/python'), *sys.argv], environment)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(TT_METAL_LOGS_PATH=str(args.output_dir/'metal-logs'),
        TT_METAL_PROFILER_DIR=str(args.output_dir/'profiler'),
        TT_METAL_CACHE=str(args.output_dir/'metal-cache'), TT_METAL_INSPECTOR='1',
        TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES='1', ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    os.environ.pop('TT_METAL_DEVICE_PROFILER', None)
    if args.profile_device:
        os.environ.update(TT_METAL_DEVICE_PROFILER='1', TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY='1',
                          TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES='0')
    report = {'status': 'running', 'scope': __doc__, 'sdk_workspace': str(sdk),
              'profile_device': args.profile_device,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, flush=True)

    device = None
    phase('validate_sdk_and_fixture')
    try:
        from experiments.tenstorrent.experiment_identity import sha256
        state = json.loads((sdk/'build-result.json').read_text())
        if state.get('status') != 'built_unvalidated' or state.get('phase') != 'complete':
            raise ValueError('SDK build/install has not completed')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        if commit != state['recipe']['metal_commit'] or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip():
            raise ValueError('SDK source identity changed')
        report['sdk_build_result_sha256'] = sha256(sdk/'build-result.json')
        report['metal_commit'] = commit
        report['script_sha256'] = sha256(Path(__file__))
        import torch
        from experiments.tenstorrent.validation import compare_tensors
        # Match the sealed trajectory digest without importing the full GenRecon
        # model stack into the intentionally minimal profiling environment.
        def tensor_digest(value):
            value = value.detach().cpu().contiguous()
            digest = hashlib.sha256(f'{value.dtype}:{list(value.shape)}'.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        torch.set_num_threads(4)
        shifts = torch.arange(512) % 32
        a = torch.stack((torch.ones(512), torch.pow(2., -shifts.float())), dim=1)
        b = torch.stack((torch.ones(512), torch.cat((torch.ones(256), -torch.ones(256)))))
        expected = a@b
        control = json.loads((args.native_control_dir/'result.json').read_text())
        if (control['status'] != 'completed_native_power_two_control' or
                sha256(args.native_control_dir/'outputs.pt') != control['outputs_sha256'] or
                [tensor_digest(a), tensor_digest(b)] != control['cases']['full_k2']['input_digests']):
            raise ValueError('Sealed canonical control or fixture identity differs')
        reference = torch.load(args.native_control_dir/'outputs.pt', weights_only=True, map_location='cpu')['full_k2']
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        import ttnn
        paths = sorted({Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
                        if '/' in line and any(key in line for key in ('/_ttnn.so', '/_ttnncpp.so', 'libtt_metal.so'))})
        if not paths or not any(path.name == '_ttnn.so' for path in paths) or any(not path.is_relative_to(sdk) for path in paths):
            raise ValueError('Native TTNN/Metal libraries are not confined to the requested SDK')
        report['native_library_sha256'] = {str(path): sha256(path) for path in paths}
        report['ttnn_module'] = ttnn.__file__
        phase('open_profile_sdk_device')
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
        config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=True, packer_l1_acc=False, dst_full_sync_en=False)
        resident = tuple(ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in (a, b))
        if not all(torch.equal(ttnn.to_torch(value), original) for value, original in zip(resident, (a, b), strict=True)):
            raise ValueError('SDK upload changed canonical values')
        phase('native_matrix_control')
        outputs = []
        for _ in range(3):
            value = ttnn.matmul(*resident, dtype=ttnn.float32, compute_kernel_config=config)
            outputs.append(ttnn.to_torch(value))
            ttnn.deallocate(value)
        if not all(torch.equal(value, reference) for value in outputs):
            raise ValueError('SDK control does not reproduce canonical outputs bitwise')
        report['all_repeats_bitwise_canonical'] = True
        report['native_vs_cpu'] = compare_tensors(outputs[0], expected)
        torch.save(outputs[0], args.output_dir/'output.pt')
        report['output_sha256'] = sha256(args.output_dir/'output.pt')
        if args.profile_device:
            phase('flush_device_profiler')
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
        ttnn.close_mesh_device(device); device = None
        if args.profile_device:
            report['profile_csvs'] = {}
            for path in args.output_dir.rglob('*.csv'):
                rows = sum(bool(re.match(r'^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,', line)) for line in path.read_text().splitlines())
                if rows:
                    report['profile_csvs'][str(path.relative_to(args.output_dir))] = {'numeric_rows': rows, 'sha256': sha256(path)}
            if not report['profile_csvs']:
                raise ValueError('No actual device-profiler numeric CSV rows were produced')
        report['status'] = 'completed_profile_sdk_control'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == '__main__':
    main()
