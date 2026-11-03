"""Run a sealed generated denoiser on the isolated matching-revision SDK.

Every run requires a canonical generated-execution control. Profiling includes
marked warm forwards, with uploads and profiler flushing outside each window.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-workspace', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--native-control-result', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--repair-embedding-indices', action='store_true')
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('runs must be positive')
    from experiments.tenstorrent.native_environment import clean_native_environment
    from experiments.tenstorrent.experiment_identity import sha256
    sdk, output = args.sdk_workspace.resolve(), args.output_dir.resolve()
    source, build, venv = sdk/'source', sdk/'build', sdk/'venv'
    state = json.loads((sdk/'build-result.json').read_text())
    if state.get('status') != 'built_unvalidated' or state.get('phase') != 'complete':
        raise ValueError('SDK build/install is incomplete')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if commit != state['recipe']['metal_commit'] or subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip():
        raise ValueError('SDK source changed')
    for name, digest in state['native_artifact_sha256'].items():
        if sha256(build/name) != digest:
            raise ValueError(f'SDK build artifact changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    environment = clean_native_environment()
    for name in list(environment):
        if name.startswith('TT_METAL_PROFILER') or name in ('TT_METAL_DEVICE_PROFILER', 'TTNN_OP_PROFILER'):
            environment.pop(name)
    environment.update(LD_LIBRARY_PATH=os.pathsep.join(map(str, (build/'lib', build/'lib64'))),
        PYTHONPATH=os.pathsep.join(map(str, (source/'ttnn', source, source/'tools'))),
        TT_METAL_HOME=str(source), TT_METAL_RUNTIME_ROOT=str(source), VIRTUAL_ENV=str(venv),
        TT_METAL_LOGS_PATH=str(output/'metal-logs'), TT_METAL_PROFILER_DIR=str(output/'profiler'),
        PATH=str(venv/'bin')+os.pathsep+environment.get('PATH', ''))
    command = [str(venv/'bin/python')]
    if args.profile:
        command += ['-m', 'tracy', '-r', '--op-support-count', '10000',
                    '--tracy-tools-folder', str(build/'tools/profiler/bin'),
                    '-o', str(output/'profiler'), '--check-exit-code']
    command += [str(ROOT/'experiments/tenstorrent/codegen_probe.py'), 'execute',
        '--directory', str(args.export_dir.resolve()), '--result', str(output/'execute.json'),
        '--native-control-result', str(args.native_control_result.resolve()), '--runs', str(args.runs)]
    if args.repair_embedding_indices:
        command += ['--repair-embedding-indices']
    if args.profile:
        command += ['--profile-markers', '--flush-device-profiler']
    report = {'status': 'running', 'command': command, 'metal_commit': commit,
        'sdk_build_result_sha256': sha256(sdk/'build-result.json'), 'launcher_sha256': sha256(Path(__file__)),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'environment': {k: environment[k] for k in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME',
            'TT_METAL_RUNTIME_ROOT', 'VIRTUAL_ENV', 'TT_METAL_LOGS_PATH', 'TT_METAL_PROFILER_DIR')},
        'profile': args.profile}

    def save():
        temporary = output/'launch.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output/'launch.json')

    with (output/'run.log').open('w') as log:
        process = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report['pid'] = process.pid
        save()
        report['returncode'] = process.wait()
    report['status'] = 'failed'
    if report['returncode'] == 0:
        result = json.loads((output/'execute.json').read_text())
        paths = [Path(name) for name in result['native_library_sha256']]
        if (result.get('status') == 'passed_smoke' and result.get('native_control', {}).get('bitwise_equal')
                and paths and all(path.is_relative_to(sdk) for path in paths)):
            report['status'] = 'completed_calibrated_execute' if not args.profile else 'completed_capture_unvalidated'
    save()
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    sys.exit(main())
