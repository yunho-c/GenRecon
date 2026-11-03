"""Build an isolated Tracy-enabled SDK at Forge's pinned Metal source revision.

Requires installed clang20, cmake, ninja, ccache, MPI and uv. Source, virtualenv,
dependencies and build outputs stay under --workspace. The canonical Forge
environment and any source cache checkout are never modified. A successful
build is still unvalidated until native numeric/profiler controls are run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
METAL_COMMIT = 'f1f4ff75579ebd7a69c7da52d45368f273026d85'
METAL_URL = 'https://github.com/tenstorrent/tt-metal.git'


def main():
    from experiments.tenstorrent.native_environment import clean_native_environment, reexec_clean_native_process
    reexec_clean_native_process()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path)
    parser.add_argument('--uv', default=shutil.which('uv'))
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--configure-only', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1 or not args.uv:
        parser.error('Positive --jobs and an available --uv executable are required')
    workspace = args.workspace.resolve(); workspace.mkdir(parents=True, exist_ok=True)
    source, venv, build = workspace/'source', workspace/'venv', workspace/'build'
    python = venv/'bin/python'
    recipe = {'metal_commit': METAL_COMMIT, 'metal_url': METAL_URL, 'build_type': 'Release',
              'enable_tracy': True, 'enable_distributed': True, 'python_base': sys.executable,
              'torch_version': '2.11.0+cpu', 'python_minor': list(sys.version_info[:2])}
    state = workspace/'build-result.json'
    if state.exists() and json.loads(state.read_text())['recipe'] != recipe:
        raise ValueError('Existing SDK build recipe differs; use a new workspace')
    report = {'status': 'running', 'recipe': recipe, 'jobs': args.jobs, 'commands': [],
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    env = clean_native_environment()

    def phase(name):
        report['phase'] = name
        state.write_text(json.dumps(report, indent=2)+'\n')
        print(name, flush=True)

    def run(arguments, label, cwd=workspace, environment=None):
        phase(label)
        log = workspace/(label+'.log')
        start = time.time()
        with log.open('a') as output:
            process = subprocess.Popen([str(value) for value in arguments], cwd=cwd,
                env=env if environment is None else environment, stdout=output, stderr=subprocess.STDOUT)
            report['active_command'] = {'pid': process.pid, 'arguments': list(map(str, arguments)), 'log': str(log)}
            state.write_text(json.dumps(report, indent=2)+'\n')
            code = process.wait()
        report['commands'].append({**report.pop('active_command'), 'returncode': code, 'seconds': time.time()-start})
        state.write_text(json.dumps(report, indent=2)+'\n')
        if code:
            raise RuntimeError(f'{label} failed with exit {code}; see {log}')

    phase('validate_workspace')
    try:
        if not source.exists():
            if args.source_cache:
                run(['git', 'clone', '--no-hardlinks', '--no-checkout', args.source_cache.resolve(), source], 'clone_source')
                run(['git', 'remote', 'set-url', 'origin', METAL_URL], 'set_source_origin', source)
            else:
                run(['git', 'clone', '--no-checkout', METAL_URL, source], 'clone_source')
            run(['git', 'checkout', '--detach', METAL_COMMIT], 'checkout_source', source)
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True)
        if actual != METAL_COMMIT or dirty.strip():
            raise ValueError('SDK source commit changed or tracked source is dirty')
        run(['git', 'submodule', 'update', '--init', '--recursive', '--jobs', str(min(args.jobs, 8)), '--depth', '1'], 'submodules', source)
        report['submodules'] = subprocess.check_output(['git', 'submodule', 'status', '--recursive'], cwd=source, text=True).splitlines()
        if not python.exists():
            run([args.uv, 'venv', '--python', sys.executable, '--seed', venv], 'create_venv')
        run([args.uv, 'pip', 'install', '--python', python, '-e', source], 'install_source_package')
        run([args.uv, 'pip', 'install', '--python', python, '--index-url',
             'https://download.pytorch.org/whl/cpu', 'torch==2.11.0+cpu'], 'install_cpu_torch')
        build_env = {**env, 'PATH': str(venv/'bin')+os.pathsep+env['PATH'], 'VIRTUAL_ENV': str(venv),
                     'PYTHON_ENV_DIR': str(venv), 'CMAKE_BUILD_PARALLEL_LEVEL': str(args.jobs)}
        # Configure all interpreter/header settings in the initial pass. A second
        # pass cannot fix stale Python metadata if the first pass already fails.
        python_paths = json.loads(subprocess.check_output([python, '-c',
            'import sysconfig,json; print(json.dumps({"include":sysconfig.get_path("include"),'
            '"library":sysconfig.get_config_var("LIBDIR")+"/libpython"+sysconfig.get_config_var("LDVERSION")+".so"}))'],
            env=build_env, text=True))
        build.mkdir(parents=True, exist_ok=True)
        source_build = source/'build'
        if not source_build.exists() and not source_build.is_symlink():
            source_build.symlink_to(build, target_is_directory=True)
        if source_build.resolve() != build:
            raise ValueError('Existing source/build points outside this SDK workspace')
        run(['cmake', '-S', source, '-B', build, '-G', 'Ninja',
             '-DCMAKE_TOOLCHAIN_FILE='+str(source/'cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake'),
             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX='+str(build), '-DENABLE_CCACHE=TRUE',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DTT_UNITY_BUILDS=OFF', '-DTT_ENABLE_LIGHT_METAL_TRACE=ON',
             '-DWITH_PYTHON_BINDINGS=ON', '-DENABLE_DISTRIBUTED=ON',
             '-DENABLE_TRACY=ON', '-U_Python_*', '-U_Python3_*',
             '-DPython_EXECUTABLE='+str(python), '-DPython3_EXECUTABLE='+str(python),
             '-DPython_INCLUDE_DIR='+python_paths['include'], '-DPython3_INCLUDE_DIR='+python_paths['include'],
             '-DPython_LIBRARY='+python_paths['library'], '-DPython3_LIBRARY='+python_paths['library']],
             'configure', source, build_env)
        cache = (build/'CMakeCache.txt').read_text()
        if 'ENABLE_TRACY:BOOL=ON' not in cache or 'WITH_PYTHON_BINDINGS:BOOL=ON' not in cache:
            raise ValueError('CMake did not retain profiling and Python binding options')
        compile_commands = json.loads((build/'compile_commands.json').read_text())
        profiler_commands = [entry for entry in compile_commands if entry['file'].endswith('/llrt/rtoptions.cpp')]
        if len(profiler_commands) != 1 or '-DTRACY_ENABLE' not in profiler_commands[0]['command'].split():
            raise ValueError('Native runtime options source lacks the actual Tracy compile definition')
        report['profiler_compile_proof'] = profiler_commands[0]
        report['cmake_options'] = [line for line in cache.splitlines() if line and not line.startswith(('#', '//')) and
            any(key in line.split('=')[0] for key in ('TRACY', 'PYTHON', 'Python', 'CMAKE_CXX_COMPILER:', 'CMAKE_BUILD_TYPE:', 'DISTRIBUTED', 'UNITY', 'LIGHT_METAL'))]
        report['toolchain'] = {tool: subprocess.check_output([tool, '--version'], env=build_env, text=True).splitlines()[0]
                               for tool in ('clang++-20', 'cmake', 'ninja', 'ccache')}
        run([args.uv, 'pip', 'freeze', '--python', python], 'python_freeze')
        if args.configure_only:
            report['status'] = 'configured_unbuilt'
            phase('complete')
            return
        run(['cmake', '--build', build, '--target', 'install', '--parallel', str(args.jobs)], 'build', source, build_env)
        from experiments.tenstorrent.experiment_identity import sha256
        artifacts = sorted({path for directory in ('lib', 'lib64') for path in (build/directory).glob('*.so*') if path.is_file()})
        if not any(path.name == '_ttnn.so' for path in artifacts) or not any(path.name == 'libtt_metal.so' for path in artifacts):
            raise ValueError('Successful install did not produce required TTNN/Metal shared libraries')
        report['native_artifact_sha256'] = {str(path.relative_to(build)): sha256(path) for path in artifacts}
        report['status'] = 'built_unvalidated'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        state.write_text(json.dumps(report, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
