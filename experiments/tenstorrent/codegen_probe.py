"""Export compiler-generated flow TTNN, then validate/profile in a separate SDK.

Export with the Forge Python; execute with the profiling-enabled Metal Python.
This alternate execution boundary must be validated before interpreting profiles
as evidence about the original PJRT path. Never mix the two native runtimes.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('export', 'execute'))
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--case', choices=('dit_block', 'ss_flow', 'shape_flow_tensor', 'tex_flow_tensor'), default='shape_flow_tensor')
    p.add_argument('--checkpoints', type=Path, default=Path.home()/'models/genrecon')
    p.add_argument('--active-voxels', type=int)
    p.add_argument('--optimization-level', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--rope-mode', choices=('stock', 'real_pairs'), default='stock')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--profile-markers', action='store_true')
    p.add_argument('--result', type=Path, help='Execute report; must not exist')
    args = p.parse_args()
    if min(args.threads, args.runs) < 1:
        p.error('threads and runs must be positive')
    directory = args.directory.resolve()
    if args.command == 'export':
        directory.mkdir(parents=True, exist_ok=False)
        for name in ('TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
            os.environ.pop(name, None)
        os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    result_path = ((args.result or directory/'execute.json').resolve()
                   if args.command == 'execute' else directory/'export.json')
    if result_path.exists():
        raise FileExistsError(f'Preserve the existing report: {result_path}')
    report = {'status': 'running', 'command': args.command, 'directory': str(directory),
              'scope': 'Compiler-generated TTNN alternate execution; synthetic inputs; not a scene-quality benchmark',
              'source_sha256': sha256(Path(__file__)), 'phase': 'imports'}

    def save():
        temp = result_path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(result_path)

    def phase(name):
        report['phase'] = name
        save()
        print(name, flush=True)

    save()
    lock = None
    device = None
    try:
        import torch
        torch.set_num_threads(args.threads)
        report['torch'] = torch.__version__
        if args.command == 'export':
            import tt_torch
            report['pjrt_build'] = importlib.metadata.metadata('pjrt-plugin-tt').get('Summary')
            from experiments.tenstorrent.probes import build_case
            from experiments.tenstorrent.reference import float32_reference
            # Flow builders retain script-style imports from the probe directory.
            sys.path.insert(0, str(Path(__file__).parent))
            torch.manual_seed(args.seed)
            phase('build')
            model, inputs, metadata = build_case(args.case, args.checkpoints, active_voxels=args.active_voxels)
            model.eval()
            report.update(case=args.case, seed=args.seed, metadata=metadata,
                          compiler_options={'optimization_level': args.optimization_level})
            phase('stock_cpu_reference')
            with torch.no_grad(), float32_reference(model):
                expected = model(*inputs)
            torch.save({'inputs': inputs, 'expected': expected}, directory/'reference.pt')
            report['rope_mode'] = args.rope_mode
            if args.rope_mode == 'real_pairs':
                from experiments.tenstorrent.inference_rope import prepare_real_rope
                report['prepared_rope_blocks'] = prepare_real_rope(model)
                with torch.no_grad(), float32_reference(model):
                    candidate = model(*inputs)
                report['prepared_cpu_relative_l2'] = (
                    (candidate.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30)).item()
            phase('waiting_for_device')
            lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            os.chdir(directory)
            phase('export_codegen')
            start = time.perf_counter()
            with torch.no_grad():
                tt_torch.codegen_py(model, *inputs, compiler_options=report['compiler_options'],
                                    export_path=str(directory/'codegen'))
            report['export_seconds'] = time.perf_counter()-start
            report['artifact_sha256'] = {str(path.relative_to(directory)): sha256(path)
                for path in sorted((directory/'codegen').rglob('*')) if path.is_file()}
            report['reference_sha256'] = sha256(directory/'reference.pt')
            report['status'] = 'exported_unvalidated'
        else:
            import ttnn
            from experiments.tenstorrent.validation import enforce_check
            export = json.loads((directory/'export.json').read_text())
            if export.get('status') != 'exported_unvalidated':
                raise ValueError('Export has not completed successfully')
            report.update(case=export['case'], seed=export['seed'],
                          compiler_options=export['compiler_options'],
                          rope_mode=export.get('rope_mode', 'stock'))
            report['export_sha256'] = sha256(directory/'export.json')
            report['ttnn_path'] = ttnn.__file__
            metal_root = os.environ.get('TT_METAL_HOME')
            if metal_root:
                report['metal_commit'] = subprocess.check_output(
                    ['git', '-C', metal_root, 'rev-parse', 'HEAD'], text=True).strip()
            report['instrumented'] = os.environ.get('TT_METAL_DEVICE_PROFILER') == '1'
            report['timing_scope'] = 'Resident weights and compiler const-eval cache; disk activation loading excluded; input transfer, generated forward and synchronized CPU output included'
            if sha256(directory/'reference.pt') != export['reference_sha256']:
                raise ValueError('Reference artifact changed')
            for name, digest in export['artifact_sha256'].items():
                if sha256(directory/name) != digest:
                    raise ValueError(f'Generated artifact changed: {name}')
            graphs = list((directory/'codegen').glob('graph_*/main.py'))
            if len(graphs) != 1:
                raise ValueError('Expected exactly one generated graph')
            graph = graphs[0].parent
            sys.path.insert(0, str(graph))
            os.chdir(graph)
            spec = importlib.util.spec_from_file_location('generated_flow', graph/'main.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            expected = torch.load(directory/'reference.pt', weights_only=True)['expected']
            phase('waiting_for_device')
            lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            phase('prepare_weights')
            weights = module.load_weights_for__main()
            device = module.utils.DeviceGetter._instance
            if args.profile_markers:
                from tracy import signpost
            else:
                def signpost(label):
                    pass

            def infer(label):
                activations = module.load_activations_for__main()
                signpost(label+'_BEGIN')
                start = time.perf_counter()
                outputs = module._main(activations, weights)
                if len(outputs) != 1:
                    raise ValueError('Expected exactly one model output')
                actual = ttnn.to_torch(outputs[0]).float()
                seconds = time.perf_counter()-start
                signpost(label+'_END')
                ttnn.deallocate(outputs[0])
                return actual, seconds

            phase('first_forward')
            actual, report['first_seconds'] = infer('GENRECON_FIRST')
            report['vs_stock_cpu_fp32'] = enforce_check(report, 'first', actual, expected)
            phase('warmup')
            for index in range(3):
                infer(f'GENRECON_WARMUP_{index}')
            phase('measured')
            report['warm_seconds'] = []
            for index in range(args.runs):
                repeated, seconds = infer(f'GENRECON_MEASURED_{index}')
                torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                report['warm_seconds'].append(seconds)
            report['warm_median_seconds'] = statistics.median(report['warm_seconds'])
            torch.save(actual, result_path.with_suffix('.pt'))
            report['output_sha256'] = sha256(result_path.with_suffix('.pt'))
            report['status'] = 'passed_smoke'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if device is not None:
            import ttnn
            ttnn.close_mesh_device(device)
        # Keep the shared lock alive through native runtime atexit teardown.
    return 0


if __name__ == '__main__':
    sys.exit(main())
