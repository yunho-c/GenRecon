"""Measure dynamic conditioning prefill plus the complete original shape sampler.

Experimental FP32 two-graph execution, retaining every checkpoint parameter,
token and original Euler/CFG operation. Exact prepared CPU trajectories and the
sealed stock CPU reference are checked before native compilation. Timing covers
two resident prefill calls plus all 21 synchronized denoiser calls; compilation,
warmups, raw-condition upload and artifact I/O are reported separately/excluded.
"""
import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


def assert_bitwise_trajectory(actual, expected):
    """Require original identity and every complete sampler state to agree."""
    import torch
    from experiments.tenstorrent.trajectory import compare_trajectories
    compare_trajectories(actual, expected)  # validates noise, coords, conditions/config and lengths
    for key in ('samples', 'pred_x_t', 'pred_x_0'):
        pairs = [(actual[key], expected[key])] if key == 'samples' else zip(actual[key], expected[key], strict=True)
        for left, right in pairs:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path.home()/'models/genrecon')
    parser.add_argument('--cpu-trajectory-cache', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'case': 'shape_flow_tensor', 'seed': args.seed,
              'reference_precision': 'float32', 'cpu_threads': args.cpu_threads,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(name, flush=True)

    phase('imports')
    try:
        import torch
        from experiments.tenstorrent.probes import build_case
        from experiments.tenstorrent.trajectory import TrajectoryFixture, compare_trajectories
        from experiments.tenstorrent.reference_cache import load_cache
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.inference_rope import prepare_real_rope
        from experiments.tenstorrent.inference_attention import prepare_attention, attention_mode_counts
        from experiments.tenstorrent.inference_precision import prepare_float32_device
        from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_files_sha256'] = {str(path.relative_to(ROOT)): sha256(path)
            for folder in (ROOT/'genrecon', ROOT/'experiments/tenstorrent') for path in sorted(folder.rglob('*.py'))}
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        phase('build')
        model, inputs, metadata = build_case(report['case'], args.checkpoint_dir)
        model.eval()
        report.update(metadata)
        report['inputs'] = [{'shape': list(value.shape), 'dtype': str(value.dtype)} for value in inputs]
        fixture = TrajectoryFixture('shape', inputs, model.coords)
        phase('load_stock_cpu_cache')
        stock, report['stock_cpu_trajectory'], report['stock_cache'] = load_cache(
            args.cpu_trajectory_cache, report, fixture, checkpoint_path=args.checkpoint_dir/'shape_slat.pt')
        torch.save(stock, args.output_dir/'trajectory-stock.pt')
        report['prepared_rope_blocks'] = prepare_real_rope(model, head_first=True, flatten=True)
        report['prepared_attention_modules'] = prepare_attention(model, 'fp32_reciprocal')
        report['effective_attention_modes'] = attention_mode_counts(model)
        report['precision_preparation'] = prepare_float32_device(model)
        phase('ordinary_prepared_cpu_trajectory')
        with torch.no_grad():
            ordinary, report['ordinary_cpu_trajectory'] = fixture.run(
                lambda x, t, cond: model(x, t, cond['context'], cond['projected']))
        torch.save(ordinary, args.output_dir/'trajectory-ordinary-cpu.pt')
        report['ordinary_cpu_vs_stock'] = compare_trajectories(ordinary, stock)
        identities = {id(value) for value in model.parameters()}
        prefill, body = prepare_conditioning_cache(model)
        if identities != {id(value) for value in body.parameters()}:
            raise ValueError('Original learned parameter identities changed')
        report['parameter_identities_preserved'] = True
        branches = (fixture.positive, fixture.negative)
        phase('cached_cpu_trajectory')
        with torch.no_grad():
            cpu_cache = {id(cond): prefill(cond['context'], cond['projected']) for cond in branches}
            cached, report['cached_cpu_trajectory'] = fixture.run(lambda x, t, cond: body(x, t, cpu_cache[id(cond)]))
        assert_bitwise_trajectory(cached, ordinary)
        report['cached_cpu_all_states_bitwise_ordinary'] = True
        report['cached_cpu_vs_stock'] = compare_trajectories(cached, stock)
        report['cache_bytes_per_branch'] = sum(value.numel()*value.element_size() for value in cpu_cache[id(branches[0])])
        torch.save(cached, args.output_dir/'trajectory-cached-cpu.pt')
        del cpu_cache
        if args.cpu_only:
            report['status'] = 'completed_exact_cpu_control'
            phase('complete')
            return 0

        import tt_torch  # registers backend
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as runtime
        phase('waiting_for_device')
        # Descriptor retained through native atexit, serializing device teardown too.
        device_lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(device_lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert runtime.device_type() == 'TT'
        report['device_type'] = runtime.device_type()
        phase('device_transfer')
        body.to(device)
        prefill.to(device)
        resident = {id(cond): tuple(cond[key].to(device) for key in ('context', 'projected')) for cond in branches}
        torch_xla.sync(wait=True)

        def compile_module(module, name):
            torch.compiler.reset()
            torch_xla.set_custom_compile_options({'optimization_level': 0, 'math_fidelity': 'hifi4', 'dry_run': 'false',
                                                  'export_path': str(args.output_dir/(name+'-native-ir'))})
            return torch.compile(module, backend='tt', fullgraph=True)

        metrics.clear_all()
        phase('compile_prefill_and_warm_both_branches')
        compiled_prefill = compile_module(prefill, 'prefill')
        start = time.perf_counter()
        with torch.no_grad():
            for cond in branches:
                warm = compiled_prefill(*resident[id(cond)])
                torch_xla.sync(wait=True)
                del warm
        report['prefill_compile_and_both_branch_warm_seconds'] = time.perf_counter()-start
        gc.collect()
        phase('measured_native_prefill')
        before_prefill = dict(torch._dynamo.utils.counters['stats'])
        start = time.perf_counter()
        with torch.no_grad():
            native_cache = {id(cond): tuple(compiled_prefill(*resident[id(cond)])) for cond in branches}
        torch_xla.sync(wait=True)
        report['prefill_pair_seconds'] = time.perf_counter()-start
        report['prefill_recompiled'] = before_prefill != dict(torch._dynamo.utils.counters['stats'])
        report['prefill_calls'] = 2
        # Keep precisely these resident caches as body graph inputs. Recomputing
        # them after body tracing could retain a second large set of graph inputs.
        compiled_body = compile_module(body, 'body')

        def forward(x, t, cond):
            return compiled_body(x.to(device), t.to(device), native_cache[id(cond)]).cpu().float()

        phase('compile_body_and_warm_both_branches')
        start = time.perf_counter()
        with torch.no_grad():
            for cond in branches:
                forward(inputs[0], inputs[1], cond)
        report['body_compile_and_both_branch_warm_seconds'] = time.perf_counter()-start
        phase('native_complete_original_sampler')
        report['dynamo_before_trajectory'] = dict(torch._dynamo.utils.counters['stats'])
        with torch.no_grad():
            actual, report['device_trajectory'] = fixture.run(forward)
        report['dynamo_after_trajectory'] = dict(torch._dynamo.utils.counters['stats'])
        report['recompiled_during_trajectory'] = report['dynamo_before_trajectory'] != report['dynamo_after_trajectory']
        report['prefill_plus_sampler_seconds'] = report['prefill_pair_seconds']+report['device_trajectory']['seconds']
        report['trajectory_vs_stock_cpu_fp32'] = compare_trajectories(actual, stock)
        report['trajectory_vs_identical_cpu'] = compare_trajectories(actual, cached)
        torch.save(actual, args.output_dir/'trajectory-tt.pt')
        report['xla_executed_fallback_ops'] = list(metrics.executed_fallback_ops())
        report['xla_aten_counters'] = {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
        (args.output_dir/'xla_metrics.txt').write_text(metrics.metrics_report())
        report['trajectory_artifacts_sha256'] = {path.name: sha256(path) for path in args.output_dir.glob('trajectory-*.pt')}
        if report['xla_executed_fallback_ops'] or report['xla_aten_counters'] or report['prefill_recompiled'] or report['recompiled_during_trajectory']:
            raise ValueError('Fallback or recompilation invalidates native performance comparison')
        report['status'] = 'passed' if report['trajectory_vs_stock_cpu_fp32']['all_states_pass_smoke'] else 'numerical_mismatch'
        phase('complete')
        return 0 if report['status'] == 'passed' else 1
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'imports'))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
