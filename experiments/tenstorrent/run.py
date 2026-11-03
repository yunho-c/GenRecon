#!/usr/bin/env python3
"""Run isolated CPU-reference / TT-XLA conversion probes (no NVIDIA device).

Each child writes a result before entering the compiler, so native crashes and
timeouts retain the last completed phase. Full logs stay in the output directory.
"""

import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import fcntl
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]


def write_result(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def worker(args):
    # Match the working DINO experiment; the wheel supplies its own Metal runtime.
    for name in ("TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "LD_LIBRARY_PATH", "PYTHONPATH"):
        os.environ.pop(name, None)
    os.environ["PJRT_DEVICE"] = "TT"
    if args.runtime_trace:
        os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = str(args.trace_region_size)
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "none"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(ROOT))
    result = {
        "case": args.worker, "status": "running", "phase": "imports",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-xla", "tt-forge", "pjrt-plugin-tt")},
        "seed": args.seed, "fullgraph": True,
        "lora_mode": args.lora_mode,
        "rope_mode": args.rope_mode,
        "attention_mode": args.attention_mode,
        "device_precision": args.device_precision,
        "requested_active_voxels": args.active_voxels,
        "profile_markers": args.profile_markers,
        "synthetic_trajectory": args.synthetic_trajectory and args.flow_inputs is None,
        "sampler_trajectory": args.synthetic_trajectory,
        "requested_compile_options": {
            **({"optimization_level": args.optimization_level} if args.optimization_level is not None else {}),
            **({"enable_trace": "true"} if args.runtime_trace else {}),
            **({"math_fidelity": args.math_fidelity} if args.math_fidelity is not None else {}),
            **({'export_path': str(args.output_dir / 'native-ir'), 'dry_run': 'false'} if args.native_ir else {}),
        },
        "trace_region_size": args.trace_region_size if args.runtime_trace else None,
        "compile_options_note": "Explicit options submitted to XLA; omitted settings retain SDK/environment defaults.",
        "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT / "genrecon", ROOT / "experiments/tenstorrent")
            for p in sorted(folder.rglob("*.py"))},
        "timing_scope": "resident model; fresh input H2D, forward, D2H and synchronization",
    }
    path = args.output_dir / "result.json"
    write_result(path, result)
    device_lock = None
    try:
        import torch
        import tt_torch  # noqa: F401 - registers torch.compile backend="tt"
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as xr
        from probes import build_case

        torch.set_num_threads(args.cpu_threads)
        result["cpu_threads"] = args.cpu_threads
        result["reference_precision"] = args.reference_precision
        torch.manual_seed(args.seed)
        result["phase"] = "build"
        write_result(path, result)
        model, inputs, metadata = build_case(args.worker, args.checkpoint_dir, active_voxels=args.active_voxels)
        model.eval()
        if args.flow_inputs is not None:
            from prepare_ss_inputs import load_ss_inputs
            inputs, fixture_metadata = load_ss_inputs(args.flow_inputs, inputs, metadata['checkpoint']['sha256'])
            metadata.update(fixture_metadata)
        result.update(metadata)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["inputs"] = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs]
        result["phase"] = "cpu_reference"
        write_result(path, result)

        def outputs(value):
            return list(value) if isinstance(value, (list, tuple)) else [value]

        from reference import float32_reference
        reference_context = float32_reference(model) if args.reference_precision == "float32" else nullcontext(model)
        reference_inputs = tuple(t.float() if args.reference_precision == "float32" and t.is_floating_point() else t for t in inputs)
        with torch.no_grad(), reference_context:
            start = time.perf_counter()
            reference = [t.detach().cpu() for t in outputs(model(*reference_inputs))]
            result["cpu_seconds"] = time.perf_counter() - start
        trajectory = None
        if args.synthetic_trajectory:
            from trajectory import TrajectoryFixture, compare_trajectories
            trajectory = TrajectoryFixture(args.worker.split('_')[0], inputs, getattr(model, 'coords', None),
                input_source='real posed-image' if args.flow_inputs is not None else 'synthetic')
            result['phase'] = 'load_cpu_trajectory_cache' if args.cpu_trajectory_cache is not None else 'stock_cpu_trajectory'
            write_result(path, result)
            if args.cpu_trajectory_cache is not None:
                from reference_cache import load_cache
                stem = {'ss': 'sparse_structure', 'shape': 'shape_slat', 'tex': 'texture_slat'}[trajectory.stage]
                trajectory_reference, result['cpu_trajectory'], result['cpu_trajectory_cache'] = load_cache(
                    args.cpu_trajectory_cache, result, trajectory,
                    checkpoint_path=args.checkpoint_dir / (stem + '.pt'))
            else:
                with torch.no_grad(), float32_reference(model):
                    trajectory_reference, result['cpu_trajectory'] = trajectory.run(
                        lambda x, t, cond: model(x, t, cond['context'], cond['projected']))
            torch.save(trajectory_reference, args.output_dir/'trajectory-reference.pt')
        if args.lora_mode != "stock":
            from inference_lora import prepare_lora
            start = time.perf_counter()
            result["prepared_lora_linears"] = prepare_lora(model, args.lora_mode)
            result["lora_prepare_seconds"] = time.perf_counter() - start
            result["reference_note"] = "Unmodified stock model; error includes LoRA preparation and device arithmetic"
            context = float32_reference(model) if args.reference_precision == "float32" else nullcontext(model)
            with torch.no_grad(), context:
                candidate = outputs(model(*reference_inputs))
                result["lora_cpu_relative_l2"] = [
                    ((out.double().cpu() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()
                    for ref, out in zip(reference, candidate, strict=True)]
        if args.rope_mode != "stock":
            from experiments.tenstorrent.inference_rope import prepare_real_rope
            start = time.perf_counter()
            result["prepared_rope_blocks"] = prepare_real_rope(model,
                head_first=args.rope_mode in ('real_pairs_head_first', 'real_pairs_flat_head_first'),
                flatten=args.rope_mode == 'real_pairs_flat_head_first')
            result["rope_prepare_seconds"] = time.perf_counter() - start
            result["reference_note"] = "Unmodified stock model; error includes all preparation and device arithmetic"
            context = float32_reference(model) if args.reference_precision == "float32" else nullcontext(model)
            with torch.no_grad(), context:
                candidate = outputs(model(*reference_inputs))
                result["prepared_cpu_relative_l2"] = [
                    ((out.double().cpu() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()
                    for ref, out in zip(reference, candidate, strict=True)]
        if args.attention_mode != 'stock':
            from experiments.tenstorrent.inference_attention import prepare_attention
            result['prepared_attention_modules'] = prepare_attention(model, args.attention_mode)
            context = float32_reference(model) if args.reference_precision == 'float32' else nullcontext(model)
            with torch.no_grad(), context:
                prepared = outputs(model(*reference_inputs))
            result['attention_cpu_relative_l2'] = [
                ((out.double().cpu() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()
                for ref, out in zip(reference, prepared, strict=True)]
            result['attention_note'] = 'Attention preparation retains learned weights and its outer dtype; stable FP32 unfused exp/reciprocal, with QK/AV precision selected by attention_mode. Any subsequent whole-model device precision change is reported separately. Error includes all preparation and device arithmetic.'
        if args.device_precision == 'float32':
            from experiments.tenstorrent.inference_precision import prepare_float32_device
            result['phase'] = 'prepare_device_float32'
            write_result(path, result)
            result['device_precision_preparation'] = prepare_float32_device(model)
            inputs = reference_inputs
            result['phase'] = 'prepared_device_cpu_reference'
            write_result(path, result)
            with torch.no_grad():
                prepared = outputs(model(*reference_inputs))
            result['device_precision_cpu_relative_l2'] = [
                ((out.double().cpu() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()
                for ref, out in zip(reference, prepared, strict=True)]
        result['device_input_dtypes'] = [str(value.dtype) for value in inputs]
        result["phase"] = "waiting_for_device"
        write_result(path, result)
        # Permit CPU references/downloads to overlap, but serialize our hardware
        # runs even when separate invocations of this harness are queued.
        device_lock = os.open("/tmp/genrecon-tt-pjrt.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(device_lock, fcntl.LOCK_EX)
        result["phase"] = "device_transfer"
        write_result(path, result)
        device = torch_xla.device()
        assert xr.device_type() == "TT", "This probe requires a real TT PJRT device"
        torch_xla.set_custom_compile_options(result["requested_compile_options"])
        result["device_type"] = xr.device_type()
        model.to(device)
        compiled = torch.compile(model, backend="tt", fullgraph=True)
        if args.profile_markers:
            from tracy import signpost
        else:
            def signpost(label):
                pass
        metrics.clear_all()
        result["phase"] = "compile_and_first_inference"
        write_result(path, result)

        def infer():
            with torch.no_grad():
                return [t.cpu() for t in outputs(compiled(*(t.to(device) for t in inputs)))]

        signpost("GENRECON_FIRST_BEGIN")
        start = time.perf_counter()
        actual = infer()
        result["first_inference_seconds"] = time.perf_counter() - start
        signpost("GENRECON_FIRST_END")
        result["phase"] = "comparison"
        write_result(path, result)
        comparisons = []
        for ref, out in zip(reference, actual, strict=True):
            assert ref.shape == out.shape
            item = {"shape": list(out.shape), "dtype": str(out.dtype)}
            if not ref.is_floating_point():
                item.update(exact_match=bool(torch.equal(ref, out)), passed=bool(torch.equal(ref, out)),
                            mismatch_count=int((ref != out).sum()), element_count=ref.numel())
            else:
                a, b = ref.float().flatten(), out.float().flatten()
                finite = bool(torch.isfinite(b).all() and torch.isfinite(a).all())
                error, pcc = None, None
                if finite:
                    error = (a - b).norm().item() / max(a.norm().item(), 1e-12)
                    pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item() if a.std() > 0 and b.std() > 0 else None
                    item.update(relative_l2_error=error, pcc=pcc, max_abs_error=(a-b).abs().max().item())
                item["all_finite"] = finite
                item["passed"] = finite and error <= 0.1 and (pcc is None or pcc >= 0.99)
            comparisons.append(item)
        result["outputs"] = comparisons
        signpost("GENRECON_WARMUP_BEGIN")
        infer()  # warmup
        signpost("GENRECON_WARMUP_END")
        latencies = []
        for repeat in range(args.runs):
            signpost(f"GENRECON_MEASURED_{repeat}_BEGIN")
            start = time.perf_counter()
            infer()
            latencies.append((time.perf_counter() - start) * 1000)
            signpost(f"GENRECON_MEASURED_{repeat}_END")
        result["warm_latency_ms"] = latencies
        result["warm_latency_median_ms"] = statistics.median(latencies)
        if trajectory is not None:
            device_conditions = {id(cond): tuple(cond[key].to(device) for key in ('context', 'projected'))
                                 for cond in (trajectory.positive, trajectory.negative)}
            def trajectory_forward(x, t, cond):
                return compiled(x.to(device), t.to(device), *device_conditions[id(cond)]).cpu().float()
            # Exercise the negative branch before timing; shapes must stay fixed.
            with torch.no_grad():
                trajectory_forward(inputs[0], inputs[1], trajectory.negative)
            result['phase'] = 'device_trajectory'
            write_result(path, result)
            result['dynamo_before_trajectory'] = dict(torch._dynamo.utils.counters['stats'])
            with torch.no_grad():
                trajectory_actual, result['device_trajectory'] = trajectory.run(trajectory_forward)
            result['dynamo_after_trajectory'] = dict(torch._dynamo.utils.counters['stats'])
            result['recompiled_during_trajectory'] = result['dynamo_after_trajectory'] != result['dynamo_before_trajectory']
            result['device_trajectory']['timing_scope'] = 'Resident weights and preuploaded conditions; input/output transfer, synchronized model calls, original CPU Euler/CFG arithmetic and instrumentation. Preparation, CPU reference, compilation/warmups and artifact I/O excluded.'
            result['trajectory_vs_stock_cpu_fp32'] = compare_trajectories(trajectory_actual, trajectory_reference)
            torch.save(trajectory_actual, args.output_dir/'trajectory-tt.pt')
            result['trajectory_artifacts_sha256'] = {name: hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest()
                for name in ('trajectory-reference.pt', 'trajectory-tt.pt')}
        result["xla_executed_fallback_ops"] = list(metrics.executed_fallback_ops())
        result["xla_aten_counters"] = {n: metrics.counter_value(n) for n in metrics.counter_names() if n.startswith("aten::")}
        (args.output_dir / "xla_metrics.txt").write_text(metrics.metrics_report())
        passed = all(c["passed"] for c in comparisons)
        if trajectory is not None:
            passed = passed and result['trajectory_vs_stock_cpu_fp32']['all_states_pass_smoke']
        fallbacks = result["xla_executed_fallback_ops"] or result["xla_aten_counters"]
        result["status"] = ("passed" if not fallbacks else "passed_with_fallbacks") if passed else "numerical_mismatch"
        if trajectory is not None and result['recompiled_during_trajectory']:
            result['status'] = 'compiled_during_trajectory'
        result["phase"] = "complete"
    except Exception as exc:
        result.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    write_result(path, result)
    print(json.dumps(result, indent=2), flush=True)
    # The worker owns the descriptor until process exit, including TT's atexit
    # teardown. Releasing it here could race the next worker's device startup.
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["projection", "dense_aggregator", "sparse_aggregator", "ss_decoder", "dit_block"])
    parser.add_argument("--checkpoint-dir", type=Path, default=Path.home() / "models/genrecon")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--reference-precision", choices=("native", "float32"), default="native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument('--math-fidelity', choices=('lofi','hifi2','hifi3','hifi4','ttnn_default'), default=None)
    parser.add_argument("--lora-mode", choices=("stock", "merged", "lowrank", "frozen_delta"), default="stock")
    parser.add_argument("--rope-mode", choices=("stock", "real_pairs", "real_pairs_head_first", "real_pairs_flat_head_first"), default="stock")
    parser.add_argument('--attention-mode', choices=('stock', 'fp32_reciprocal', 'fp32_scores_bf16_av', 'bf16_qk_fp32_softmax'), default='stock')
    parser.add_argument('--device-precision', choices=('pipeline', 'float32'), default='pipeline',
                        help='FP32 device diagnostic promotes the already quantized model after original CPU references')
    parser.add_argument('--native-ir', action='store_true', help='Dump actual native compiler IR with dry run disabled')
    parser.add_argument('--synthetic-trajectory', '--sampler-trajectory', action='store_true',
                        help='Compare the complete original 12-step schedule against stock FP32 CPU (complete tensor flows only)')
    parser.add_argument('--flow-inputs', type=Path,
                        help='Prepared real-image occupancy inputs; currently requires ss_flow')
    parser.add_argument('--cpu-trajectory-cache', type=Path,
                        help='Sealed stock FP32 reference cache; strict model/fixture identity required')
    parser.add_argument("--active-voxels", type=int, default=None,
                        help="Synthetic sparse flow size; default 512. Not a real-scene fixture.")
    parser.add_argument("--runtime-trace", action="store_true")
    parser.add_argument("--profile-markers", action="store_true",
                        help="Emit Tracy signposts around first/warmup/measured synchronized forwards")
    parser.add_argument("--trace-region-size", type=int, default=10_000_000)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.runs, args.timeout, args.cpu_threads, args.trace_region_size) < 1:
        parser.error("runs, timeout, cpu-threads and trace-region-size must be positive")
    if args.device_precision == 'float32' and args.reference_precision != 'float32':
        parser.error('FP32 device diagnostic requires the stock FP32 reference')
    if args.active_voxels is not None and args.active_voxels < 1:
        parser.error("active-voxels must be positive")
    if args.flow_inputs is not None:
        cases = [args.worker] if args.worker else args.cases
        if any(case != 'ss_flow' for case in cases):
            parser.error('flow-inputs currently supports ss_flow only')
        args.flow_inputs = args.flow_inputs.resolve()
    if args.cpu_trajectory_cache is not None:
        if not args.synthetic_trajectory:
            parser.error('cpu-trajectory-cache requires a sampler trajectory')
        args.cpu_trajectory_cache = args.cpu_trajectory_cache.resolve()
    if args.synthetic_trajectory:
        cases = [args.worker] if args.worker else args.cases
        if args.reference_precision != 'float32' or any(case not in ('ss_flow', 'shape_flow_tensor', 'tex_flow_tensor') for case in cases):
            parser.error('sampler trajectories require FP32 reference and ss_flow/shape_flow_tensor/tex_flow_tensor')
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    summary = []
    for case in args.cases:
        dest = args.output_dir / case
        dest.mkdir(exist_ok=True)
        if (dest / "result.json").exists():
            raise FileExistsError(f"Use a new output directory; refusing to overwrite {dest}")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                   "--output-dir", str(dest), "--checkpoint-dir", str(args.checkpoint_dir),
                   "--runs", str(args.runs), "--seed", str(args.seed), "--cpu-threads", str(args.cpu_threads),
                   "--reference-precision", args.reference_precision]
        command += ["--lora-mode", args.lora_mode]
        command += ["--rope-mode", args.rope_mode]
        command += ['--attention-mode', args.attention_mode]
        command += ['--device-precision', args.device_precision]
        if args.native_ir:
            command += ['--native-ir']
        if args.synthetic_trajectory:
            command += ['--synthetic-trajectory']
        if args.flow_inputs is not None:
            command += ['--flow-inputs', str(args.flow_inputs)]
        if args.cpu_trajectory_cache is not None:
            command += ['--cpu-trajectory-cache', str(args.cpu_trajectory_cache)]
        if args.active_voxels is not None:
            command += ["--active-voxels", str(args.active_voxels)]
        if args.optimization_level is not None:
            command += ["--optimization-level", str(args.optimization_level)]
        if args.math_fidelity is not None:
            command += ['--math-fidelity', args.math_fidelity]
        if args.runtime_trace:
            command += ["--runtime-trace", "--trace-region-size", str(args.trace_region_size)]
        if args.profile_markers:
            command += ["--profile-markers"]
        print(f"Running {case}; log: {dest / 'run.log'}", flush=True)
        with (dest / "run.log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=dest, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        path = dest / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {"case": case, "phase": "startup"}
        result["returncode"] = code
        if result.get("status") in (None, "running"):
            result["status"] = "timeout" if code == 124 else "process_failed"
            with (dest / "run.log").open("rb") as log:
                log.seek(max(0, (dest / "run.log").stat().st_size - 16384))
                result["log_tail"] = log.read().decode("utf-8",errors="replace").splitlines()[-40:]
        write_result(path, result)
        summary.append(result)
        write_result(args.output_dir / "summary.json", summary)
        print(f"{case}: {result['status']} ({result['phase']})", flush=True)
    return 0 if all(r["status"] == "passed" for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
