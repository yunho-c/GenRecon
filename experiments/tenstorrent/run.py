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
    for name in ("TT_METAL_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"):
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
        "requested_active_voxels": args.active_voxels,
        "profile_markers": args.profile_markers,
        "requested_compile_options": {
            **({"optimization_level": args.optimization_level} if args.optimization_level is not None else {}),
            **({"enable_trace": "true"} if args.runtime_trace else {}),
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
            from inference_rope import prepare_real_rope
            start = time.perf_counter()
            result["prepared_rope_blocks"] = prepare_real_rope(model)
            result["rope_prepare_seconds"] = time.perf_counter() - start
            result["reference_note"] = "Unmodified stock model; error includes all preparation and device arithmetic"
            context = float32_reference(model) if args.reference_precision == "float32" else nullcontext(model)
            with torch.no_grad(), context:
                candidate = outputs(model(*reference_inputs))
                result["prepared_cpu_relative_l2"] = [
                    ((out.double().cpu() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()
                    for ref, out in zip(reference, candidate, strict=True)]
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
        result["xla_executed_fallback_ops"] = list(metrics.executed_fallback_ops())
        result["xla_aten_counters"] = {n: metrics.counter_value(n) for n in metrics.counter_names() if n.startswith("aten::")}
        (args.output_dir / "xla_metrics.txt").write_text(metrics.metrics_report())
        passed = all(c["passed"] for c in comparisons)
        fallbacks = result["xla_executed_fallback_ops"] or result["xla_aten_counters"]
        result["status"] = ("passed" if not fallbacks else "passed_with_fallbacks") if passed else "numerical_mismatch"
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
    parser.add_argument("--lora-mode", choices=("stock", "merged", "lowrank", "frozen_delta"), default="stock")
    parser.add_argument("--rope-mode", choices=("stock", "real_pairs"), default="stock")
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
    if args.active_voxels is not None and args.active_voxels < 1:
        parser.error("active-voxels must be positive")
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
        if args.active_voxels is not None:
            command += ["--active-voxels", str(args.active_voxels)]
        if args.optimization_level is not None:
            command += ["--optimization-level", str(args.optimization_level)]
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
