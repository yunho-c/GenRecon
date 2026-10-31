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

ROOT = Path(__file__).resolve().parents[2]


def write_result(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def worker(args):
    # Match the working DINO experiment; the wheel supplies its own Metal runtime.
    for name in ("TT_METAL_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"):
        os.environ.pop(name, None)
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "none"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(ROOT))
    result = {
        "case": args.worker, "status": "running", "phase": "imports",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-xla", "tt-forge", "pjrt-plugin-tt")},
        "seed": args.seed, "fullgraph": True,
        "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (ROOT / "genrecon", ROOT / "experiments/tenstorrent")
            for p in sorted(folder.rglob("*.py"))},
        "timing_scope": "resident model; fresh input H2D, forward, D2H and synchronization",
    }
    path = args.output_dir / "result.json"
    write_result(path, result)
    try:
        import torch
        import tt_torch  # noqa: F401 - registers torch.compile backend="tt"
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as xr
        from probes import build_case

        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        result["phase"] = "build"
        write_result(path, result)
        model, inputs, metadata = build_case(args.worker, args.checkpoint_dir)
        model.eval()
        result.update(metadata)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["inputs"] = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs]
        result["phase"] = "cpu_reference"
        write_result(path, result)

        def outputs(value):
            return list(value) if isinstance(value, (list, tuple)) else [value]

        with torch.no_grad():
            start = time.perf_counter()
            reference = [t.detach().cpu() for t in outputs(model(*inputs))]
            result["cpu_seconds"] = time.perf_counter() - start
        result["phase"] = "device_transfer"
        write_result(path, result)
        device = torch_xla.device()
        assert xr.device_type() == "TT", "This probe requires a real TT PJRT device"
        result["device_type"] = xr.device_type()
        model.to(device)
        compiled = torch.compile(model, backend="tt", fullgraph=True)
        metrics.clear_all()
        result["phase"] = "compile_and_first_inference"
        write_result(path, result)

        def infer():
            with torch.no_grad():
                return [t.cpu() for t in outputs(compiled(*(t.to(device) for t in inputs)))]

        start = time.perf_counter()
        actual = infer()
        result["first_inference_seconds"] = time.perf_counter() - start
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
                if finite:
                    error = (a - b).norm().item() / max(a.norm().item(), 1e-12)
                    pcc = torch.corrcoef(torch.stack((a, b)))[0, 1].item() if a.std() > 0 and b.std() > 0 else None
                    item.update(relative_l2_error=error, pcc=pcc, max_abs_error=(a-b).abs().max().item())
                item["all_finite"] = finite
                item["passed"] = finite and error <= 0.1 and (pcc is None or pcc >= 0.99)
            comparisons.append(item)
        result["outputs"] = comparisons
        infer()  # warmup
        latencies = []
        for _ in range(args.runs):
            start = time.perf_counter()
            infer()
            latencies.append((time.perf_counter() - start) * 1000)
        result["warm_latency_ms"] = latencies
        result["warm_latency_median_ms"] = statistics.median(latencies)
        result["xla_executed_fallback_ops"] = list(metrics.executed_fallback_ops())
        result["xla_aten_counters"] = {n: metrics.counter_value(n) for n in metrics.counter_names() if n.startswith("aten::")}
        (args.output_dir / "xla_metrics.txt").write_text(metrics.metrics_report())
        passed = all(c["passed"] for c in comparisons)
        result["status"] = ("passed" if not result["xla_executed_fallback_ops"] else "passed_with_fallbacks") if passed else "numerical_mismatch"
        result["phase"] = "complete"
    except Exception as exc:
        result.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    write_result(path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["projection", "dense_aggregator", "sparse_aggregator", "ss_decoder", "dit_block"])
    parser.add_argument("--checkpoint-dir", type=Path, default=Path.home() / "models/genrecon")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1 or args.timeout < 1:
        parser.error("runs and timeout must be positive")
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
                   "--runs", str(args.runs), "--seed", str(args.seed)]
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
        write_result(path, result)
        summary.append(result)
        write_result(args.output_dir / "summary.json", summary)
        print(f"{case}: {result['status']} ({result['phase']})", flush=True)
    return 0 if all(r["status"] == "passed" for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
