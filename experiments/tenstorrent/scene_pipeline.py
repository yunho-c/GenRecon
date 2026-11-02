"""Measure a 512-resolution single-cube pipeline with isolated TT runtimes.

This is an experimental measurement harness, not the public CUDA pipeline.
CPU handles projection, sampling arithmetic, sparse topology and stage files;
learned networks execute on TT-XLA or TTNN. Mesh extraction is excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from .experiment_identity import experiment_identity, validate_resume
else:
    from experiment_identity import experiment_identity, validate_resume

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("dino", "ss", "ss_decode", "shape", "tex", "decode")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, default=Path.home() / "models/genrecon")
    p.add_argument("--forge-python", default="/home/user/tt-forge-venv/bin/python")
    p.add_argument("--metal-python", default="/home/user/tt-metal/python_env/bin/python")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--flow-group-size", type=int, default=0,
                   help="Sparse flow blocks per compiled group; 0 compiles the whole flow")
    p.add_argument("--sparse-flow-layout", choices=("native", "sequence"), default="sequence")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.threads < 1 or args.flow_group_size < 0:
        p.error("threads must be positive and flow-group-size nonnegative")
    if args.sparse_flow_layout == "sequence" and args.flow_group_size:
        p.error("block grouping requires --sparse-flow-layout native")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    report_path = args.output_dir / "result.json"
    print("Fingerprinting source, settings and checkpoints (outside inference timing)", flush=True)
    identity = experiment_identity(ROOT, {
        "threads": args.threads, "flow_group_size": args.flow_group_size,
        "sparse_flow_layout": args.sparse_flow_layout,
        "forge_python": args.forge_python, "metal_python": args.metal_python,
    }, args.checkpoints)
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "scope": "512, one bounded poster cube, single Wormhole chip, no mesh extraction",
        "fixture": str(args.fixture.resolve()), "seed": args.seed, "stages": {},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "hardware": subprocess.check_output(["tt-smi", "-s"], text=True),
        "experiment_identity": identity,
    }
    validate_resume(report.get("experiment_identity"), identity)
    if report["fixture"] != str(args.fixture.resolve()) or report["seed"] != args.seed:
        raise ValueError("resume must use the same fixture and seed")
    report["status"] = "running"
    start = time.perf_counter()
    worker = Path(__file__).with_name("scene_pipeline_worker.py")
    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    save()
    for stage in STAGES:
        if stage not in args.stages:
            continue
        destination = args.output_dir / stage
        stage_result = destination / "result.json"
        if stage_result.exists():
            old = json.loads(stage_result.read_text())
            if args.resume and old.get("status") == "completed":
                # Identity above checks source/settings/checkpoints; also check
                # this stage's actual fixture and upstream tensor inputs.
                inputs = dict(old.get("input_files_sha256", {}))
                inputs.update(old.get("output_files_sha256", {}))
                inputs.update({item["path"]: item["sha256"] for item in old.get("images", [])})
                for path, expected in inputs.items():
                    if sha256(Path(path)) != expected:
                        raise ValueError(f"Completed {stage} input changed: {path}; use a fresh output directory")
                report["stages"][stage] = old
                print("reuse completed stage", stage, flush=True)
                continue
            raise RuntimeError(f"Preserve/move the previous attempt before rerunning {destination}")
        destination.mkdir(exist_ok=True)
        env = os.environ.copy()
        for name in ("TT_METAL_HOME", "LD_LIBRARY_PATH", "PYTHONPATH"):
            env.pop(name, None)
        if stage == "decode":
            env["TT_METAL_HOME"] = "/home/user/tt-metal"
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        command = [args.metal_python if stage == "decode" else args.forge_python,
            str(worker), "--stage", stage, "--fixture", str(args.fixture.resolve()),
            "--output-dir", str(args.output_dir.resolve()), "--checkpoints", str(args.checkpoints.resolve()),
            "--threads", str(args.threads), "--seed", str(args.seed)]
        command += ["--flow-group-size", str(args.flow_group_size)]
        command += ["--sparse-flow-layout", args.sparse_flow_layout]
        print("start", stage, flush=True)
        stage_start = time.perf_counter()
        with (destination / "worker.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        data = json.loads(stage_result.read_text()) if stage_result.exists() else {"status": "worker_failed"}
        data.update(process_seconds=time.perf_counter() - stage_start, returncode=process.returncode)
        stage_result.write_text(json.dumps(data, indent=2) + "\n")
        report["stages"][stage] = data
        save()
        if process.returncode or data.get("status") != "completed":
            report["status"] = "failed"
            save()
            raise RuntimeError(f"Stage {stage} failed; see {destination / 'worker.log'}")
        print("complete", stage, data["process_seconds"], flush=True)
    report.update(status="completed" if all(report["stages"].get(s, {}).get("status") == "completed" for s in STAGES) else "partial",
                  invocation_seconds=time.perf_counter() - start)
    report["warm_component_sum_seconds"] = sum(s.get("warm_path_seconds", 0.) for s in report["stages"].values())
    report["timing_note"] = ("Warm component sum includes measured sampler loops and CPU projection/control. "
        "It excludes loading, compilation, CPU reference checks and interprocess files/startup. "
        "It is not a measured resident end-to-end pipeline or an n300 benchmark.")
    save()


if __name__ == "__main__":
    main()
