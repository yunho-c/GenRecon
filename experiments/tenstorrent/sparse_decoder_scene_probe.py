"""Validate a large scene with saved CPU references and sequential TT weights.

Run sparse_decoder_probe.py first to produce CPU references. This follow-up
checks their input/checkpoint/source provenance and executes independently
predicted shape plus fixed/predicted texture. Coordinate differences remain
strict failures; intersection metrics only explain their extent.
"""

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from genrecon.modules.sparse import config, SparseTensor
from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache
from experiments.tenstorrent.sparse_decoder_probe import (
    load_model, file_sha256, source_provenance, device_memory,
    metrics, passed, compare, save_sparse, save_guides)


def aligned_rows(actual, expected):
    """Exact integer coordinate join; no nearest-neighbor matching."""
    if not len(actual) or not len(expected):
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    bounds = torch.maximum(actual.amax(0), expected.amax(0)).long() + 1
    def keys(coords):
        key = coords[:, 0].long()
        for axis in range(1, 4):
            key = key * bounds[axis] + coords[:, axis]
        return key
    actual_keys = keys(actual)
    expected_keys, order = keys(expected).sort()
    positions = torch.searchsorted(expected_keys, actual_keys)
    found = (positions < len(expected)) & (expected_keys[positions.clamp_max(len(expected) - 1)] == actual_keys)
    rows = found.nonzero().flatten()
    return rows, order[positions[found]]


def scene_compare(actual, expected, geometry=False):
    result = compare(actual, expected)
    a_rows, e_rows = aligned_rows(actual.coords, expected.coords)
    a, e = actual.feats[a_rows], expected.feats[e_rows]
    result.update(common_voxels=len(a_rows), missing_voxels=len(expected.coords) - len(a_rows),
                  extra_voxels=len(actual.coords) - len(a_rows), common_features=metrics(a, e),
                  common_rows_with_abs_error_over_1e_3=int(((a - e).abs().amax(1) > 1e-3).sum()))
    if geometry:
        result["common_edge_sign_mismatches"] = int(((a[:, 3:6] > 0) != (e[:, 3:6] > 0)).sum())
        result["common_edge_decisions"] = 3 * len(a_rows)
    return result


def scene_passed(report):
    """Enforce the declared continuous, subdivision and edge gates together."""
    fixed = report.get("fixed_shape", report["prior_fixed_shape"])
    return bool(
        all(passed(report[k], report["gate"]) for k in
            ("shape_predicted", "texture_fixed", "texture_predicted"))
        and passed(fixed["shape_fixed"], report["gate"])
        and report["chain_coordinates_equal"]
        and fixed["captured_wide"]["passed"]
        and len(fixed["subdivisions_fixed"]) == len(report["subdivisions_predicted"]) == 4
        and all(s["sign_mismatches"] == 0 for s in fixed["subdivisions_fixed"])
        and fixed["geometry_fixed"]["edge_sign_mismatches"] == 0
        and report["shape_predicted"]["common_edge_sign_mismatches"] == 0
        and all(s["common_sign_mismatches"] == s["missing_parents"] == s["extra_parents"] == 0
                for s in report["subdivisions_predicted"])
        and all(w["repeat_outputs_equal"] for w in report.get("warm", {}).values()))


def read_sparse(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    return SparseTensor(data["feats"], data["coords"])


def read_guides(path):
    result = []
    for data in torch.load(path, map_location="cpu", weights_only=True):
        value = SparseTensor(data["feats"], data["coords"], torch.Size(data["shape"]))
        value.register_spatial_cache("shape", torch.Size(data["spatial_shape"]))
        result.append(value)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, default=Path.home() / "models/genrecon/trellis2/ckpts")
    p.add_argument("--row-chunk", type=int, default=1024)
    p.add_argument("--precision", choices=("accurate", "bf16"), default="accurate")
    p.add_argument("--refresh-fixed-shape", action="store_true",
                   help="rerun the fixed-guide shape and wide convolution checks; automatic when precision changes")
    p.add_argument("--warm-repeats", type=int, default=0,
                   help="additional complete predicted-guide calls per decoder after its first execution")
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    if args.warm_repeats < 0 or args.row_chunk < 1 or args.threads < 1:
        p.error("warm-repeats must be nonnegative; row-chunk and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config.set_conv_backend("reference")
    baseline_path = args.reference_dir / "result.json"
    baseline = json.loads(baseline_path.read_text())
    input_path = args.reference_dir / "decoder_input.npz"
    if file_sha256(input_path) != baseline["generated_input_sha256"]:
        raise ValueError("saved reference inputs changed")
    for name, digest in baseline["checkpoint_sha256"].items():
        if file_sha256(args.checkpoint_dir / name) != digest:
            raise ValueError("reference checkpoint changed: " + name)
    for name in ("genrecon/models/sc_vaes/sparse_unet_vae.py", "genrecon/modules/sparse/conv/conv_reference.py"):
        if file_sha256(ROOT / name) != baseline["source"]["sha256"][name]:
            raise ValueError("CPU reference implementation changed: " + name)
    import flex_gemm
    flex_root = Path(flex_gemm.__file__).resolve().parents[1]
    name = "flex_gemm/tenstorrent/reference.py"
    if file_sha256(flex_root / name) != baseline["flex_gemm"]["sha256"][name]:
        raise ValueError("CPU convolution oracle changed")
    with np.load(input_path, allow_pickle=False) as data:
        coords = torch.from_numpy(data["coords"]).int()
        xshape = SparseTensor(torch.from_numpy(data["shape_feats"]).float(), coords)
        xtex = SparseTensor(torch.from_numpy(data["texture_feats"]).float(), coords)
    shape_ref = read_sparse(args.reference_dir / "shape_reference.pt")
    tex_ref = read_sparse(args.reference_dir / "texture_reference.pt")
    guides = read_guides(args.reference_dir / "reference_guides.pt")
    refresh_fixed = args.refresh_fixed_shape or args.precision != baseline["precision"]
    report = {"status": "running", "phase": "load_shape", "precision": args.precision,
        "algorithm": "explicit_gemm", "row_chunk": args.row_chunk, "input_voxels": len(coords),
        "input_scope": "saved_real_image_latent", "original_input_sha256": baseline["input_sha256"],
        "reference_dir": str(args.reference_dir), "reference_manifest_sha256": file_sha256(baseline_path),
        "reference_files_sha256": {name: file_sha256(args.reference_dir / name) for name in
            ("decoder_input.npz", "shape_reference.pt", "texture_reference.pt", "reference_guides.pt")},
        "checkpoint_sha256": baseline["checkpoint_sha256"], "gate": baseline["gate"],
        "threads": args.threads, "versions": {"torch": torch.__version__, "python": sys.version},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixed_shape_precision": args.precision if refresh_fixed else baseline["precision"],
        "warm_repeats": args.warm_repeats, "warm": {},
        "timing_scope": "complete calls including CPU boundary transfers, topology and output synchronization; excludes weight preparation, comparisons and artifact I/O",
        "watcher": os.environ.get("TT_METAL_WATCHER", "disabled"),
        "scope": "saved CPU references; one decoder's weights resident at a time; all voxel rows retained",
        "source": source_provenance(ROOT, [Path(__file__).resolve(),
            *sorted((ROOT / "genrecon/backends/tenstorrent").glob("*.py"))]),
        "flex_gemm": source_provenance(flex_root, sorted(
            p for p in (flex_root / "flex_gemm/tenstorrent").rglob("*")
            if p.suffix in (".py", ".cpp", ".hpp"))),
        "tt_metal": source_provenance(Path(os.environ["TT_METAL_HOME"]), []),
        "prior_fixed_shape": {k: baseline[k] for k in ("shape_fixed", "subdivisions_fixed", "geometry_fixed", "captured_wide")}}
    def phase(name):
        report["phase"] = name
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, flush=True)
    def warm_calls(label, call, expected):
        if not args.warm_repeats:
            return
        warm = report["warm"][label] = {"seconds": [], "repeat_outputs_equal": True}
        for repeat in range(args.warm_repeats):
            phase(f"warm_{label}_{repeat + 1}")
            start = time.perf_counter()
            output = call()
            seconds = time.perf_counter() - start
            actual = output[0] if isinstance(output, tuple) else output
            warm["seconds"].append(seconds)
            warm["repeat_outputs_equal"] &= bool(torch.equal(actual.coords, expected.coords)
                                                  and torch.equal(actual.feats, expected.feats))
            del output, actual
        warm["median_seconds"] = statistics.median(warm["seconds"])
        warm["min_seconds"], warm["max_seconds"] = min(warm["seconds"]), max(warm["seconds"])
    phase("load_shape")
    import ttnn
    shape = load_model(args.checkpoint_dir / "shape_dec_next_dc_f16c32_fp16")
    tex = None
    with open("/tmp/genrecon-tt-pjrt.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = ttnn.open_device(device_id=0)
        cache = TopologyCache()
        try:
            grid = device.compute_with_storage_grid_size()
            report["device"] = {"id": device.id(), "architecture": str(device.arch()), "grid": [grid.x, grid.y]}
            shape.enable_tenstorrent(device, row_chunk=args.row_chunk, precision=args.precision, topology_cache=cache)
            report["shape_prepare_seconds"] = shape._tenstorrent_backend.prepare_seconds
            report["memory_after_shape_prepare"] = device_memory(device)
            if refresh_fixed:
                phase("tt_captured_wide_convolution")
                wide_path = args.reference_dir / "shape_wide_input.pt"
                report["reference_files_sha256"][wide_path.name] = file_sha256(wide_path)
                wide_input = read_sparse(wide_path)
                module = shape.blocks[0][-1].conv1
                with torch.no_grad():
                    wide_ref = module(wide_input).feats
                backend = shape._tenstorrent_backend
                backend.stats = {"topology_seconds": 0.}
                wide_actual = ttnn.to_torch(backend.conv("blocks.0.4.conv1", module,
                    backend.upload(wide_input.feats), wide_input.coords, wide_input.shape[0],
                    tuple(wide_input.spatial_shape), 0)).float()
                wide_ok = bool(torch.isfinite(wide_actual).all() and
                    torch.all((wide_actual - wide_ref).abs() <= 0.02 + 0.02 * wide_ref.abs()))
                fixed_report = report["fixed_shape"] = {"captured_wide": {
                    "input_channels": 1024, "output_channels": 4096,
                    "features": metrics(wide_actual, wide_ref), "atol": 0.02, "rtol": 0.02, "passed": wide_ok}}
                del backend, wide_input, wide_ref, wide_actual
                phase("tt_shape_fixed_guides")
                fixed, predicted = shape.forward_features(xshape, guide_subs=guides, return_subs=True)
                fixed_report["shape_fixed"] = scene_compare(fixed, shape_ref, geometry=True)
                fixed_report["stats"] = dict(shape._tenstorrent_backend.stats)
                fixed_report["subdivisions_fixed"] = [
                    {"features": metrics(a.feats, e.feats), "sign_mismatches": int(((a.feats > 0) != (e.feats > 0)).sum()),
                     "decisions": a.feats.numel()} for a, e in zip(predicted, guides)]
                a, e = fixed.feats, shape_ref.feats
                fixed_report["geometry_fixed"] = {
                    "vertices": metrics(2 * a[:, :3].sigmoid() - 0.5, 2 * e[:, :3].sigmoid() - 0.5),
                    "edge_sign_mismatches": int(((a[:, 3:6] > 0) != (e[:, 3:6] > 0)).sum()),
                    "edge_decisions": a[:, 3:6].numel(),
                    "quad_weights": metrics(torch.nn.functional.softplus(a[:, 6:]), torch.nn.functional.softplus(e[:, 6:]))}
                save_sparse(args.output_dir / "shape_tt_fixed.pt", fixed)
                save_guides(args.output_dir / "fixed_tt_guides.pt", predicted)
                del fixed, predicted, a, e
            phase("tt_shape_predicted_guides")
            start = time.perf_counter()
            shape_actual, own_guides = shape.forward_features(xshape, return_subs=True)
            report["shape_wall_seconds"] = time.perf_counter() - start
            report["shape_predicted"] = scene_compare(shape_actual, shape_ref, geometry=True)
            report["shape_stats"] = dict(shape._tenstorrent_backend.stats)
            save_sparse(args.output_dir / "shape_tt_predicted.pt", shape_actual)
            save_guides(args.output_dir / "predicted_tt_guides.pt", own_guides)
            report["subdivisions_predicted"] = []
            for a, e in zip(own_guides, guides):
                ai, ei = aligned_rows(a.coords, e.coords)
                report["subdivisions_predicted"].append({"common_parents": len(ai),
                    "extra_parents": len(a.coords) - len(ai), "missing_parents": len(e.coords) - len(ai),
                    "common_sign_mismatches": int(((a.feats[ai] > 0) != (e.feats[ei] > 0)).sum())})
            warm_calls("shape", lambda: shape.forward_features(xshape, return_subs=True), shape_actual)
            shape.disable_tenstorrent()
            cache.clear()
            gc.collect()
            ttnn.synchronize_device(device)
            report["memory_after_shape_release"] = device_memory(device)
            phase("load_texture")
            tex = load_model(args.checkpoint_dir / "tex_dec_next_dc_f16c32_fp16")
            tex.enable_tenstorrent(device, row_chunk=args.row_chunk, precision=args.precision, topology_cache=cache)
            report["texture_prepare_seconds"] = tex._tenstorrent_backend.prepare_seconds
            report["memory_after_texture_prepare"] = device_memory(device)
            phase("tt_texture_fixed_guides")
            fixed = tex(xtex, guide_subs=guides)
            report["texture_fixed"] = scene_compare(fixed, tex_ref)
            report["texture_fixed_stats"] = dict(tex._tenstorrent_backend.stats)
            save_sparse(args.output_dir / "texture_tt_fixed.pt", fixed)
            phase("tt_texture_predicted_guides")
            start = time.perf_counter()
            actual = tex(xtex, guide_subs=own_guides)
            report["texture_wall_seconds"] = time.perf_counter() - start
            report["texture_predicted"] = scene_compare(actual, tex_ref)
            report["texture_predicted_stats"] = dict(tex._tenstorrent_backend.stats)
            report["chain_coordinates_equal"] = torch.equal(actual.coords, shape_actual.coords)
            save_sparse(args.output_dir / "texture_tt_predicted.pt", actual)
            warm_calls("texture", lambda: tex(xtex, guide_subs=own_guides), actual)
            report["status"] = "passed" if scene_passed(report) else "numerical_failure"
            phase("complete")
        except Exception as exc:
            report.update(status="runtime_failure", error=str(exc).split("backtrace:")[0],
                          traceback=traceback.format_exc())
            phase(report["phase"])
            raise
        finally:
            shape.disable_tenstorrent()
            if tex is not None:
                tex.disable_tenstorrent()
            cache.clear()
            ttnn.close_device(device)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
