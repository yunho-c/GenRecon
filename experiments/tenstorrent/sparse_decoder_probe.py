"""Execute both complete pretrained sparse decoders on CPU and Tenstorrent.

An optional NPZ contains coords [N,4], shape_feats [N,32], texture_feats [N,32].
Without it, inputs are explicitly synthetic; no reconstruction-quality claim is
made. Raw decoder outputs and captured wide-convolution inputs stay outside git.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from genrecon.modules.sparse import config, SparseTensor
from genrecon.models import __getattr__ as model_class
from genrecon.backends.tenstorrent.sparse_decoder import TopologyCache


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(root, paths):
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
            "sha256": {str(path.relative_to(root)): file_sha256(path) for path in paths}}


def device_memory(device):
    """Live allocator snapshots, explicitly not a transient peak measurement."""
    import ttnn
    result = {}
    for name, kind in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
        view = ttnn.get_memory_view(device, kind)
        result[name] = {"banks": view.num_banks,
            "capacity_bytes": view.num_banks * view.total_bytes_per_bank,
            "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank}
    return result


def metrics(actual, expected):
    if actual.shape != expected.shape:
        return {"shape_match": False}
    a, e = actual.double().reshape(-1), expected.double().reshape(-1)
    if not a.numel():
        return {"shape_match": True, "finite": True, "relative_l2": 0., "pcc": 1., "max_abs": 0.}
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return {"shape_match": True, "finite": False}
    ac, ec = a - a.mean(), e - e.mean()
    return {"shape_match": True, "finite": finite,
        "relative_l2": ((a - e).norm() / e.norm().clamp_min(1e-30)).item(),
        "pcc": ((ac * ec).sum() / (ac.norm() * ec.norm()).clamp_min(1e-30)).item(),
        "max_abs": (a - e).abs().max().item()}


def compare(actual, expected):
    same = torch.equal(actual.coords, expected.coords)
    return {"coordinates_equal": same, "actual_voxels": len(actual.coords),
            "reference_voxels": len(expected.coords),
            "features": metrics(actual.feats, expected.feats) if same else None}


def passed(comparison, gate):
    m = comparison.get("features")
    return bool(comparison["coordinates_equal"] and m and m.get("finite")
                and m["relative_l2"] <= gate["relative_l2"] and m["pcc"] >= gate["pcc"])


def load_model(path):
    from safetensors.torch import load_file
    cfg = json.loads(path.with_suffix(".json").read_text())
    cfg["args"]["use_fp16"] = False
    with torch.device("meta"):
        model = model_class(cfg["name"])(**cfg["args"])
    model.load_state_dict(load_file(str(path.with_suffix(".safetensors"))), strict=True, assign=True)
    return model.float().eval()


def save_sparse(path, value):
    torch.save({"coords": value.coords, "feats": value.feats}, path)


def save_guides(path, guides):
    torch.save([{"coords": g.coords, "feats": g.feats, "shape": tuple(g.shape),
                 "spatial_shape": tuple(g.spatial_shape)} for g in guides], path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=Path.home() / "models/genrecon/trellis2/ckpts")
    p.add_argument("--latent", type=Path)
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--algorithm", default="explicit_gemm")
    p.add_argument("--row-chunk", type=int, default=256)
    p.add_argument("--precision", choices=("bf16", "accurate"), default="accurate")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--warm-repeats", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.points < 1 or args.warm_repeats < 0:
        p.error("points must be positive and warm-repeats nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config.set_conv_backend("reference")
    report = {"status": "running", "phase": "input", "seed": args.seed,
        "algorithm": args.algorithm, "row_chunk": args.row_chunk, "precision": args.precision,
        "watcher": os.environ.get("TT_METAL_WATCHER", "disabled"),
        "input_scope": "saved_latent" if args.latent else "synthetic_gaussian_latent",
        "input_path": str(args.latent) if args.latent else None,
        "gate": {"relative_l2": 1e-4 if args.precision == "accurate" else 0.1,
                 "pcc": 0.999999 if args.precision == "accurate" else 0.99, "coordinates": "exact",
                 "subdivision_signs": "exact", "geometry_edge_signs": "exact",
                 "scope": "decoder numerical parity; real-scene reconstruction quality is separate"},
        "versions": {"torch": torch.__version__, "python": sys.version},
        "source": source_provenance(ROOT, [Path(__file__).resolve(),
            *sorted((ROOT / "genrecon/backends/tenstorrent").glob("*.py")),
            ROOT / "genrecon/models/sc_vaes/sparse_unet_vae.py",
            ROOT / "genrecon/modules/sparse/conv/conv_reference.py"])}
    import flex_gemm
    flex_root = Path(flex_gemm.__file__).resolve().parents[1]
    report["flex_gemm"] = source_provenance(flex_root, sorted(
        p for p in (flex_root / "flex_gemm/tenstorrent").rglob("*")
        if p.suffix in (".py", ".cpp", ".hpp")))
    metal_root = Path(os.environ["TT_METAL_HOME"])
    report["tt_metal"] = source_provenance(metal_root, [])
    report["checkpoint_sha256"] = {name: file_sha256(args.checkpoint_dir / name)
        for stem in ("shape_dec_next_dc_f16c32_fp16", "tex_dec_next_dc_f16c32_fp16")
        for name in (stem + ".json", stem + ".safetensors")}

    def phase(name):
        report["phase"] = name
        (args.output_dir / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, flush=True)

    phase("input")
    if args.latent:
        report["input_sha256"] = file_sha256(args.latent)
        with np.load(args.latent, allow_pickle=False) as data:
            coords = torch.from_numpy(data["coords"]).int()
            shape_feats = torch.from_numpy(data["shape_feats"]).float()
            texture_feats = torch.from_numpy(data["texture_feats"]).float()
    else:
        generator = torch.Generator().manual_seed(args.seed)
        side = math.ceil(args.points**(1 / 3))
        ids = torch.randperm(side**3, generator=generator)[:args.points]
        coords = torch.stack((ids * 0, ids // side**2, ids // side % side, ids % side), dim=1).int()
        shape_feats = torch.randn(args.points, 32, generator=generator)
        texture_feats = torch.randn(args.points, 32, generator=generator)
    report["input_voxels"] = len(coords)
    if (coords.ndim != 2 or coords.shape[1] != 4 or
            shape_feats.shape != (len(coords), 32) or texture_feats.shape != shape_feats.shape
            or not len(coords) or not torch.isfinite(shape_feats).all()
            or not torch.isfinite(texture_feats).all()):
        raise ValueError("expected nonempty coords [N,4] and finite shape/texture features [N,32]")
    np.savez(args.output_dir / "decoder_input.npz", coords=coords.numpy(),
             shape_feats=shape_feats.numpy(), texture_feats=texture_feats.numpy())
    report["generated_input_sha256"] = file_sha256(args.output_dir / "decoder_input.npz")
    xshape, xtex = SparseTensor(shape_feats, coords), SparseTensor(texture_feats, coords)
    phase("load_shape")
    shape = load_model(args.checkpoint_dir / "shape_dec_next_dc_f16c32_fp16")
    report["shape_parameters"] = sum(p.numel() for p in shape.parameters())
    # Actual internal activations from the full pretrained CPU decoder, with
    # input provenance retained (synthetic or supplied scene latent).
    def capture_wide(module, inputs):
        save_sparse(args.output_dir / "shape_wide_input.pt", inputs[0])
    capture = shape.blocks[0][-1].conv1.register_forward_pre_hook(capture_wide)
    phase("cpu_shape")
    start = time.perf_counter()
    with torch.no_grad():
        shape_ref, guides = shape.forward_features(xshape, return_subs=True)
    capture.remove()
    report["cpu_shape_seconds"] = time.perf_counter() - start
    report["reference_stage_voxels"] = [len(g.coords) for g in guides] + [len(shape_ref.coords)]
    save_sparse(args.output_dir / "shape_reference.pt", shape_ref)
    save_guides(args.output_dir / "reference_guides.pt", guides)
    phase("load_texture")
    tex = load_model(args.checkpoint_dir / "tex_dec_next_dc_f16c32_fp16")
    report["texture_parameters"] = sum(p.numel() for p in tex.parameters())
    phase("cpu_texture")
    start = time.perf_counter()
    with torch.no_grad():
        tex_ref = tex(xtex, guide_subs=guides)
    report["cpu_texture_seconds"] = time.perf_counter() - start
    save_sparse(args.output_dir / "texture_reference.pt", tex_ref)
    phase("waiting_for_device")
    import ttnn
    with open("/tmp/genrecon-tt-pjrt.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = ttnn.open_device(device_id=0)
        try:
            grid = device.compute_with_storage_grid_size()
            report["device"] = {"id": device.id(), "architecture": str(device.arch()),
                                "grid": [grid.x, grid.y]}
            cache = TopologyCache()
            phase("prepare_shape")
            shape.enable_tenstorrent(device, algorithm=args.algorithm, row_chunk=args.row_chunk,
                                     precision=args.precision, topology_cache=cache)
            report["shape_prepare_seconds"] = shape._tenstorrent_backend.prepare_seconds
            report["memory_after_shape_prepare"] = device_memory(device)
            phase("tt_captured_wide_convolution")
            captured = torch.load(args.output_dir / "shape_wide_input.pt", weights_only=True)
            wide_input = SparseTensor(captured["feats"], captured["coords"])
            wide_module = shape.blocks[0][-1].conv1
            with torch.no_grad():
                wide_reference = wide_module(wide_input).feats
            backend = shape._tenstorrent_backend
            backend.stats = {"topology_seconds": 0.}
            wide_actual = ttnn.to_torch(backend.conv("blocks.0.4.conv1", wide_module,
                backend.upload(wide_input.feats), wide_input.coords, wide_input.shape[0],
                tuple(wide_input.spatial_shape), 0)).float()
            wide_atol, wide_rtol = 0.02, 0.02
            wide_ok = bool(torch.isfinite(wide_actual).all() and
                torch.all((wide_actual - wide_reference).abs() <= wide_atol + wide_rtol * wide_reference.abs()))
            report["captured_wide"] = {"input_channels": 1024, "output_channels": 4096,
                "input_origin": report["input_scope"], "features": metrics(wide_actual, wide_reference),
                "atol": wide_atol, "rtol": wide_rtol, "passed": wide_ok}
            phase("tt_shape_fixed_guides")
            shape_fixed, predicted = shape.forward_features(xshape, guide_subs=guides, return_subs=True)
            report["shape_fixed"] = compare(shape_fixed, shape_ref)
            report["shape_fixed_stats"] = dict(shape._tenstorrent_backend.stats)
            report["subdivisions_fixed"] = [
                {"features": metrics(a.feats, b.feats), "sign_mismatches": int(((a.feats > 0) != (b.feats > 0)).sum()),
                 "decisions": a.feats.numel()} for a, b in zip(predicted, guides)]
            if torch.equal(shape_fixed.coords, shape_ref.coords):
                a, b = shape_fixed.feats, shape_ref.feats
                report["geometry_fixed"] = {
                    "vertices": metrics(2 * a[:, :3].sigmoid() - 0.5, 2 * b[:, :3].sigmoid() - 0.5),
                    "edge_sign_mismatches": int(((a[:, 3:6] > 0) != (b[:, 3:6] > 0)).sum()),
                    "edge_decisions": a[:, 3:6].numel(),
                    "quad_weights": metrics(torch.nn.functional.softplus(a[:, 6:]), torch.nn.functional.softplus(b[:, 6:]))}
            save_sparse(args.output_dir / "shape_tt_fixed.pt", shape_fixed)
            save_guides(args.output_dir / "fixed_tt_guides.pt", predicted)
            phase("tt_shape_predicted_guides")
            shape_free, own_guides = shape.forward_features(xshape, return_subs=True)
            report["shape_predicted"] = compare(shape_free, shape_ref)
            report["shape_predicted_stats"] = dict(shape._tenstorrent_backend.stats)
            save_sparse(args.output_dir / "shape_tt_predicted.pt", shape_free)
            save_guides(args.output_dir / "predicted_tt_guides.pt", own_guides)
            phase("prepare_texture")
            tex.enable_tenstorrent(device, algorithm=args.algorithm, row_chunk=args.row_chunk,
                                   precision=args.precision, topology_cache=cache)
            report["texture_prepare_seconds"] = tex._tenstorrent_backend.prepare_seconds
            report["memory_after_both_prepare"] = device_memory(device)
            phase("tt_texture_fixed_guides")
            tex_fixed = tex(xtex, guide_subs=guides)
            report["texture_fixed"] = compare(tex_fixed, tex_ref)
            report["texture_fixed_stats"] = dict(tex._tenstorrent_backend.stats)
            save_sparse(args.output_dir / "texture_tt_fixed.pt", tex_fixed)
            phase("tt_shape_texture_chain")
            tex_free = tex(xtex, guide_subs=own_guides)
            report["texture_predicted"] = compare(tex_free, tex_ref)
            report["chain_coordinates_equal"] = torch.equal(tex_free.coords, shape_free.coords)
            report["warm_seconds"] = {"shape": [], "texture": [],
                "scope": "complete decoder with CPU boundary transfers, subdivision selection and map lookup; resident weights; after initial execution"}
            for _ in range(args.warm_repeats):
                start = time.perf_counter()
                _, warm_guides = shape.forward_features(xshape, return_subs=True)
                report["warm_seconds"]["shape"].append(time.perf_counter() - start)
                start = time.perf_counter()
                tex(xtex, guide_subs=warm_guides)
                report["warm_seconds"]["texture"].append(time.perf_counter() - start)
            report["topology_cache"] = {"builds": cache.builds, "hits": cache.hits}
            sign_match = (all(s["sign_mismatches"] == 0 for s in report["subdivisions_fixed"])
                          and report.get("geometry_fixed", {}).get("edge_sign_mismatches") == 0)
            report["status"] = "passed" if all(passed(report[k], report["gate"]) for k in (
                "shape_fixed", "shape_predicted", "texture_fixed", "texture_predicted")) and report["chain_coordinates_equal"] and sign_match and wide_ok else "numerical_failure"
            phase("complete")
        finally:
            shape.disable_tenstorrent()
            tex.disable_tenstorrent()
            ttnn.close_device(device)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Preserve the last completed phase and classify a runtime failure.
        # Numerical failures use the normal return path above.
        if "--output-dir" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--output-dir") + 1]) / "result.json"
            if path.exists():
                report = json.loads(path.read_text())
                report.update(status="runtime_failure", error=repr(exc), traceback=traceback.format_exc())
                path.write_text(json.dumps(report, indent=2) + "\n")
        raise
