"""Import calibrated images into a validated, reusable GenRecon scene bundle."""

import argparse
import json
import platform
import sys
import resource
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from inference.frontends import available_frontends, get_frontend
from inference.frontends.colmap import image_path, image_records
from inference.scene_geometry import (
    file_hash,
    read_scene_bundle,
    request_fingerprint,
    write_scene_bundle,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", choices=available_frontends(), default="colmap")
    parser.add_argument("--model_dir", help="Directory containing COLMAP text files (COLMAP only)")
    parser.add_argument("--images", required=True, help="Explicit root for COLMAP image names")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", help="JSON list of image names relative to --images, in desired order")
    parser.add_argument("--config", help="JSON configuration; explicit CLI options override it")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--checkpoint", help="Local pretrained checkpoint directory")
    parser.add_argument("--model_id", default="facebook/map-anything")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditioning", choices=["processed", "source"], default="processed")
    parser.add_argument("--resolution_set", type=int, choices=[518], default=518)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--mask_edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apply_confidence_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--confidence_percentile", type=float, default=10.0)
    orientation = parser.add_mutually_exclusive_group(required=True)
    orientation.add_argument("--up_vector", type=float, nargs=3)
    orientation.add_argument("--assume_z_up", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--scale_status", choices=["arbitrary", "predicted_metric", "externally_calibrated"], default="arbitrary"
    )
    parser.add_argument("--cleaning", choices=["none", "legacy"], default="none")
    parser.add_argument("--max_reproj_error", type=float, default=None)
    parser.add_argument("--min_track_len", type=int, default=0)
    parser.add_argument("--stat_std_ratio", type=float, default=2.0)
    parser.add_argument("--radius_nb_points", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.1, help="Legacy radius in source world units")
    parser.add_argument("--voxel_fraction", type=float, default=0.0)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument(
        "--min_points_per_chunk", type=int, default=1, help="Explicit threshold after preparation; saved in bundle"
    )
    return parser


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    preliminary.add_argument("--frontend", default="colmap")
    selected, _ = preliminary.parse_known_args(argv)
    parser = build_parser()
    defaults = {}
    if selected.frontend == "mapanything":
        defaults = json.loads((Path(__file__).parent / "configs/frontends/mapanything.json").read_text())
        defaults["scale_status"] = "predicted_metric"
    if selected.config:
        supplied = json.loads(Path(selected.config).read_text())
        if not isinstance(supplied, dict):
            raise ValueError("configuration must be a JSON object")
        defaults.update(supplied)
    actions = {a.dest: a for a in parser._actions}
    for key, value in defaults.items():
        if key not in actions or key in {"help", "config", "frontend", "images", "output", "frames", "up_vector", "assume_z_up"}:
            raise ValueError(f"unsupported configuration field: {key}")
        action = actions[key]
        expected = action.type or (bool if isinstance(action, argparse.BooleanOptionalAction) else str)
        if value is not None and (not isinstance(value, expected) or (expected is int and isinstance(value, bool))):
            if not (expected is float and type(value) is int):
                raise ValueError(f"invalid configuration type for {key}")
        if action.choices and value not in action.choices:
            raise ValueError(f"invalid configuration value for {key}")
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def select_frames(args):
    root = Path(args.images).resolve()
    if args.frontend == "colmap" and not args.model_dir:
        raise ValueError("COLMAP requires --model_dir")
    if args.frames:
        names = json.loads(Path(args.frames).read_text())
    elif args.frontend == "colmap":
        names = [r["name"] for r in image_records(args.model_dir)]
    else:
        names = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or len(names) != len(set(names)):
        raise ValueError("frame manifest must be a list of unique image names")
    if args.max_frames is not None:
        if args.max_frames < 2:
            raise ValueError("max_frames must be at least two")
        if len(names) > args.max_frames:
            names = [names[i] for i in np.linspace(0, len(names) - 1, args.max_frames, dtype=int)]
    if len(names) < 2:
        raise ValueError("preparation requires at least two selected images")
    frames = [image_path(root, name) for name in names]
    if len(set(frames)) != len(frames):
        raise ValueError("selected frames resolve to duplicate files")
    return names, frames


def prepare(args, report=None):
    start = time.perf_counter()
    if args.min_points_per_chunk < 1 or args.max_points < 1 or args.min_track_len < 0:
        raise ValueError("point counts must be positive and minimum track length nonnegative")
    if not np.isfinite([args.stat_std_ratio, args.radius, args.voxel_fraction, args.scale]).all():
        raise ValueError("preparation options must be finite")
    if min(args.stat_std_ratio, args.radius, args.scale) <= 0 or args.radius_nb_points < 1 or args.voxel_fraction < 0:
        raise ValueError("invalid cleaning, voxel, or scale option")
    report = {} if report is None else report
    root = Path(args.images).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    names, frames = select_frames(args)
    config = {k: v for k, v in vars(args).items() if k not in {"output", "frames", "images", "model_dir", "config"}}
    if not np.isfinite(args.confidence_percentile) or not 0 <= args.confidence_percentile <= 100:
        raise ValueError("confidence_percentile must be between zero and 100")
    if args.frontend == "mapanything":
        if args.cleaning != "none" or args.max_reproj_error is not None or args.min_track_len:
            raise ValueError("COLMAP cleaning/track/error thresholds do not apply to MapAnything")
        from inference.frontends.mapanything import checkpoint_identity
        config["checkpoint_identity"] = checkpoint_identity(config)
    packages = {"python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version}
    for name in ["opencv-python", "opencv-python-headless"] + (["open3d"] if args.cleaning == "legacy" else []):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    if args.frontend == "mapanything":
        for name in ["torch", "mapanything", "huggingface-hub", "uniception", "safetensors"]:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
    source_dir = Path(__file__).resolve().parent
    code_files = [
        "prepare_scene.py",
        "inference/scene_geometry.py",
        "inference/geometry.py",
        "inference/frontends/__init__.py",
        f"inference/frontends/{args.frontend}.py",
    ]
    request = {
        "backend": args.frontend,
        "options": config,
        "packages": packages,
        "model_files": ({name: file_hash(model_dir / name) for name in ["cameras.txt", "images.txt", "points3D.txt"]} if args.frontend == "colmap" else config["checkpoint_identity"]),
        "frames": [{"name": name, "sha256": file_hash(path)} for name, path in zip(names, frames)],
        "adapter_sources": {name: file_hash(source_dir / name) for name in code_files},
    }
    if Path(args.output).exists():
        read_scene_bundle(args.output, expected_fingerprint=request_fingerprint(request))
        report.update(status="complete", cache_reused=True)
        print(f"Reusing validated bundle: {args.output}")
        return
    with tempfile.TemporaryDirectory(prefix="genrecon-prepare-") as work:
        scene = get_frontend(args.frontend)(
            frames, {**config, "model_dir": model_dir, "image_root": root, "work_dir": work, "run_report": report}
        )
        report["stage"] = "alignment"
        scene = scene.align(args.up_vector if args.up_vector is not None else [0, 0, 1], args.scale)
        scene.provenance.update(
            {
                "request_fingerprint": request_fingerprint(request),
                "packages": packages,
                "device": scene.provenance.get("device", "cpu"),
                "prepare_before_serialization_seconds": time.perf_counter() - start,
            }
        )
        report["stage"] = "publish_bundle"
        output = write_scene_bundle(scene, args.output, request=request)
    report.update(status="complete", cache_reused=False, point_policy=scene.point_policy, frames=len(scene.frames))
    print(f"Prepared {len(scene.frames)} cameras, {len(scene.points)} points: {output}")
    print(f"Complete preparation wall time: {time.perf_counter() - start:.3f}s")


def main():
    start = time.perf_counter()
    report = {"status": "running", "stage": "configuration"}
    args = None
    try:
        args = parse_args()
        report["stage"] = "validate_request"
        prepare(args, report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - start
        report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if args is not None:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Report is outside the immutable bundle; each attempt has its own record.
            path = output.with_name(output.name + f".run-{time.time_ns()}.json")
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
