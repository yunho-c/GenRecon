"""Import calibrated images into a validated, reusable GenRecon scene bundle."""

import argparse
import json
import platform
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
    parser.add_argument("--model_dir", required=True, help="Directory containing COLMAP text files")
    parser.add_argument("--images", required=True, help="Explicit root for COLMAP image names")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", help="JSON list of image names relative to --images, in desired order")
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


def prepare(args):
    start = time.perf_counter()
    if args.min_points_per_chunk < 1 or args.max_points < 1 or args.min_track_len < 0:
        raise ValueError("point counts must be positive and minimum track length nonnegative")
    if not np.isfinite([args.stat_std_ratio, args.radius, args.voxel_fraction, args.scale]).all():
        raise ValueError("preparation options must be finite")
    if min(args.stat_std_ratio, args.radius, args.scale) <= 0 or args.radius_nb_points < 1 or args.voxel_fraction < 0:
        raise ValueError("invalid cleaning, voxel, or scale option")
    root, model_dir = Path(args.images).resolve(), Path(args.model_dir).resolve()
    records = image_records(model_dir)
    names = json.loads(Path(args.frames).read_text()) if args.frames else [r["name"] for r in records]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("frame manifest must be a list of unique image names")
    if len(names) < 2:
        raise ValueError("preparation requires at least two selected images")
    frames = [image_path(root, name) for name in names]
    config = {k: v for k, v in vars(args).items() if k not in {"output", "frames", "images", "model_dir"}}
    packages = {"python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version}
    for name in ["opencv-python", "opencv-python-headless"] + (["open3d"] if args.cleaning == "legacy" else []):
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
        "inference/frontends/colmap.py",
    ]
    request = {
        "backend": args.frontend,
        "options": config,
        "packages": packages,
        "model_files": {name: file_hash(model_dir / name) for name in ["cameras.txt", "images.txt", "points3D.txt"]},
        "frames": [{"name": name, "sha256": file_hash(path)} for name, path in zip(names, frames)],
        "adapter_sources": {name: file_hash(source_dir / name) for name in code_files},
    }
    if Path(args.output).exists():
        read_scene_bundle(args.output, expected_fingerprint=request_fingerprint(request))
        print(f"Reusing validated bundle: {args.output}")
        return
    with tempfile.TemporaryDirectory(prefix="genrecon-prepare-") as work:
        scene = get_frontend(args.frontend)(
            frames, {**config, "model_dir": model_dir, "image_root": root, "work_dir": work}
        )
        scene = scene.align(args.up_vector if args.up_vector is not None else [0, 0, 1], args.scale)
        scene.provenance.update(
            {
                "request_fingerprint": request_fingerprint(request),
                "packages": packages,
                "device": "cpu",
                "prepare_before_serialization_seconds": time.perf_counter() - start,
            }
        )
        output = write_scene_bundle(scene, args.output, request=request)
    print(f"Prepared {len(scene.frames)} cameras, {len(scene.points)} points: {output}")
    print(f"Complete preparation wall time: {time.perf_counter() - start:.3f}s")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
