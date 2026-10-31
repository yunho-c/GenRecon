"""Thin adapter for the official MapAnything API; model dependencies are lazy."""

import importlib.util
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..geometry import process_points
from ..scene_geometry import Frame, SceneGeometry, file_hash, request_fingerprint


def checkpoint_identity(config):
    """Resolve immutable identity without loading torch, weights, or the model."""
    checkpoint = config.get("checkpoint")
    if checkpoint:
        root = Path(checkpoint).expanduser().resolve()
        required = [root / "config.json", root / "model.safetensors"]
        if not all(p.is_file() for p in required):
            raise ValueError("--checkpoint requires a pretrained directory with config.json and model.safetensors")
        identity = {"files": {p.name: file_hash(p) for p in required}}
    else:
        revision = config.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("MapAnything requires an immutable 40-character model revision or --checkpoint")
        identity = {"model_id": config["model_id"], "revision": revision}
    spec = importlib.util.find_spec("mapanything")
    if spec is None:
        raise RuntimeError("MapAnything is not installed; follow docs/prepared-scenes.md in a separate frontend environment")
    root = Path(next(iter(spec.submodule_search_locations)))
    identity["implementation_sha256"] = request_fingerprint(
        {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))}
    )
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    identity["implementation_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    return identity


def preprocessing(source_size, target_size):
    """Exact PIL resize/center-crop pixel transform used by upstream load_images.

    Intrinsics use OpenCV centers: x_out = (x_in + .5) * scale - .5 - crop.
    Use actual floored dimensions, not the pre-rounding scalar resize factor.
    """
    source = np.asarray(source_size)
    target = np.asarray(target_size)
    resized = np.floor(source * (max(target / source) + 1e-8)).astype(int)
    offset = (resized - target) // 2
    scale = resized / source
    affine = np.array([[scale[0], 0, (scale[0] - 1) / 2 - offset[0]],
                       [0, scale[1], (scale[1] - 1) / 2 - offset[1]], [0, 0, 1]])
    return affine, {"oriented_source_size": source.tolist(), "resized_size": resized.tolist(),
                    "crop_offset": offset.tolist(), "processed_size": target.tolist(),
                    "source_to_processed": affine.tolist(), "pixel_convention": "opencv_half_pixel"}


def adapt_predictions(frames, predictions, config):
    """Conversion boundary also exercised by small CPU fixtures."""
    if len(predictions) != len(frames):
        raise ValueError("MapAnything output/frame count mismatch")
    result, intrinsics, poses, points, counts = [], [], [], [], []
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    for i, (source, pred) in enumerate(zip(frames, predictions)):
        def array(key, shape):
            value = pred[key]
            if hasattr(value, "detach"):
                value = value.detach().float().cpu().numpy()
            value = np.asarray(value)
            if value.shape != (1, *shape):
                raise ValueError(f"MapAnything {key} shape {value.shape}; expected {(1, *shape)}")
            return value[0]
        rgb_shape = tuple(pred["img_no_norm"].shape)
        if len(rgb_shape) != 4 or rgb_shape[0] != 1 or rgb_shape[-1] != 3:
            raise ValueError("MapAnything img_no_norm must be [1,H,W,3]")
        h, w = rgb_shape[1:3]
        rgb = array("img_no_norm", (h, w, 3))
        k = array("intrinsics", (3, 3))
        pose = array("camera_poses", (4, 4))
        xyz = array("pts3d", (h, w, 3))
        depth = array("depth_z", (h, w, 1))[..., 0]
        mask = array("mask", (h, w, 1))[..., 0]
        valid = np.isfinite(xyz).all(-1) & np.isfinite(depth) & (depth > 0) & np.isfinite(mask) & (mask > 0)
        points.append(xyz[valid])
        counts.append({"input": h * w, "retained": int(valid.sum())})
        if not np.isfinite(rgb).all():
            raise ValueError(f"nonfinite processed image: {source}")
        with Image.open(source) as original:
            exif = int(original.getexif().get(274, 1))
            original_size = list(original.size)
            oriented = ImageOps.exif_transpose(original).convert("RGB")
        affine, info = preprocessing(oriented.size, (w, h))
        info.update(source_size=original_size, exif_orientation=exif, conditioning=config["conditioning"])
        if config["conditioning"] == "source":
            image = oriented
            k = np.linalg.inv(affine) @ k
        else:
            image = Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
        output = work / f"frame_{i:06d}.png"
        image.save(output)
        result.append(Frame(str(i), str(output), *image.size, str(source), info))
        intrinsics.append(k)
        poses.append(pose)
    points, policy = process_points(np.concatenate(points), voxel_fraction=config.get("voxel_fraction", 0),
                                    max_points=config["max_points"])
    policy.update(per_frame=counts, min_points_per_chunk=config.get("min_points_per_chunk", 1),
                  mask_edges=config["mask_edges"], apply_confidence_mask=config["apply_confidence_mask"],
                  confidence_percentile=config["confidence_percentile"])
    scene = SceneGeometry(result, np.asarray(intrinsics, dtype=np.float32), np.asarray(poses, dtype=np.float32),
                          points, scale_status=config.get("scale_status", "predicted_metric"),
                          point_policy=policy, provenance={"backend": "mapanything"})
    scene.validate()
    return scene


def infer(frames, config):
    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
    except ImportError as error:
        raise RuntimeError(f"MapAnything dependency missing: {error}. See docs/prepared-scenes.md") from error
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for MapAnything but unavailable; use a CUDA environment or --device cpu")
    if device.type == "cuda" and config["amp_dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unsupported on this GPU; select fp32")
    report = config.get("run_report", {})
    report["stage"] = "resolve_checkpoint"
    start = time.perf_counter()
    checkpoint = config.get("checkpoint")
    cache_hit = True
    if not checkpoint:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        kwargs = dict(repo_id=config["model_id"], revision=config["revision"], allow_patterns=["config.json", "model.safetensors"])
        try:
            checkpoint = snapshot_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError:
            cache_hit = False
            checkpoint = snapshot_download(**kwargs)
    timings = {"checkpoint_resolution_seconds": time.perf_counter() - start}
    report["stage"] = "load_model"
    start = time.perf_counter()
    model = MapAnything.from_pretrained(str(checkpoint)).to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["model_load_seconds"] = time.perf_counter() - start
    report["stage"] = "preprocess"
    start = time.perf_counter()
    # Prevalidate all inputs: upstream otherwise silently skips unreadable images.
    for frame in frames:
        with Image.open(frame) as image:
            image.load()
    views = load_images([str(p) for p in frames], resolution_set=config["resolution_set"])
    if len(views) != len(frames) or [v["idx"] for v in views] != list(range(len(frames))):
        raise ValueError("MapAnything preprocessing skipped or reordered input images")
    timings["preprocessing_seconds"] = time.perf_counter() - start
    report["stage"] = "inference"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                  use_amp=config["amp_dtype"] != "fp32", amp_dtype=config["amp_dtype"],
                                  apply_mask=True, mask_edges=config["mask_edges"],
                                  apply_confidence_mask=config["apply_confidence_mask"],
                                  confidence_percentile=config["confidence_percentile"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["inference_seconds"] = time.perf_counter() - start
    report["stage"] = "convert_predictions"
    start = time.perf_counter()
    scene = adapt_predictions(frames, predictions, config)
    timings["conversion_seconds"] = time.perf_counter() - start
    scene.provenance.update(device=str(device), timings=timings, checkpoint_cache_hit=cache_hit,
                            checkpoint=config["checkpoint_identity"],
                            hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    if device.type == "cuda":
        scene.provenance.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    report.update(scene.provenance)
    return scene
