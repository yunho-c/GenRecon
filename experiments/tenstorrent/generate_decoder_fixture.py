"""Generate matched decoder inputs from a posed image scene using CPU inference.

Uses the original pretrained DINOv3-L, all three GenRecon flows, their original
Euler/CFG samplers, and the sparse-structure decoder. Stops before shape/texture
decoding or mesh extraction. No TT runtime or CUDA extension is required.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_CONV_BACKEND"] = "reference"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from experiments.tenstorrent.flow_probes import load_model
from experiments.tenstorrent.reference import float32_reference
from genrecon.modules.image_feature_extractor import DinoV3FeatureExtractor
from genrecon.modules.sparse import SparseTensor
from genrecon.pipelines.samplers import FlowEulerGuidanceIntervalSampler


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def colmap_to_scene(images_path, transforms_path):
    """Recover and verify the world-frame similarity from matching cameras.

    Nerfstudio exports can rotate, translate and scale COLMAP's world frame.
    Inferring that transform from the actual camera pairs avoids assuming a
    particular exporter version or applying camera-axis flips to world points.
    """
    from scipy.spatial.transform import Rotation
    cameras = {}
    with images_path.open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            values = struct.unpack("<i7di", stream.read(64))
            q, t = np.asarray(values[1:5]), np.asarray(values[5:8])
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char:
                    raise ValueError("truncated COLMAP image name")
                if char == b"\0":
                    break
                name.extend(char)
            cameras[Path(name.decode()).name] = (-rotation.T @ t,
                rotation.T @ np.diag([1., -1., -1.]))
            length, = struct.unpack("<Q", stream.read(8))
            stream.seek(length * 24, 1)
    pairs = [(cameras[Path(frame["file_path"]).name], np.asarray(frame["transform_matrix"]))
             for frame in json.loads(transforms_path.read_text())["frames"]
             if Path(frame["file_path"]).name in cameras]
    if len(pairs) < 2:
        raise ValueError("need at least two matching COLMAP/JSON cameras, or use an explicit cube")
    source = np.stack([pair[0][0] for pair in pairs])
    target = np.stack([pair[1][:3, 3] for pair in pairs])
    rotation = pairs[0][1][:3, :3] @ pairs[0][0][1].T
    a = (source - source.mean(0)) @ rotation.T
    b = target - target.mean(0)
    scale = float((a * b).sum() / (a * a).sum())
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    error = np.max(np.abs(scale * source @ rotation.T + translation - target))
    orientation_error = max(np.max(np.abs(rotation @ pair[0][1] - pair[1][:3, :3])) for pair in pairs)
    if not np.isfinite(scale) or scale <= 0 or error > 1e-4 * max(1., np.ptp(target, axis=0).max()) or orientation_error > 1e-4:
        raise ValueError("COLMAP and JSON cameras do not share a consistent similarity transform")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = scale * rotation, translation
    return transform


def point_bounds(path, transforms_path):
    """Align COLMAP points to the camera JSON, then define a bounded crop."""
    transform = colmap_to_scene(path.with_name("images.bin"), transforms_path)
    points = []
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            values = struct.unpack("<QdddBBBd", stream.read(43))
            points.append(values[1:4])
            length, = struct.unpack("<Q", stream.read(8))
            stream.seek(length * 8, 1)
    points = np.asarray(points)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("COLMAP points must be nonempty and finite")
    points = points @ transform[:3, :3].T + transform[:3, 3]
    lower, upper = np.quantile(points, [0.05, 0.95], axis=0)
    return (lower + upper) / 2, float((upper - lower).max() * 1.1), transform


def load_views(transforms_path, image_dir, views, center, extent):
    """Nerfstudio/OpenGL poses to normalized-cube OpenCV cameras, with crop K."""
    import cv2
    data = json.loads(transforms_path.read_text())
    frames = sorted(data["frames"], key=lambda f: f["file_path"])
    ids = np.linspace(0, len(frames) - 1, min(views, len(frames)), dtype=int)
    images, intrinsics, extrinsics, sources = [], [], [], []
    world_to_cube = np.eye(4)
    world_to_cube[:3, :3] /= extent
    world_to_cube[:3, 3] = -np.asarray(center) / extent
    for index in ids:
        frame = frames[index]
        cfg = {**data, **frame}
        if cfg.get("camera_model", "OPENCV") not in ("OPENCV", "PINHOLE"):
            raise ValueError("fixture preprocessing supports OPENCV/PINHOLE cameras only")
        source = image_dir / Path(frame["file_path"]).name
        image = np.asarray(Image.open(source).convert("RGB"))
        height, width = image.shape[:2]
        sx, sy = width / cfg["w"], height / cfg["h"]
        k = np.array([[cfg["fl_x"] * sx, 0., cfg["cx"] * sx],
                      [0., cfg["fl_y"] * sy, cfg["cy"] * sy], [0., 0., 1.]])
        distortion = np.array([cfg.get(key, 0.) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
        if np.any(distortion):
            image = cv2.undistort(image, k, distortion)
        # Match GenRecon's principal-point square crop and LANCZOS resize.
        size = int(2 * min(k[0, 2], width - k[0, 2], k[1, 2], height - k[1, 2]))
        if size <= 0:
            raise ValueError("principal point lies outside the image")
        left = max(0, min(round(k[0, 2] - size / 2), width - size))
        top = max(0, min(round(k[1, 2] - size / 2), height - size))
        image = Image.fromarray(image).crop((left, top, left + size, top + size))
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        k[0, 2] -= left
        k[1, 2] -= top
        k[:2] /= size
        c2w = np.asarray(frame["transform_matrix"])
        extr = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(world_to_cube @ c2w)
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
        intrinsics.append(k)
        extrinsics.append(extr)
        sources.append({"path": str(source), "sha256": sha256(source), "frame_index": int(index)})
    return (torch.stack(images), torch.tensor(np.stack(intrinsics), dtype=torch.float32)[None],
            torch.tensor(np.stack(extrinsics), dtype=torch.float32)[None], sources)


def conditioning(model, features, extrinsics, intrinsics, coords=None):
    x = None if coords is None else SparseTensor(torch.zeros(len(coords), 1), coords)
    projected = model.get_3D_cond(features, extrinsics, intrinsics, x)
    negative = (torch.zeros_like(projected) if isinstance(projected, torch.Tensor)
                else projected.replace(torch.zeros_like(projected.feats)))
    return dict(cond={"cond_2D": features[:, 0], "cond_3D": projected},
                neg_cond={"cond_2D": torch.zeros_like(features[:, 0]), "cond_3D": negative})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transforms", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--points3d", type=Path)
    p.add_argument("--center", nargs=3, type=float)
    p.add_argument("--extent", type=float)
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--checkpoints", type=Path, default=Path.home() / "models/genrecon")
    p.add_argument("--dino", type=Path, default=Path.home() / "models/genrecon/dinov3-vitl16")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.views < 1 or args.threads < 1 or ((args.center is None) != (args.extent is None)):
        p.error("views/threads must be positive; center and extent must be specified together")
    if args.center is None and args.points3d is None:
        p.error("supply a COLMAP points3D.bin or explicit cube center and extent")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "device": "cpu", "precision": "float32", "seed": args.seed,
        "scope": "one bounded scene cube, 512 output resolution; no mesh generation",
        "script_sha256": sha256(Path(__file__)), "transforms_sha256": sha256(args.transforms),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            ROOT / "genrecon/modules/image_feature_extractor.py", ROOT / "configs/pipelines/texture.json",
            ROOT / "experiments/tenstorrent/flow_probes.py", ROOT / "experiments/tenstorrent/reference.py"]},
        "source_transforms": str(args.transforms), "torch": torch.__version__}
    started = time.perf_counter()
    def phase(name):
        report["phase"] = name
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, flush=True)
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.center is None:
            center, extent, transform = point_bounds(args.points3d, args.transforms)
            report["colmap_to_scene"] = transform.tolist()
            report["colmap_cameras_sha256"] = sha256(args.points3d.with_name("images.bin"))
        else:
            center, extent = args.center, args.extent
        if not np.isfinite(extent) or extent <= 0 or not np.isfinite(center).all():
            raise ValueError("cube center must be finite and extent positive and finite")
        report.update(center=np.asarray(center).tolist(), extent=extent,
            cube_selection="COLMAP 5%-95% bounds, 10% margin" if args.center is None else "explicit")
        if args.points3d:
            report["points3d_sha256"] = sha256(args.points3d)
        phase("load_scene")
        images, intrinsics, extrinsics, sources = load_views(
            args.transforms, args.image_dir, args.views, center, extent)
        report["images"] = sources
        phase("dino_features")
        dino = DinoV3FeatureExtractor(str(args.dino))
        report["dino_sha256"] = sha256(args.dino / "model.safetensors")
        # Use the original extractor's exact preprocessing and layer path on CPU.
        with torch.no_grad():
            features = torch.cat([dino.extract_features(dino.transform(image[None])) for image in images])[None]
        torch.save({"features": features, "intrinsics": intrinsics, "extrinsics": extrinsics},
                   args.output_dir / "conditioning.pt")
        del dino
        gc.collect()
        cfg = json.loads((ROOT / "configs/pipelines/texture.json").read_text())["args"]
        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        report["sampling"] = {key: cfg[key]["params"] for key in
            ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler")}
        phase("load_ss_flow")
        model, _, provenance = load_model("ss", args.checkpoints)
        provenance.update(input_kind="posed real-image DINO conditioning", execution_dtype="float32")
        report["ss_checkpoint"] = provenance
        with torch.no_grad(), float32_reference(model):
            phase("sample_sparse_structure")
            cond = conditioning(model, features, extrinsics, intrinsics)
            noise = torch.randn(1, model.in_channels, *([model.resolution] * 3))
            latent = sampler.sample(model, noise, **cond, **cfg["sparse_structure_sampler"]["params"]).samples
        del model, cond
        gc.collect()
        phase("decode_sparse_structure")
        from genrecon import models
        decoder_path = args.checkpoints / "trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        report["ss_decoder_sha256"] = {suffix: sha256(decoder_path.with_suffix(suffix))
                                       for suffix in (".json", ".safetensors")}
        decoder = models.from_pretrained(str(decoder_path)).float().eval()
        decoder.convert_to_fp32()
        with torch.no_grad():
            occupied = F.max_pool3d((decoder(latent) > 0).float(), 2, 2) > 0
        coords = torch.argwhere(occupied)[:, [0, 2, 3, 4]].int()
        if not len(coords):
            raise RuntimeError("scene produced no occupied decoder-input voxels")
        report["active_voxels"] = len(coords)
        torch.save({"latent": latent, "coords": coords}, args.output_dir / "sparse_structure.pt")
        del decoder
        gc.collect()
        outputs = {}
        for stage, family in (("shape", "shape_slat"), ("tex", "tex_slat")):
            phase("load_" + stage + "_flow")
            model, _, provenance = load_model(stage, args.checkpoints)
            provenance.update(input_kind="posed real-image DINO conditioning", execution_dtype="float32")
            report[stage + "_checkpoint"] = provenance
            with torch.no_grad(), float32_reference(model):
                phase("sample_" + stage)
                cond = conditioning(model, features, extrinsics, intrinsics, coords)
                noise = SparseTensor(torch.randn(len(coords), 32), coords)
                extra = {} if stage == "shape" else {"concat_cond": outputs["shape_normalized"]}
                output = sampler.sample(model, noise, **cond, **extra, **cfg[family + "_sampler"]["params"]).samples
                if stage == "shape":
                    outputs["shape_normalized"] = output
                norm = cfg[family + "_normalization"]
                outputs[stage] = output.feats * torch.tensor(norm["std"])[None] + torch.tensor(norm["mean"])[None]
                if not torch.equal(output.coords, coords) or not torch.isfinite(outputs[stage]).all():
                    raise ValueError("sampler changed coordinate order or produced nonfinite features")
                torch.save({"coords": coords, "feats": outputs[stage]}, args.output_dir / (stage + "_slat.pt"))
            del model, cond
            gc.collect()
        path = args.output_dir / "decoder_input.npz"
        np.savez_compressed(path, coords=coords.numpy(), shape_feats=outputs["shape"].numpy(),
                            texture_feats=outputs["tex"].numpy())
        report.update(status="passed", input_sha256=sha256(path), denormalization_applied=True)
        phase("complete")
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        phase(report.get("phase", "initialization"))
        raise


if __name__ == "__main__":
    main()
