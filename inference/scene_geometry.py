"""Portable scene bundles, independent of PyTorch and generation extensions."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import align_world

SCHEMA_VERSION = 1


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_fingerprint(request: dict) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Frame:
    id: str
    image_path: str
    width: int
    height: int
    source_path: str = ""
    preprocessing: dict = field(default_factory=dict)


@dataclass
class SceneGeometry:
    frames: list[Frame]
    intrinsics: np.ndarray
    c2w: np.ndarray
    points: np.ndarray
    aligned: bool = False
    source_to_scene: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    scale_status: str = "arbitrary"
    provenance: dict = field(default_factory=dict)
    point_policy: dict = field(default_factory=dict)

    def validate(self, *, require_aligned=False, check_images=True):
        n = len(self.frames)
        if n < 2 or len({f.id for f in self.frames}) != n or any(not f.id for f in self.frames):
            raise ValueError("scene requires at least two frames with unique nonempty IDs")
        for name, value, shape in (("intrinsics", self.intrinsics, (n, 3, 3)), ("c2w", self.c2w, (n, 4, 4))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        k = np.asarray(self.intrinsics)
        if np.any(k[:, 0, 0] <= 0) or np.any(k[:, 1, 1] <= 0) or not np.allclose(k[:, 2], [0, 0, 1]):
            raise ValueError("intrinsics require positive focal lengths and homogeneous last row")
        # Current square-crop code only adjusts focal lengths and principal points.
        if not np.allclose(k[:, 0, 1], 0) or not np.allclose(k[:, 1, 0], 0):
            raise ValueError("GenRecon currently requires zero-skew pinhole intrinsics")
        if np.any(np.abs(np.linalg.det(k)) < 1e-12):
            raise ValueError("intrinsics must be invertible")
        poses = np.asarray(self.c2w)
        r = poses[:, :3, :3]
        if (
            not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-4)
        ):
            raise ValueError("c2w must contain rigid, right-handed camera poses")
        p = np.asarray(self.points)
        if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
            raise ValueError("points must be a nonempty finite [P, 3] array")
        if np.max(np.ptp(p, axis=0)) < 1e-8:
            raise ValueError("point cloud has degenerate extent")
        t = np.asarray(self.source_to_scene)
        if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError("source_to_scene must be a finite homogeneous transform")
        scale = np.cbrt(np.linalg.det(t[:3, :3]))
        if (
            not np.isfinite(scale)
            or scale <= 0
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3) * scale**2, atol=1e-5)
        ):
            raise ValueError("source_to_scene must be a positive similarity transform")
        if self.scale_status not in {"arbitrary", "predicted_metric", "externally_calibrated"}:
            raise ValueError("unknown scale_status")
        if require_aligned and not self.aligned:
            raise ValueError("scene must be explicitly aligned to z-up before reconstruction")
        for frame in self.frames:
            if (
                not isinstance(frame.width, int)
                or not isinstance(frame.height, int)
                or min(frame.width, frame.height) <= 0
            ):
                raise ValueError(f"invalid image dimensions for frame {frame.id}")
            if check_images:
                with Image.open(frame.image_path) as image:
                    if image.size != (frame.width, frame.height):
                        raise ValueError(f"image dimensions do not match calibration for frame {frame.id}")
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError(
                            f"frame {frame.id} has EXIF rotation; normalize pixels and calibration explicitly"
                        )
                    image.load()
        json.dumps({"provenance": self.provenance, "point_policy": self.point_policy}, allow_nan=False)

    def align(self, up_vector, scale=1.0):
        self.validate()
        if self.aligned:
            raise ValueError("scene is already aligned")
        points, poses, transform = align_world(self.points, self.c2w, up_vector, scale)
        result = replace(
            self,
            points=points.astype(np.float32),
            c2w=poses.astype(np.float32),
            aligned=True,
            source_to_scene=(transform @ self.source_to_scene).astype(np.float32),
        )
        result.validate(require_aligned=True)
        return result


def _bundle_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"bundle path escapes its directory: {relative}")
    return resolved


def read_scene_bundle(path: str | Path, *, expected_fingerprint: str | None = None) -> SceneGeometry:
    root = Path(path)
    try:
        metadata = json.loads((root / "scene.json").read_text())
        if metadata["schema_version"] != SCHEMA_VERSION or metadata["complete"] is not True:
            raise ValueError("unsupported or incomplete scene bundle")
        fingerprint = request_fingerprint(metadata["request"])
        if fingerprint != metadata["fingerprint"] or (
            expected_fingerprint is not None and fingerprint != expected_fingerprint
        ):
            raise ValueError("scene bundle fingerprint does not match the requested inputs/options")
        if metadata["coordinate_frame"] != "z_up_opencv_cameras" or metadata["aligned"] is not True:
            raise ValueError("scene bundle is not aligned to z-up")
        expected_files = {"geometry.npz", *(f["image_path"] for f in metadata["frames"])}
        if set(metadata["files"]) != expected_files:
            raise ValueError("scene bundle file manifest is incomplete")
        for name, checksum in metadata["files"].items():
            if file_hash(_bundle_path(root, name)) != checksum:
                raise ValueError(f"scene bundle checksum mismatch: {name}")
        with np.load(root / "geometry.npz", allow_pickle=False) as arrays:
            geometry = {key: arrays[key] for key in ("intrinsics", "c2w", "points", "source_to_scene")}
        frames = [Frame(**{**f, "image_path": str(_bundle_path(root, f["image_path"]))}) for f in metadata["frames"]]
        scene = SceneGeometry(
            frames=frames,
            **geometry,
            aligned=True,
            scale_status=metadata["scale_status"],
            provenance={**metadata["provenance"], "request_fingerprint": fingerprint},
            point_policy=metadata["point_policy"],
        )
        scene.validate(require_aligned=True)
        if not np.allclose(np.linalg.inv(scene.source_to_scene), metadata["scene_to_source"], atol=1e-5):
            raise ValueError("scene-to-source transform does not match its inverse")
        return scene
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"Invalid scene bundle {root}: {error}") from error


def write_scene_bundle(scene: SceneGeometry, path: str | Path, *, request: dict) -> Path:
    """Publish a complete self-contained bundle; never replace an existing one."""
    scene.validate(require_aligned=True)
    root = Path(path).resolve()
    fingerprint = request_fingerprint(request)
    if root.exists():
        read_scene_bundle(root, expected_fingerprint=fingerprint)
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        (temporary / "images").mkdir()
        frames = []
        files = {}
        for i, frame in enumerate(scene.frames):
            relative = f"images/{i:06d}.png"
            # Lossless re-encoding preserves the decoded RGB/alpha pixels while
            # removing incidental image metadata and absolute path dependencies.
            with Image.open(frame.image_path) as image:
                image.convert("RGBA" if "A" in image.getbands() else "RGB").save(temporary / relative)
            frames.append({**asdict(frame), "image_path": relative})
            files[relative] = file_hash(temporary / relative)
        np.savez(
            temporary / "geometry.npz",
            **{
                name: np.asarray(getattr(scene, name), dtype=np.float32)
                for name in ("intrinsics", "c2w", "points", "source_to_scene")
            },
        )
        files["geometry.npz"] = file_hash(temporary / "geometry.npz")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "aligned": True,
            "coordinate_frame": "z_up_opencv_cameras",
            "scale_status": scene.scale_status,
            "scene_to_source": np.linalg.inv(np.asarray(scene.source_to_scene, dtype=np.float32)).tolist(),
            "frames": frames,
            "files": files,
            "request": request,
            "fingerprint": fingerprint,
            "provenance": scene.provenance,
            "point_policy": scene.point_policy,
        }
        (temporary / "scene.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
        read_scene_bundle(temporary, expected_fingerprint=fingerprint)
        if root.exists():
            raise FileExistsError(f"output appeared while preparing scene: {root}")
        temporary.rename(root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root
