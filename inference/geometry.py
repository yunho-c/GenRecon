"""NumPy-only geometry at the frontend boundary.

Camera axes are OpenCV (right, down, forward); intrinsics are in pixels.
Depth is camera z, not distance along the ray. Pixel sample centers default
to integer coordinates; adapters must specify a different offset explicitly.
"""

from __future__ import annotations

import numpy as np


def homogeneous(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-2:] == (4, 4):
        result = poses.copy()
    elif poses.shape[-2:] == (3, 4):
        result = np.zeros((*poses.shape[:-2], 4, 4), dtype=np.float64)
        result[..., :3, :] = poses
        result[..., 3, 3] = 1
    else:
        raise ValueError("poses must end in [3, 4] or [4, 4]")
    if not np.isfinite(result).all() or not np.allclose(result[..., 3, :], [0, 0, 0, 1]):
        raise ValueError("poses must be finite homogeneous transforms")
    return result


def invert_poses(poses: np.ndarray) -> np.ndarray:
    return np.linalg.inv(homogeneous(poses))


def normalize_intrinsics(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return np.diag([1.0 / width, 1.0 / height, 1.0]) @ np.asarray(intrinsics)


def image_affine(*, crop_xy=(0, 0), scale_xy=(1, 1), pad_xy=(0, 0), half_pixel=False) -> np.ndarray:
    """Map source pixels through crop, resize, then padding.

    With half_pixel=True resizing maps (u + .5) * scale - .5, as in
    OpenCV/PyTorch half-pixel sampling. False is pure coordinate scaling
    (the legacy GenRecon calibration convention). Pass actual rounded sizes
    as scale ratios. This cannot represent lens undistortion or EXIF rotation.
    """
    cx, cy = crop_xy
    sx, sy = scale_xy
    px, py = pad_xy
    if not np.isfinite([cx, cy, sx, sy, px, py]).all() or min(sx, sy) <= 0:
        raise ValueError("image transform requires finite values and positive scales")
    offset = 0.5 if half_pixel else 0.0
    return np.array(
        [[sx, 0, px - sx * cx + offset * (sx - 1)], [0, sy, py - sy * cy + offset * (sy - 1)], [0, 0, 1]],
        dtype=np.float64,
    )


def transform_intrinsics(intrinsics: np.ndarray, source_to_image: np.ndarray) -> np.ndarray:
    return np.asarray(source_to_image) @ np.asarray(intrinsics)


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, *, pixel_offset=0.0):
    """Return [H, W, 3] world points and a validity mask for one z-depth map."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be [H, W]")
    if np.asarray(intrinsics).shape != (3, 3) or np.asarray(c2w).shape != (4, 4):
        raise ValueError("unprojection requires one [3, 3] K and [4, 4] c2w")
    y, x = np.indices(depth.shape, dtype=np.float64)
    pixels = np.stack([x + pixel_offset, y + pixel_offset, np.ones_like(x)], axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    camera = rays * depth[..., None]
    world = camera @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3]
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(world).all(axis=-1)
    return world, valid


def up_rotation(up_vector) -> np.ndarray:
    """Proper rotation mapping a supplied source-world up direction to +Z."""
    up = np.asarray(up_vector, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-12:
        raise ValueError("up vector must contain three finite, nonzero components")
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(up @ target, -1, 1))
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(up, target)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def align_world(points, c2w, up_vector, scale=1.0):
    """Rotate/scale points and camera centers; camera rotations stay rigid."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("world scale must be finite and positive")
    rotation = up_rotation(up_vector)
    poses = homogeneous(c2w)
    poses[..., :3, :3] = rotation @ poses[..., :3, :3]
    poses[..., :3, 3] = scale * (poses[..., :3, 3] @ rotation.T)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    return np.asarray(points) @ (scale * rotation).T, poses, transform


def process_points(points, *, voxel_fraction=0.0, max_points=1_000_000):
    """Finite filtering, deterministic voxel representatives, then a fixed cap.

    Voxel width is a fraction of the largest 1st-to-99th percentile extent.
    No metric units or model confidence calibration are assumed.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [P, 3]")
    if not np.isfinite(voxel_fraction) or voxel_fraction < 0 or max_points < 1:
        raise ValueError("voxel_fraction must be nonnegative and max_points positive")
    stats = {"input": len(points), "voxel_fraction": voxel_fraction, "max_points": max_points}
    points = points[np.isfinite(points).all(axis=1)]
    stats["finite"] = len(points)
    if not len(points):
        raise ValueError("point processing left no finite points")
    if voxel_fraction:
        extent = float(np.max(np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)))
        if extent <= 1e-12:
            raise ValueError("point cloud has degenerate extent")
        width = extent * voxel_fraction
        cells = np.floor((points - np.min(points, axis=0)) / width)
        _, indices = np.unique(cells, axis=0, return_index=True)
        points = points[np.sort(indices)]
        stats["voxel_size"] = width
    stats["after_voxel"] = len(points)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).round().astype(int)]
    stats["output"] = len(points)
    return points.astype(np.float32), stats
