"""COLMAP text import with explicit image paths and optional rectification.

Format: https://colmap.github.io/format.html. images.txt order is preserved
unless an explicit frame sequence is supplied; image IDs are not timestamps.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ..geometry import invert_poses, process_points
from ..scene_geometry import Frame, SceneGeometry


def image_records(model_dir):
    records = []
    with (Path(model_dir) / "images.txt").open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError("invalid COLMAP images.txt header")
            # The second line may be empty. Do not strip blank lines before pairing.
            observations = stream.readline()
            while observations.lstrip().startswith("#"):
                observations = stream.readline()
            if not observations or len(observations.split()) % 3:
                raise ValueError(f"missing or invalid POINTS2D line for image {parts[0]}")
            records.append(
                {
                    "id": parts[0],
                    "quaternion": [float(v) for v in parts[1:5]],
                    "translation": [float(v) for v in parts[5:8]],
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                }
            )
    if len({r["id"] for r in records}) != len(records) or len({r["name"] for r in records}) != len(records):
        raise ValueError("COLMAP images have duplicate IDs or names")
    return records


def image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"COLMAP image path escapes image root: {name}")
    return path


def _camera_table(model_dir):
    table = {}
    lengths = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3, "OPENCV": 8, "RADIAL": 5, "SIMPLE_RADIAL": 4}
    with (Path(model_dir) / "cameras.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            camera_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            if model not in lengths:
                raise ValueError(f"unsupported camera model {model}; rectify to PINHOLE before import")
            params = [float(x) for x in parts[4:]]
            if len(params) != lengths[model] or not np.isfinite(params).all():
                raise ValueError(f"invalid parameters for camera {camera_id}")
            if camera_id in table or min(width, height) <= 0:
                raise ValueError(f"duplicate camera or invalid dimensions: {camera_id}")
            dist = None
            if model in {"PINHOLE", "OPENCV"}:
                fx, fy, cx, cy = params[:4]
                if model == "OPENCV":
                    dist = np.array(params[4:])
            else:
                focal, cx, cy = params[:3]
                fx = fy = focal
                if model == "RADIAL":
                    dist = np.array([params[3], params[4], 0, 0])
                elif model == "SIMPLE_RADIAL":
                    dist = np.array([params[3], 0, 0, 0])
            if min(fx, fy) <= 0:
                raise ValueError(f"nonpositive focal length for camera {camera_id}")
            table[camera_id] = (model, width, height, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), dist)
    return table


def _c2w(record):
    q = np.asarray(record["quaternion"])
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or abs(norm - 1) > 1e-3:
        raise ValueError(f"invalid unit quaternion for image {record['id']}")
    w, x, y, z = q / norm
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return invert_poses(np.column_stack([r, record["translation"]]))


def _read_points(model_dir, config):
    points = []
    total = 0
    max_error = config.get("max_reproj_error")
    min_track = config.get("min_track_len", 0)
    if min_track < 0 or (max_error is not None and (not np.isfinite(max_error) or max_error < 0)):
        raise ValueError("COLMAP point thresholds must be nonnegative")
    with (Path(model_dir) / "points3D.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = line.split()
            if len(row) < 8 or (len(row) - 8) % 2:
                raise ValueError("invalid COLMAP points3D.txt row")
            total += 1
            error = float(row[7])
            if (
                not np.isfinite(error)
                or (max_error is not None and error > max_error)
                or (len(row) - 8) // 2 < min_track
            ):
                continue
            points.append([float(v) for v in row[1:4]])
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    after_thresholds = len(points)
    if not after_thresholds:
        raise ValueError("COLMAP thresholds left no valid points")
    cleaning = config.get("cleaning", "none")
    if cleaning == "legacy":
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("legacy point cleaning requires open3d; install it or select --cleaning none") from error
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=config.get("stat_std_ratio", 2.0))
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=config.get("radius_nb_points", 10), radius=config.get("radius", 0.1)
        )
        points = np.asarray(cloud.points)
    elif cleaning != "none":
        raise ValueError(f"unknown point cleaning policy: {cleaning}")
    points, stats = process_points(
        points, voxel_fraction=config.get("voxel_fraction", 0.0), max_points=config.get("max_points", 1_000_000)
    )
    return points, {
        **stats,
        "colmap_input": total,
        "after_colmap_thresholds": after_thresholds,
        "cleaning": cleaning,
        "min_track_len": min_track,
        "max_reproj_error": max_error,
        "min_points_per_chunk": config.get("min_points_per_chunk", 1),
    }


def infer(frames, config):
    model_dir, root = Path(config["model_dir"]), Path(config["image_root"])
    records = image_records(model_dir)
    if frames:
        by_path = {image_path(root, r["name"]): r for r in records}
        try:
            records = [by_path[Path(p).resolve()] for p in frames]
        except KeyError as error:
            raise ValueError(f"requested image has no COLMAP camera: {error.args[0]}") from error
    table = _camera_table(model_dir)
    result_frames, intrinsics, poses = [], [], []
    for i, record in enumerate(records):
        source = image_path(root, record["name"])
        model, width, height, k, dist = table[record["camera_id"]]
        with Image.open(source) as image:
            if image.size != (width, height):
                raise ValueError(f"image dimensions do not match COLMAP camera: {source}")
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"EXIF rotation in {source}; normalize pixels and calibration explicitly")
        preprocessing = {
            "source_size": [width, height],
            "source_intrinsics": k.tolist(),
            "camera_model": model,
            "operation": "identity",
            "pixel_convention": "colmap",
        }
        output = source
        if dist is not None and np.any(dist):
            import cv2

            new_k, roi = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), alpha=0, newImgSize=(width, height))
            x, y, w, h = roi
            if min(w, h) <= 0:
                raise ValueError(f"empty rectification ROI for {source}")
            remap = cv2.initUndistortRectifyMap(k, dist, None, new_k, (width, height), cv2.CV_16SC2)
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode image {source}")
            rectified = cv2.remap(image, *remap, cv2.INTER_LINEAR)[y : y + h, x : x + w]
            output = Path(config["work_dir"]) / f"rectified_{i:06d}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), rectified):
                raise OSError(f"cannot save rectified image {output}")
            k = new_k.copy()
            k[0, 2] -= x
            k[1, 2] -= y
            width, height = w, h
            preprocessing.update(operation="opencv_undistort", distortion=dist.tolist(), roi=list(roi))
        result_frames.append(Frame(record["id"], str(output), width, height, str(source), preprocessing))
        intrinsics.append(k)
        poses.append(_c2w(record))
    points, policy = _read_points(model_dir, config)
    scene = SceneGeometry(
        result_frames,
        np.asarray(intrinsics, dtype=np.float32),
        np.asarray(poses, dtype=np.float32),
        points,
        scale_status=config.get("scale_status", "arbitrary"),
        point_policy=policy,
        provenance={"backend": "colmap", "frame_order": "explicit" if frames else "images.txt"},
    )
    scene.validate()
    return scene
