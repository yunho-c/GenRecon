import json
import shutil

import numpy as np
import pytest
from PIL import Image

from inference.geometry import (
    align_world,
    homogeneous,
    image_affine,
    invert_poses,
    normalize_intrinsics,
    process_points,
    transform_intrinsics,
    unproject_depth,
    up_rotation,
)
from inference.scene_geometry import (
    read_scene_bundle,
    request_fingerprint,
    write_scene_bundle,
)


def project(points, k, c2w):
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    uv = camera @ k.T
    return uv[:, :2] / uv[:, 2:]


def test_unprojection_on_known_plane_and_invalid_depth():
    k = np.array([[2, 0, 1], [0, 4, 0.5], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = [10, 20, 30]
    depth = np.full((2, 3), 4.0)
    world, valid = unproject_depth(depth, k, pose)
    np.testing.assert_allclose(world[0, 0], [8, 19.5, 34])
    np.testing.assert_allclose(world[1, 2], [12, 20.5, 34])
    assert valid.all()
    depth[0] = [0, np.nan, -1]
    _, valid = unproject_depth(depth, k, pose)
    assert not valid[0].any() and valid[1].all()


@pytest.mark.parametrize("half_pixel", [False, True])
def test_image_coordinate_transform(half_pixel):
    k = np.array([[80, 0, 35], [0, 60, 20], [0, 0, 1]], dtype=float)
    point = np.array([[1, 2, 10.0]])
    uv = project(point, k, np.eye(4))[0]
    affine = image_affine(crop_xy=(7, 3), scale_xy=(0.5, 2), pad_xy=(4, 5), half_pixel=half_pixel)
    expected = (uv - [7, 3] + (0.5 if half_pixel else 0)) * [0.5, 2] + [4, 5] - (0.5 if half_pixel else 0)
    new_k = transform_intrinsics(k, affine)
    np.testing.assert_allclose(project(point, new_k, np.eye(4))[0], expected)
    np.testing.assert_allclose(transform_intrinsics(new_k, np.linalg.inv(affine)), k)
    np.testing.assert_allclose(normalize_intrinsics(new_k, 48, 64)[:2], new_k[:2] / [[48], [64]])


@pytest.mark.parametrize("up", [[1, 2, 3], [0, 0, -1], [0, 0, 1]])
def test_world_alignment_preserves_reprojection(scene, up):
    points, poses, transform = align_world(scene.points, scene.c2w, up, 2.7)
    np.testing.assert_allclose(up_rotation(up) @ (np.array(up) / np.linalg.norm(up)), [0, 0, 1], atol=1e-12)
    for old_pose, new_pose, k in zip(scene.c2w, poses, scene.intrinsics):
        np.testing.assert_allclose(project(points, k, new_pose), project(scene.points, k, old_pose), atol=1e-5)
        np.testing.assert_allclose(new_pose[:3, :3].T @ new_pose[:3, :3], np.eye(3), atol=1e-6)
    back = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(transform).T
    np.testing.assert_allclose(back[:, :3], scene.points, atol=1e-6)
    np.testing.assert_allclose(invert_poses(poses) @ poses, np.broadcast_to(np.eye(4), poses.shape), atol=1e-12)
    np.testing.assert_allclose(homogeneous(poses[:, :3]), poses)


def test_point_filtering_is_deterministic_and_bounded():
    points = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2], [np.nan, 0, 0]])
    a, stats = process_points(points, voxel_fraction=0.1, max_points=2)
    b, _ = process_points(points, voxel_fraction=0.1, max_points=2)
    np.testing.assert_array_equal(a, [[0, 0, 0], [2, 2, 2]])
    np.testing.assert_array_equal(a, b)
    assert stats["finite"] == 4 and stats["after_voxel"] == 3 and stats["output"] == 2


def test_bundle_roundtrip_portable_and_reusable(scene, tmp_path):
    bundle = write_scene_bundle(scene, tmp_path / "bundle", request={"model": "fixture", "seed": 3})
    assert write_scene_bundle(scene, bundle, request={"seed": 3, "model": "fixture"}) == bundle
    moved = tmp_path / "moved"
    shutil.copytree(bundle, moved)
    for frame in scene.frames:
        from pathlib import Path

        Path(frame.image_path).unlink()
    restored = read_scene_bundle(moved)
    for name in ["intrinsics", "c2w", "points", "source_to_scene"]:
        np.testing.assert_array_equal(getattr(restored, name), getattr(scene, name))
    assert all(str(moved) in f.image_path for f in restored.frames)
    assert [f.id for f in restored.frames] == ["11", "29"]
    with pytest.raises(ValueError, match="fingerprint"):
        read_scene_bundle(moved, expected_fingerprint=request_fingerprint({"model": "different"}))


@pytest.mark.parametrize("target", ["geometry.npz", "images/000000.png"])
def test_bundle_detects_corruption(scene, tmp_path, target):
    bundle = write_scene_bundle(scene, tmp_path / "bundle", request={})
    with (bundle / target).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        read_scene_bundle(bundle)


@pytest.mark.parametrize("change", ["version", "incomplete", "missing_file", "traversal", "inverse"])
def test_bundle_rejects_bad_manifests(scene, tmp_path, change):
    bundle = write_scene_bundle(scene, tmp_path / "bundle", request={})
    path = bundle / "scene.json"
    data = json.loads(path.read_text())
    if change == "version":
        data["schema_version"] = 100
    elif change == "incomplete":
        data["complete"] = False
    elif change == "missing_file":
        del data["files"]["geometry.npz"]
    elif change == "traversal":
        data["frames"][0]["image_path"] = "../outside.png"
        data["files"]["../outside.png"] = data["files"].pop("images/000000.png")
    else:
        data["scene_to_source"][0][3] = 5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_scene_bundle(bundle)


@pytest.mark.parametrize(
    "change", ["unaligned", "nan", "empty", "rotation", "focal", "count", "dimensions", "duplicates"]
)
def test_validation_failures(scene, tmp_path, change):
    if change == "unaligned":
        scene.aligned = False
    elif change == "nan":
        scene.points[0, 0] = np.nan
    elif change == "empty":
        scene.points = np.empty((0, 3))
    elif change == "rotation":
        scene.c2w[0, :3, :3] *= 2
    elif change == "focal":
        scene.intrinsics[0, 0, 0] = 0
    elif change == "count":
        scene.intrinsics = scene.intrinsics[:1]
    elif change == "dimensions":
        scene.frames[0].width += 1
    else:
        scene.frames[0].id = scene.frames[1].id
    with pytest.raises(ValueError):
        write_scene_bundle(scene, tmp_path / "invalid", request={})
    assert not (tmp_path / "invalid").exists()


def test_exif_orientation_requires_explicit_normalization(scene):
    path = scene.frames[0].image_path
    with Image.open(path) as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    with pytest.raises(ValueError, match="EXIF"):
        scene.validate()


def test_failed_publication_leaves_no_complete_bundle(scene, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="injected"):
        write_scene_bundle(scene, tmp_path / "bundle", request={})
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-*"))
