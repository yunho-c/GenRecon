import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.frontends.colmap import image_records, infer
from inference.get_chunks import IphoneChunker, PreparedChunker
from inference.get_images import IphoneImageSelecter, PreparedImageSelecter
from inference.scene_geometry import read_scene_bundle, write_scene_bundle

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def assert_selections_match(a, b):
    assert a.chunk_indices == b.chunk_indices
    for name in a.__dataclass_fields__:
        if name == "chunk_indices":
            continue
        left, right = getattr(a, name), getattr(b, name)
        if isinstance(left, list):
            left, right = torch.stack(left), torch.stack(right)
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("center_crop", [False, True])
def test_legacy_and_prepared_conditioning(colmap_scene, scene, tmp_path, center_crop):
    root, _, model, points = colmap_scene
    # Independent expected OpenCV pose: camera forward +Y, down -Z.
    np.testing.assert_allclose(scene.c2w[0, :3, :3], [[1, 0, 0], [0, 0, 1], [0, -1, 0]], atol=1e-6)
    np.testing.assert_allclose(scene.c2w[0, :3, 3], [-0.5, -3, 1], atol=1e-6)
    old_out, new_out = tmp_path / "legacy", tmp_path / "prepared"
    old_out.mkdir()
    legacy_chunker = IphoneChunker(
        colmap_subdir="colmap", skip_point_cleaning=True, min_points_per_chunk=1, chunk_size_factor=1.11
    )
    old = legacy_chunker.get_chunks(root, old_out)
    restored = read_scene_bundle(write_scene_bundle(scene, tmp_path / "bundle", request={}))
    new = PreparedChunker(min_points_per_chunk=1).get_chunks(restored, new_out)
    np.testing.assert_allclose(old[0], new[0], atol=1e-6)
    for a, b in zip(old[1:], new[1:]):
        torch.testing.assert_close(torch.stack(a), torch.stack(b))
    old_sel = IphoneImageSelecter(center_crop=center_crop).get_images(
        old[1], model / "cameras.txt", 2, old_out, seed=17
    )
    new_sel = PreparedImageSelecter(center_crop=center_crop).get_images(new[1], restored, 2, new_out, seed=17)
    assert_selections_match(old_sel, new_sel)
    metadata = json.loads((new_out / "cameras.json").read_text())
    expected_ids = ["11", "29"] if center_crop else ["11", "11", "29", "29"]
    assert [v["frame_id"] for v in metadata["scene"]] == expected_ids
    assert all(v["cond2d_view"]["frame_id"] in {"11", "29"} for v in metadata["chunks"])


def test_import_order_and_observation_lines(colmap_scene, tmp_path):
    _, images, model, _ = colmap_scene
    assert [r["id"] for r in image_records(model)] == ["11", "29"]
    frames = [images / "frame 001.png", images / "frame 002.png"]
    imported = infer(frames, {"model_dir": model, "image_root": images, "work_dir": tmp_path})
    assert [f.id for f in imported.frames] == ["29", "11"]
    (model / "images.txt").write_text((model / "images.txt").read_text().rstrip() + "\n")
    with pytest.raises(ValueError, match="POINTS2D"):
        image_records(model)


def test_undistortion_matches_legacy(colmap_scene, tmp_path):
    _, images, model, _ = colmap_scene
    (model / "cameras.txt").write_text("7 OPENCV 32 24 28 26 15 11 0.1 -0.02 0.001 -0.001\n")
    scene = infer([], {"model_dir": model, "image_root": images, "work_dir": tmp_path / "rectified"})
    old = IphoneImageSelecter()._get_cameras(model / "cameras.txt")
    from PIL import Image

    for frame, k, camera in zip(scene.frames, scene.intrinsics, old):
        normal = k.copy()
        normal[0] /= frame.width
        normal[1] /= frame.height
        np.testing.assert_allclose(normal, camera["intrinsics"].numpy(), atol=1e-6)
        np.testing.assert_array_equal(
            np.asarray(Image.open(frame.image_path)), np.asarray(Image.open(camera["img_path"]))
        )


def test_chunk_failure_limits(scene, tmp_path):
    with pytest.raises(ValueError, match="max_chunks"):
        PreparedChunker(min_points_per_chunk=1, max_chunks=1).get_chunks(scene, tmp_path / "limit")
    with pytest.raises(ValueError, match="no chunks"):
        PreparedChunker(min_points_per_chunk=1_000_000).get_chunks(scene, tmp_path / "empty")
    scene.points[:, 2] = 0
    with pytest.raises(ValueError, match="vertical extent"):
        PreparedChunker(min_points_per_chunk=1).get_chunks(scene, tmp_path / "flat")


def test_cli_prepare_cache_and_preflight(colmap_scene, tmp_path):
    _, images, model, _ = colmap_scene
    bundle, output = tmp_path / "bundle", tmp_path / "result"
    command = [
        sys.executable,
        "prepare_scene.py",
        "--model_dir",
        str(model),
        "--images",
        str(images),
        "--output",
        str(bundle),
        "--assume_z_up",
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0 and "Reusing validated" in second.stdout
    mismatch = subprocess.run(command + ["--scale", "2"], cwd=ROOT, capture_output=True, text=True)
    assert mismatch.returncode != 0 and "fingerprint" in mismatch.stderr
    # Block generation and optional backend imports, rather than merely relying
    # on their absence from this machine. Preflight must remain model-free.
    code = """
import importlib.abc, sys, runpy
import torch
torch.set_num_threads(2)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("genrecon.pipelines.full_scene_images_to_3d", "inference.transform_to_original", "open3d", "mapanything", "vggt_omega", "depth_anything_3")):
            raise AssertionError("Forbidden preflight import: " + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ["reconstruct_scene.py", "--mode", "Prepared", "--path", sys.argv[1], "--output_path", sys.argv[2], "--preflight"]
runpy.run_path("reconstruct_scene.py", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(bundle), str(output)], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "preflight.json").read_text())
    assert summary["chunks"] > 0 and summary["scene_views"] == 4 and summary["generation_run"] is False
    assert summary["status"] == "complete"
    assert (output / "scene" / "view_000.png").exists()
    assert (output / "clean_points.ply").exists()
    coordinates = json.loads((output / "scene_coordinates.json").read_text())
    assert coordinates["scale_status"] == "arbitrary"
    assert coordinates["source_to_scene"] == np.eye(4).tolist()
    assert not (output / "mesh.ply").exists()
    with (bundle / "geometry.npz").open("ab") as stream:
        stream.write(b"corrupt")
    failure = subprocess.run(
        [sys.executable, "-c", code, str(bundle), str(output)], cwd=ROOT, capture_output=True, text=True
    )
    assert failure.returncode != 0 and "checksum" in failure.stderr
    assert json.loads((output / "preflight.json").read_text())["status"] != "complete"


@pytest.mark.parametrize(
    "model_line",
    [
        "7 SIMPLE_PINHOLE 32 24 28 15 11",
        "7 SIMPLE_RADIAL 32 24 28 15 11 0",
        "7 RADIAL 32 24 28 15 11 0 0",
    ],
)
def test_supported_pinhole_equivalents(colmap_scene, tmp_path, model_line):
    _, images, model, _ = colmap_scene
    (model / "cameras.txt").write_text(model_line + "\n")
    scene = infer([], {"model_dir": model, "image_root": images, "work_dir": tmp_path})
    np.testing.assert_array_equal(scene.intrinsics[0], [[28, 0, 15], [0, 28, 11], [0, 0, 1]])


def test_import_rejects_unsupported_cameras_and_missing_frames(colmap_scene, tmp_path):
    _, images, model, _ = colmap_scene
    config = {"model_dir": model, "image_root": images, "work_dir": tmp_path}
    with pytest.raises(ValueError, match="no COLMAP camera"):
        infer([images / "missing.png"], config)
    (model / "cameras.txt").write_text("7 OPENCV_FISHEYE 32 24 28 26 15 11 0 0 0 0\n")
    with pytest.raises(ValueError, match="unsupported camera model"):
        infer([], config)


def test_glb_inputs_keep_grid_frame_and_coordinate_metadata(tmp_path):
    from types import SimpleNamespace

    from reconstruct_scene import _save_to_glb_inputs

    mesh = SimpleNamespace(
        voxel_shape=(1, 1, 64, 64, 64),
        faces=torch.tensor([[0, 1, 2]]),
        attrs=torch.ones(1, 6),
        coords=torch.ones(1, 4),
        layout={"base_color": [0, 3]},
    )
    vertices = torch.tensor([[2.0, 3.0, 4.0], [3.0, 3.0, 4.0], [2.0, 4.0, 4.0]])
    coordinates = {
        "coordinate_frame": "aligned_z_up",
        "scene_to_source": [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    }
    _save_to_glb_inputs(
        tmp_path,
        mesh,
        vertices=vertices,
        voxel_size=0.1,
        origin=[1.0, 2.0, 3.0],
        label="test",
        scene_coordinates=coordinates,
    )
    result = torch.load(tmp_path / "to_glb_inputs.pt", weights_only=True)
    torch.testing.assert_close(result["vertices"], vertices)
    assert result["aabb"][0] == [1.0, 2.0, 3.0]
    assert result["scene_coordinates"] == coordinates


def test_cli_rejects_invalid_option_combinations(tmp_path):
    base = [
        sys.executable,
        "reconstruct_scene.py",
        "--path",
        str(tmp_path / "missing"),
        "--output_path",
        str(tmp_path / "result"),
        "--preflight",
    ]
    for mode, options, expected in [
        ("Sage_gt", ["--skip_point_cleaning"], "point-cloud mode"),
        ("Prepared", ["--skip_point_cleaning"], "already cleaned"),
        ("Prepared", ["--num_imgs_per_scene", "0"], "must be positive"),
    ]:
        result = subprocess.run(base + ["--mode", mode] + options, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode != 0 and expected in result.stderr


def test_empty_frame_manifest_does_not_import_everything(colmap_scene, tmp_path):
    _, images, model, _ = colmap_scene
    manifest = tmp_path / "frames.json"
    manifest.write_text("[]")
    result = subprocess.run(
        [
            sys.executable,
            "prepare_scene.py",
            "--model_dir",
            str(model),
            "--images",
            str(images),
            "--output",
            str(tmp_path / "bundle"),
            "--frames",
            str(manifest),
            "--assume_z_up",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "at least two" in result.stderr
    assert not (tmp_path / "bundle").exists()
