"""CPU checks for scene-frame alignment and camera-aware image preprocessing."""

import json
import struct
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from experiments.tenstorrent.generate_decoder_fixture import colmap_to_scene, load_views


def test_colmap_world_alignment(tmp_path):
    rotation = Rotation.from_euler("xyz", [20, -30, 40], degrees=True).as_matrix()
    translation = np.array([2., -3., 5.])
    scale = 1.7
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    frames = []
    with (tmp_path / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(centers)))
        for index, center in enumerate(centers):
            name = f"frame_{index}.png"
            stream.write(struct.pack("<i7di", index, 1., 0., 0., 0., *(-center), 1))
            stream.write(name.encode() + b"\0" + struct.pack("<Q", 0))
            c2w = np.eye(4)
            c2w[:3, :3] = rotation @ np.diag([1., -1., -1.])
            c2w[:3, 3] = scale * rotation @ center + translation
            frames.append({"file_path": "images/" + name, "transform_matrix": c2w.tolist()})
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps({"frames": frames}))
    actual = colmap_to_scene(tmp_path / "images.bin", path)
    np.testing.assert_allclose(actual[:3, :3], scale * rotation, atol=1e-12)
    np.testing.assert_allclose(actual[:3, 3], translation, atol=1e-12)


def test_cube_camera_projection_and_principal_point_crop(tmp_path):
    Image.new("RGB", (16, 10), (10, 20, 30)).save(tmp_path / "image.png")
    c2w = np.eye(4)
    c2w[:3, 3] = [1., 2., 5.]
    data = {"w": 16, "h": 10, "fl_x": 12., "fl_y": 12., "cx": 8., "cy": 5.,
        "camera_model": "OPENCV", "frames": [{"file_path": "images/image.png",
        "transform_matrix": c2w.tolist()}]}
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(data))
    images, intrinsics, extrinsics, sources = load_views(path, tmp_path, 1, [1., 2., 3.], 4.)
    assert tuple(images.shape) == (1, 3, 512, 512)
    torch.testing.assert_close(images[0, :, 0, 0], torch.tensor([10., 20., 30.]) / 255)
    camera = extrinsics[0, 0] @ torch.tensor([0., 0., 0., 1.])
    torch.testing.assert_close(camera, torch.tensor([0., 0., 2., 1.]))
    uv = intrinsics[0, 0] @ (camera[:3] / camera[2])
    torch.testing.assert_close(uv, torch.tensor([0.5, 0.5, 1.]))
    assert len(sources) == 1 and len(sources[0]["sha256"]) == 64
