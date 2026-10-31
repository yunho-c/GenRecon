"""The GPU smoke exposed a failure in generic per-face binary serialization."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

plyfile = pytest.importorskip("plyfile")
spec = importlib.util.spec_from_file_location("mesh_utils", Path(__file__).parents[1] / "genrecon/utils/mesh_utils.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("ascii", [False, True])
def test_pbr_schema_roundtrip(tmp_path, ascii):
    path = tmp_path / "mesh.ply"
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    channels = np.array([0, 127, 255], dtype=np.uint8)
    module.write_pbr_ply(path, vertices, faces, color, channels, channels, channels, ascii=ascii)
    result = plyfile.PlyData.read(path)
    for i, name in enumerate(["x", "y", "z"]):
        np.testing.assert_equal(result["vertex"][name], vertices[:, i])
    for i, name in enumerate(["red", "green", "blue"]):
        np.testing.assert_equal(result["vertex"][name], color[:, i])
    for name in ["metallic", "roughness", "alpha"]:
        np.testing.assert_equal(result["vertex"][name], channels)
    np.testing.assert_equal(result["face"]["vertex_indices"][0], faces[0])
