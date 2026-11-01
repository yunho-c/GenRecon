"""Coordinate-intersection diagnostics must not hide topology differences."""

import torch
from genrecon.modules.sparse import SparseTensor
from experiments.tenstorrent.sparse_decoder_scene_probe import scene_compare


def test_reordered_coordinates_are_not_strict_parity():
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    feats = torch.arange(14).reshape(2, 7).float()
    ref = SparseTensor(feats, coords)
    actual = SparseTensor(feats.flip(0), coords.flip(0))
    result = scene_compare(actual, ref, geometry=True)
    assert not result["coordinates_equal"] and result["features"] is None
    assert result["common_voxels"] == 2 and result["common_features"]["max_abs"] == 0


def test_missing_extra_and_empty_voxels():
    ref = SparseTensor(torch.zeros(2, 7), torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32))
    actual = SparseTensor(torch.ones(2, 7), torch.tensor([[0, 0, 0, 0], [0, 2, 0, 0]], dtype=torch.int32))
    result = scene_compare(actual, ref, geometry=True)
    assert not result["coordinates_equal"]
    assert result["missing_voxels"] == result["extra_voxels"] == result["common_voxels"] == 1
    assert result["common_rows_with_abs_error_over_1e_3"] == 1
    assert result["common_edge_sign_mismatches"] == 3
    empty = SparseTensor(torch.empty(0, 7), torch.empty(0, 4, dtype=torch.int32), torch.Size([1, 7]))
    result = scene_compare(empty, ref)
    assert result["common_voxels"] == 0 and result["missing_voxels"] == 2
