"""Coordinate-intersection diagnostics must not hide topology differences."""

import copy
import pytest
import torch
from genrecon.modules.sparse import SparseTensor
from experiments.tenstorrent.sparse_decoder_scene_probe import scene_compare, scene_passed


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


@pytest.mark.parametrize("failure", [
    "fixed_subdivision", "predicted_subdivision", "fixed_edge", "predicted_edge",
    "missing_parent", "missing_stage", "wide", "chain", "continuous", "coordinates",
])
def test_strict_gate_keeps_discrete_failures_even_with_small_feature_errors(failure):
    comparison = {"coordinates_equal": True,
                  "features": {"finite": True, "relative_l2": 1e-7, "pcc": 1.0}}
    report = {
        "gate": {"relative_l2": 1e-4, "pcc": 0.999999},
        **{k: copy.deepcopy(comparison) for k in
           ("shape_predicted", "texture_fixed", "texture_predicted")},
        "chain_coordinates_equal": True,
        "subdivisions_predicted": [{"common_sign_mismatches": 0, "missing_parents": 0,
                                    "extra_parents": 0} for _ in range(4)],
        "prior_fixed_shape": {"shape_fixed": copy.deepcopy(comparison),
            "captured_wide": {"passed": True},
            "subdivisions_fixed": [{"sign_mismatches": 0} for _ in range(4)],
            "geometry_fixed": {"edge_sign_mismatches": 0}},
    }
    report["shape_predicted"]["common_edge_sign_mismatches"] = 0
    assert scene_passed(report)
    fixed = report["prior_fixed_shape"]
    if failure == "fixed_subdivision":
        fixed["subdivisions_fixed"][2]["sign_mismatches"] = 1
    elif failure == "predicted_subdivision":
        report["subdivisions_predicted"][2]["common_sign_mismatches"] = 1
    elif failure == "fixed_edge":
        fixed["geometry_fixed"]["edge_sign_mismatches"] = 1
    elif failure == "predicted_edge":
        report["shape_predicted"]["common_edge_sign_mismatches"] = 1
    elif failure == "missing_parent":
        report["subdivisions_predicted"][3]["missing_parents"] = 1
    elif failure == "missing_stage":
        report["subdivisions_predicted"].pop()
    elif failure == "wide":
        fixed["captured_wide"]["passed"] = False
    elif failure == "chain":
        report["chain_coordinates_equal"] = False
    elif failure == "continuous":
        report["texture_fixed"]["features"]["relative_l2"] = 1e-2
    elif failure == "coordinates":
        report["texture_predicted"]["coordinates_equal"] = False
    assert not scene_passed(report)
