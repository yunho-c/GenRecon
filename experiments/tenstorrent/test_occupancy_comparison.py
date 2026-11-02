import pytest
import torch

from experiments.tenstorrent.compare_occupancy import overlap


def test_overlap_counts_and_reference_direction():
    actual = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    reference = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    result = overlap(actual, reference)
    assert result['intersection'] == 1 and result['union'] == 4
    assert result['iou'] == .25 and result['dice'] == .4
    assert result['false_positive'] == 2 and result['false_negative'] == 1


def test_identical_and_empty_grids():
    for value in (torch.tensor([True, False]), torch.zeros(3, dtype=torch.bool)):
        result = overlap(value, value)
        assert result['iou'] == 1 and result['dice'] == 1


def test_mismatched_or_unthresholded_grids_rejected():
    with pytest.raises(ValueError): overlap(torch.zeros(3, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError): overlap(torch.zeros(3), torch.zeros(3))
