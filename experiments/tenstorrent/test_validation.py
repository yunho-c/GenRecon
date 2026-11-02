import pytest
import torch
from experiments.tenstorrent.validation import compare_tensors, enforce_check


@pytest.mark.parametrize("actual,expected", [
    (torch.tensor([float('nan')]), torch.ones(1)),
    (torch.ones(1), torch.tensor([float('inf')])),
    (torch.ones(2, 2), torch.ones(4)),
    (torch.ones(8)*2, torch.ones(8)),
    (torch.tensor([0, 1]), torch.tensor([0, 2])),
    (torch.tensor([1., 2., 3.]), torch.tensor([3., 2., 1.])),
])
def test_bad_candidate_cannot_complete(actual, expected):
    report = {}
    with pytest.raises(ValueError, match="smoke gate"):
        enforce_check(report, "first_forward", actual, expected)
    assert report['numerical_checks']['first_forward']['passed'] is False


@pytest.mark.parametrize("value", [torch.ones(4), torch.zeros(4), torch.arange(8.), torch.arange(8)])
def test_exact_outputs_pass(value):
    assert compare_tensors(value, value)['passed']


def test_thresholds_are_enforced_and_recorded():
    x = torch.arange(10.)
    a = compare_tensors(x * 1.02, x, max_relative_l2=.01)
    assert not a['passed'] and a['pcc'] > .9999
    assert a['max_relative_l2'] == .01
