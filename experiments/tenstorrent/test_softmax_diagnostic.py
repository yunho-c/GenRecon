import pytest
import torch

from experiments.tenstorrent.softmax_diagnostic import SoftmaxMath


@pytest.mark.parametrize('width', [32, 37])
@pytest.mark.parametrize('mode', ['renormalized', 'explicit_exp', 'explicit_exp2', 'explicit_reciprocal', 'refined_reciprocal'])
def test_cpu_formulations_match_stable_softmax_at_large_offsets(mode, width):
    torch.manual_seed(47)
    value = 1000 + torch.randn(2, 3, width) * 20
    expected = torch.softmax(value, dim=-1)
    actual = SoftmaxMath(mode)(value)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, 3), rtol=2e-6, atol=2e-6)
