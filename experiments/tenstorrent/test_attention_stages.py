import os
os.environ['ATTN_BACKEND'] = 'sdpa'

import pytest
import torch

from experiments.tenstorrent.attention_stages import AttentionStages, mantissa_stats


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_stage_output_matches_fused_cpu_attention(dtype):
    torch.manual_seed(43)
    q, k, v = [torch.randn(1, 17, 4, 12).to(dtype) for _ in range(3)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2).to(dtype)
    final, *stages = AttentionStages()(q, k, v)
    assert final.dtype == dtype and all(x.dtype == torch.float32 for x in stages)
    tolerance = .002 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(final, expected, rtol=tolerance, atol=tolerance)


def test_mantissa_statistics_exclude_zero_and_nonfinite_values():
    one = torch.tensor(1.)
    next_value = torch.nextafter(one, torch.tensor(2.)).item()
    stats = mantissa_stats(torch.tensor([0., 1., next_value, float('inf'), float('nan')]))
    assert stats['nonzero_finite'] == 2 and not stats['finite']
    assert stats['low_13_bits_zero_fraction'] == .5
    assert stats['low_16_bits_zero_fraction'] == .5
    stats = mantissa_stats(torch.randn(31).bfloat16().float())
    assert stats['low_16_bits_zero_fraction'] == 1
    with pytest.raises(ValueError): mantissa_stats(torch.ones(3, dtype=torch.bfloat16))
