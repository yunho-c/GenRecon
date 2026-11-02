import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'

import pytest
import torch

from experiments.tenstorrent.attention_diagnostic import AttentionMath, extract_qkv
from experiments.tenstorrent.precision_reference import PipelinePrecisionMode
from genrecon.modules.attention import MultiHeadAttention, RotaryPositionEmbedder
from genrecon.modules.lora import apply_lora_to_attention
from genrecon.modules.utils import convert_module_to


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind', ['self', 'cross'])
def test_extracted_attention_reconstructs_original_with_nonzero_lora(dtype, kind):
    torch.set_num_threads(4)
    torch.manual_seed(31)
    attn = MultiHeadAttention(48, 4, ctx_channels=24, type=kind,
                             use_rope=kind == 'self', qk_rms_norm=True).eval()
    apply_lora_to_attention(attn, 4, 8)
    for name, param in attn.named_parameters():
        if 'lora_B' in name:
            torch.nn.init.normal_(param, std=.05)
    attn.apply(lambda module: convert_module_to(module, dtype))
    x = torch.randn(2, 17, 48).to(dtype)
    context = torch.randn(2, 9, 24).to(dtype) if kind == 'cross' else None
    phases = RotaryPositionEmbedder(12, 3)(torch.randint(0, 32, (17, 3))) if kind == 'self' else None
    with torch.no_grad(), PipelinePrecisionMode():
        expected = attn(x, context=context, phases=phases)
        qkv = extract_qkv(attn, x, context=context, phases=phases)
        actual = attn.to_out(AttentionMath()(*qkv).reshape_as(x))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp32_sdpa_matches_cpu_precision_diagnostic_and_restores_dtype():
    torch.manual_seed(37)
    q, k, v = [torch.randn(1, 17, 4, 12).bfloat16() for _ in range(3)]
    with PipelinePrecisionMode(): expected = AttentionMath()(q, k, v)
    actual = AttentionMath(True)(q, k, v)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
