import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'

import pytest
import torch
from torch import nn

from genrecon.modules.attention import MultiHeadAttention, RotaryPositionEmbedder
from genrecon.modules.lora import apply_lora_to_attention
from genrecon.modules.utils import convert_module_to
from experiments.tenstorrent.inference_attention import prepare_attention
from experiments.tenstorrent.inference_rope import RealPairRoPE, RealRoPESelfAttention
from experiments.tenstorrent.precision_reference import PipelinePrecisionMode


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind,real_rope', [('self', False), ('self', True), ('cross', False)])
def test_preparation_preserves_weights_lora_and_outer_dtype(dtype, kind, real_rope):
    torch.set_num_threads(4)
    torch.manual_seed(53)
    attention = MultiHeadAttention(48, 4, ctx_channels=24, type=kind,
                                   use_rope=kind == 'self', qk_rms_norm=True)
    apply_lora_to_attention(attention, 4, 8)
    for name, parameter in attention.named_parameters():
        if 'lora_B' in name:
            nn.init.normal_(parameter, std=.05)
    attention.apply(lambda module: convert_module_to(module, dtype))
    phases = RotaryPositionEmbedder(12, 3)(torch.randint(0, 32, (17, 3)))
    original = RealRoPESelfAttention(attention, RealPairRoPE(phases)) if real_rope else attention
    model = nn.Module()
    model.self_attn = original
    model.eval()
    identities = {id(p) for p in model.parameters()}
    values = [p.clone() for p in model.parameters()]
    x = torch.randn(2, 17, 48).to(dtype)
    context = torch.randn(2, 9, 24).to(dtype) if kind == 'cross' else None
    with torch.no_grad(), PipelinePrecisionMode():
        expected = model.self_attn(x, context=context, phases=phases)
        assert prepare_attention(model, 'fp32_reciprocal') == 1
        actual = model.self_attn(x, context=context, phases=phases)
    assert actual.dtype == dtype
    assert identities == {id(p) for p in model.parameters()}
    for a, b in zip(model.parameters(), values, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    tolerance = .002 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    model.train()
    with pytest.raises(RuntimeError, match='inference-only'):
        model.self_attn(x, context=context, phases=phases)


def test_native_sparse_and_training_preparation_rejected_without_mutation():
    from genrecon.modules.sparse.attention import SparseMultiHeadAttention
    model = nn.Module()
    attention = SparseMultiHeadAttention(48, 4)
    model.self_attn = attention
    with pytest.raises(ValueError, match='eval'):
        prepare_attention(model, 'fp32_reciprocal')
    model.eval()
    with pytest.raises(ValueError, match='dense/fixed-sequence'):
        prepare_attention(model, 'fp32_reciprocal')
    assert model.self_attn is attention


def test_combined_block_prepares_both_attention_paths_and_rejects_unknown_wrappers():
    from genrecon.modules.transformer import ModulatedTransformerCrossBlock
    from experiments.tenstorrent.inference_rope import prepare_real_rope
    torch.manual_seed(54)
    model = nn.Module()
    model.block = ModulatedTransformerCrossBlock(48, 24, 4, use_rope=True, share_mod=True,
        qk_rms_norm=True, qk_rms_norm_cross=True)
    model.register_buffer('phases', RotaryPositionEmbedder(12, 3)(torch.randint(0, 32, (17, 3))))
    model.eval()
    x, mod, context = torch.randn(2, 17, 48), torch.randn(2, 288), torch.randn(2, 9, 24)
    with torch.no_grad():
        expected = model.block(x, mod, context, model.phases)
        assert prepare_real_rope(model) == 1
        assert prepare_attention(model, 'fp32_reciprocal') == 2
        actual = model.block(x, mod, context, model.phases)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    other = nn.Module()
    other.self_attn = nn.Identity()
    other.cross_attn = MultiHeadAttention(48, 4, ctx_channels=24, type='cross')
    other.eval()
    original_cross = other.cross_attn
    with pytest.raises(ValueError, match='Unsupported attention module'):
        prepare_attention(other, 'fp32_reciprocal')
    assert other.cross_attn is original_cross
