import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

import pytest
import torch
from torch import nn
from genrecon.modules.attention import RotaryPositionEmbedder
from genrecon.modules.transformer import ModulatedTransformerCrossBlock
from genrecon.modules.lora import apply_lora_to_attention
from genrecon.modules.utils import convert_module_to
from experiments.tenstorrent.inference_rope import RealPairRoPE, prepare_real_rope


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('tokens,head_dim', [(17, 12), (129, 128)])
def test_adjacent_complex_pair_equivalence(dtype, tokens, head_dim):
    torch.manual_seed(21)
    phases = RotaryPositionEmbedder(head_dim, 3)(torch.randint(0, 64, (tokens, 3)))
    x = torch.randn(2, tokens, 4, head_dim).to(dtype)
    expected = RotaryPositionEmbedder.apply_rotary_embedding(x, phases)
    actual = RealPairRoPE(phases)(x)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('head_first,flatten,fp32_split', [
    (False, False, False), (False, True, False), (True, False, False),
    (True, True, False), (True, True, True)])
def test_complete_block_with_nonzero_lora(dtype, head_first, flatten, fp32_split):
    torch.set_num_threads(4)
    torch.manual_seed(21)
    model = nn.Module()
    model.block = ModulatedTransformerCrossBlock(48, 24, 4, use_rope=True,
        share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)
    for attn in (model.block.self_attn, model.block.cross_attn):
        apply_lora_to_attention(attn, 4, 8)
        for name, param in attn.named_parameters():
            if 'lora_B' in name:
                nn.init.normal_(param, std=.05)
    model.apply(lambda module: convert_module_to(module, dtype))
    model.register_buffer('phases', RotaryPositionEmbedder(12, 3)(torch.randint(0, 32, (17, 3))))
    model.eval()
    x, mod, context = torch.randn(2, 17, 48).to(dtype), torch.randn(2, 288).to(dtype), torch.randn(2, 9, 24).to(dtype)
    with torch.no_grad():
        expected = model.block(x, mod, context, model.phases)
        assert prepare_real_rope(model, head_first=head_first, flatten=flatten, fp32_split=fp32_split) == 1
        actual = model.block(x, mod, context, model.phases)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-6, atol=2e-6)


def test_changed_layout_rejected():
    rope = RealPairRoPE(torch.ones(17, 8, dtype=torch.complex64))
    with pytest.raises(ValueError, match='layout'):
        rope(torch.randn(1, 18, 4, 16))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('tokens', [17, 129])
def test_compensated_lookup_preserves_rotation(dtype, tokens):
    torch.manual_seed(57)
    phases = RotaryPositionEmbedder(128, 3)(torch.randint(0, 64, (tokens, 3)))
    x = torch.randn(2, tokens, 12, 128).to(dtype)
    expected = RealPairRoPE(phases)(x)
    actual = RealPairRoPE(phases, head_first=True, flatten=True, fp32_split=True)(
        x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('head_first,flatten', [(False, False), (False, True), (True, False)])
def test_compensated_lookup_requires_flat_head_first_before_mutation(head_first, flatten):
    model = nn.Module()
    model.eval()
    with pytest.raises(ValueError, match='flattened head-first'):
        prepare_real_rope(model, head_first=head_first, flatten=flatten, fp32_split=True)
    assert not list(model.modules())[1:]
    with pytest.raises(ValueError, match='flattened head-first'):
        RealPairRoPE(torch.ones(17, 8, dtype=torch.complex64),
                     head_first=head_first, flatten=flatten, fp32_split=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('tokens', [17, 129])
@pytest.mark.parametrize('flatten', [False, True])
def test_head_first_rope_bitwise_to_token_first(dtype, tokens, flatten):
    torch.manual_seed(56)
    phases = RotaryPositionEmbedder(128, 3)(torch.randint(0, 64, (tokens, 3)))
    x = torch.randn(2, tokens, 12, 128).to(dtype)
    expected = RealPairRoPE(phases)(x)
    actual = RealPairRoPE(phases, head_first=True, flatten=flatten)(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_head_first_native_sparse_rejected_before_mutation():
    from genrecon.modules.sparse.attention import SparseMultiHeadAttention
    model = nn.Module()
    model.self_attn = SparseMultiHeadAttention(48, 4)
    original = model.self_attn
    model.eval()
    with pytest.raises(ValueError, match='dense/fixed-sequence'):
        prepare_real_rope(model, head_first=True)
    assert model.self_attn is original


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_sparse_block_preserves_features_and_layout(dtype):
    from genrecon.modules.sparse import SparseTensor
    from genrecon.modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
    from experiments.tenstorrent.flow_probes import NativeSparseFlowProbe
    torch.set_num_threads(4)
    torch.manual_seed(21)
    inner = nn.Module()
    inner.blocks = nn.ModuleList([ModulatedSparseTransformerCrossBlock(48, 24, 4,
        use_rope=True, share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True)])
    block = inner.blocks[0]
    for attn in (block.self_attn, block.cross_attn):
        apply_lora_to_attention(attn, 4, 8)
        for name, param in attn.named_parameters():
            if 'lora_B' in name:
                nn.init.normal_(param, std=.05)
    inner.apply(lambda module: convert_module_to(module, dtype))
    coords = torch.cat([torch.zeros(17, 1, dtype=torch.int32),
                        torch.randint(0, 32, (17, 3), dtype=torch.int32)], 1)
    probe = NativeSparseFlowProbe(inner, coords, precompute_rope=True).eval()
    x = probe.sparse(torch.randn(17, 48).to(dtype))
    mod, context = torch.randn(1, 288).to(dtype), torch.randn(1, 9, 24).to(dtype)
    original_attention, original_qkv = block.self_attn, block.self_attn.to_qkv
    with torch.no_grad():
        expected = block(x, mod, context)
        assert prepare_real_rope(probe) == 1
        assert block.self_attn is original_attention
        assert block.self_attn.to_qkv is original_qkv
        actual = block(x, mod, context)
    assert isinstance(actual, SparseTensor)
    assert actual.coords is x.coords
    assert actual.layout == x.layout
    torch.testing.assert_close(actual.feats.float(), expected.feats.float(), rtol=2e-6, atol=2e-6)
