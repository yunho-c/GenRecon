import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'

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
def test_complete_block_with_nonzero_lora(dtype):
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
        assert prepare_real_rope(model) == 1
        actual = model.block(x, mod, context, model.phases)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-6, atol=2e-6)


def test_changed_layout_rejected():
    rope = RealPairRoPE(torch.ones(17, 8, dtype=torch.complex64))
    with pytest.raises(ValueError, match='layout'):
        rope(torch.randn(1, 18, 4, 16))
