import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'

import pytest
import torch
from torch import nn

from genrecon.modules.attention import MultiHeadAttention, RotaryPositionEmbedder
from genrecon.modules.lora import apply_lora_to_attention
from genrecon.modules.utils import convert_module_to
from experiments.tenstorrent.inference_attention import ATTENTION_MODES, prepare_attention, reciprocal_attention
from experiments.tenstorrent.inference_rope import RealPairRoPE, RealRoPESelfAttention
from experiments.tenstorrent.precision_reference import PipelinePrecisionMode


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('kind,real_rope', [('self', False), ('self', True), ('self', 'head_first'), ('cross', False)])
@pytest.mark.parametrize('mode', ['fp32_reciprocal', 'fp32_scores_bf16_av'])
def test_preparation_preserves_weights_lora_and_outer_dtype(dtype, kind, real_rope, mode):
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
    original = RealRoPESelfAttention(attention, RealPairRoPE(phases, head_first=real_rope == 'head_first')) if real_rope else attention
    model = nn.Module()
    model.self_attn = original
    model.eval()
    identities = {id(p) for p in model.parameters()}
    values = [p.clone() for p in model.parameters()]
    x = torch.randn(2, 17, 48).to(dtype)
    context = torch.randn(2, 9, 24).to(dtype) if kind == 'cross' else None
    with torch.no_grad(), PipelinePrecisionMode():
        expected = model.self_attn(x, context=context, phases=phases)
        if mode == 'fp32_scores_bf16_av' and dtype == torch.bfloat16:
            # This candidate intentionally rounds probabilities before AV.
            # Check wrapper reconstruction against original learned QKV/LoRA
            # and projection; stock SDPA is no longer the same arithmetic.
            from experiments.tenstorrent.attention_diagnostic import extract_qkv
            qkv = extract_qkv(attention, x, context=context, phases=phases)
            expected = attention.to_out(reciprocal_attention(*qkv, mode).reshape_as(x))
        assert prepare_attention(model, mode) == 1
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


@pytest.mark.parametrize('mode', ATTENTION_MODES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('head_first', [False, True])
def test_matrix_precision_boundaries_and_uniform_attention(mode, dtype, head_first):
    from torch.utils._python_dispatch import TorchDispatchMode

    class ObserveProducts(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.dtypes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func._schema.name == 'aten::bmm':
                self.dtypes.append(args[0].dtype)
            return func(*args, **(kwargs or {}))

    q, k = torch.zeros(1, 17, 4, 12, dtype=dtype), torch.zeros(1, 32, 4, 12, dtype=dtype)
    v = torch.arange(32, dtype=dtype)[None, :, None, None].expand(1, 32, 4, 12)
    observer = ObserveProducts()
    values = (q, k, v)
    if head_first:
        values = tuple(x.permute(0, 2, 1, 3) for x in values)
    with torch.no_grad(), PipelinePrecisionMode(), observer:
        actual = reciprocal_attention(*values, mode, head_first=head_first)
    expected = torch.full_like(q, 15.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first = dtype if mode == 'bf16_qk_fp32_softmax' else torch.float32
    second = torch.float32 if mode == 'fp32_reciprocal' else dtype
    assert observer.dtypes == [first, second]


def test_fp32_preparation_modes_share_the_original_formula():
    from experiments.tenstorrent.attention_precision import PrecisionAttention
    torch.manual_seed(55)
    values = (torch.randn(2, 17, 4, 12), torch.randn(2, 9, 4, 12), torch.randn(2, 9, 4, 12))
    expected = PrecisionAttention('legacy')(*values)
    for mode in ATTENTION_MODES:
        torch.testing.assert_close(reciprocal_attention(*values, mode), expected, rtol=0, atol=0)
