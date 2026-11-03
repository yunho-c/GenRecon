"""Opt-in attention precision boundaries with stable reciprocal softmax.

Prepare disposable eval models after strict loading and stock CPU references.
Preserve original projections, LoRA, Q/K normalization, RoPE and output dtype.
Only SDPA intermediates change; complete trajectory/quality validation remains
mandatory. Dense and fixed active-token sequence boundaries only.
"""
import math

import torch
from torch import nn

ATTENTION_MODES = ('fp32_reciprocal', 'fp32_scores_bf16_av', 'bf16_qk_fp32_softmax')


def reciprocal_attention(q, k, v, mode='fp32_reciprocal'):
    """BTHD attention with original output dtype and explicit inner precision."""
    if mode not in ATTENTION_MODES:
        raise ValueError('Unknown prepared attention mode')
    score_dtype = q.dtype if mode == 'bf16_qk_fp32_softmax' else torch.float32
    a, b = [value.to(score_dtype).permute(0, 2, 1, 3) for value in (q, k)]
    # Promote before scaling even when the QK product retains the outer dtype.
    scores = (a @ b.transpose(-2, -1)).float() * (1 / math.sqrt(q.shape[-1]))
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exponentials = torch.exp(shifted)
    # Division would fuse this back into the native approximate softmax.
    probabilities = exponentials * torch.reciprocal(exponentials.sum(dim=-1, keepdim=True))
    av_dtype = torch.float32 if mode == 'fp32_reciprocal' else q.dtype
    c = v.to(av_dtype).permute(0, 2, 1, 3)
    return (probabilities.to(av_dtype) @ c).permute(0, 2, 1, 3).to(q.dtype)


class ReciprocalAttention(nn.Module):
    def __init__(self, attention, mode='fp32_reciprocal'):
        super().__init__()
        if mode not in ATTENTION_MODES:
            raise ValueError('Unknown prepared attention mode')
        self.mode = mode
        from experiments.tenstorrent.inference_rope import RealRoPESelfAttention
        if isinstance(attention, RealRoPESelfAttention):
            self.attention = attention.attention
            self.fixed_rope = attention.rope
        else:
            self.attention = attention
            self.fixed_rope = None
        self.eval()

    def forward(self, x, context=None, phases=None):
        from genrecon.modules.attention import RotaryPositionEmbedder
        if self.training:
            raise RuntimeError('Prepared attention is inference-only')
        attn = self.attention
        batch, tokens, channels = x.shape
        if attn._type == 'self':
            q, k, v = attn.to_qkv(x).reshape(batch, tokens, 3, attn.num_heads, -1).unbind(2)
        else:
            q = attn.to_q(x).reshape(batch, tokens, attn.num_heads, -1)
            k, v = attn.to_kv(context).reshape(batch, context.shape[1], 2, attn.num_heads, -1).unbind(2)
        if attn.qk_rms_norm:
            q, k = attn.q_rms_norm(q), attn.k_rms_norm(k)
        if attn.use_rope:
            if self.fixed_rope is not None:
                q, k = self.fixed_rope(q), self.fixed_rope(k)
            else:
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
        h = reciprocal_attention(q, k, v, self.mode)
        return attn.to_out(h.reshape(batch, tokens, channels))


def prepare_attention(model, mode):
    from genrecon.modules.attention import MultiHeadAttention
    from genrecon.modules.sparse.attention import SparseMultiHeadAttention
    from experiments.tenstorrent.inference_rope import RealRoPESelfAttention
    if mode not in ATTENTION_MODES:
        raise ValueError('Unknown prepared attention mode')
    if model.training:
        raise ValueError('Attention preparation requires model.eval()')
    planned = []
    for owner in list(model.modules()):
        for name in ('self_attn', 'cross_attn'):
            attention = getattr(owner, name, None)
            if attention is None:
                continue
            if isinstance(attention, SparseMultiHeadAttention):
                raise ValueError('Prepared attention supports dense/fixed-sequence boundaries only')
            if not isinstance(attention, (MultiHeadAttention, RealRoPESelfAttention)):
                raise ValueError(f'Unsupported attention module: {type(attention).__module__}.{type(attention).__name__}')
            original = attention.attention if isinstance(attention, RealRoPESelfAttention) else attention
            if original.attn_mode != 'full':
                raise ValueError('Prepared attention requires full attention')
            planned.append((owner, name, attention))
    if not planned:
        raise ValueError('No supported attention modules found')
    for owner, name, attention in planned:
        setattr(owner, name, ReciprocalAttention(attention, mode))
    return len(planned)
