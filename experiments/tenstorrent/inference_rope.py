"""Fixed-layout real RoPE candidate that avoids a tiled [..., pairs, 2] axis.

Preserves adjacent complex pairs: y = x*cos + swap_adjacent(x)*signed_sin.
Prepare after checkpoint loading on a disposable eval model with fixed phases.
The stock implementation and production default remain unchanged.
"""
import torch
from torch import nn


class RealPairRoPE(nn.Module):
    def __init__(self, phases):
        super().__init__()
        if phases.device.type != 'cpu' or not phases.is_complex() or phases.ndim != 2:
            raise ValueError('Expected fixed CPU complex phases [tokens, head_dim/2]')
        tokens, pairs = phases.shape
        self.tokens, self.head_dim = tokens, pairs*2
        ids = torch.arange(self.head_dim)
        signs = torch.where(ids % 2 == 0, -1., 1.)
        self.register_buffer('partner_ids', ids ^ 1)
        self.register_buffer('cosine', phases.real.repeat_interleave(2, -1)[None, :, None, :].contiguous())
        self.register_buffer('signed_sine', (phases.imag.repeat_interleave(2, -1)*signs)[None, :, None, :].contiguous())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.tokens or x.shape[-1] != self.head_dim:
            raise ValueError('RoPE candidate requires the prepared token/head layout')
        # Gather before promotion: BF16 input pairs stay exactly representable.
        partner = torch.index_select(x, -1, self.partner_ids).float()
        return (x.float()*self.cosine + partner*self.signed_sine).to(x.dtype)


class RealRoPESelfAttention(nn.Module):
    def __init__(self, attention, rope):
        super().__init__()
        self.attention = attention
        self.rope = rope
        self.eval()

    def forward(self, x, context=None, phases=None):
        if self.training:
            raise RuntimeError('Prepared real RoPE is inference-only')
        from genrecon.modules.attention.full_attn import scaled_dot_product_attention
        attn = self.attention
        batch, tokens, channels = x.shape
        q, k, v = attn.to_qkv(x).reshape(batch, tokens, 3, attn.num_heads, -1).unbind(2)
        if attn.qk_rms_norm:
            q, k = attn.q_rms_norm(q), attn.k_rms_norm(k)
        h = scaled_dot_product_attention(self.rope(q), self.rope(k), v)
        return attn.to_out(h.reshape(batch, tokens, channels))


def prepare_real_rope(model):
    from genrecon.modules.attention import MultiHeadAttention
    if model.training:
        raise ValueError('Real RoPE preparation requires model.eval()')
    count = 0
    for owner in list(model.modules()):
        phases = getattr(owner, 'phases', None)
        if phases is None:
            phases = getattr(owner, 'rope_phases', None)
        if phases is None:
            continue
        blocks = getattr(owner, 'blocks', None)
        if blocks is None:
            block = getattr(owner, 'block', None)
            blocks = [block] if block is not None else []
        shared_rope = None
        for block in blocks:
            attn = getattr(block, 'self_attn', None)
            if not isinstance(attn, MultiHeadAttention) or not attn.use_rope:
                continue
            if attn._type != 'self' or attn.attn_mode != 'full':
                raise ValueError('Only full dense active-token self-attention is supported')
            if shared_rope is None:
                shared_rope = RealPairRoPE(phases)
            if attn.head_dim != shared_rope.head_dim:
                raise ValueError('Attention head dimension disagrees with prepared phases')
            block.self_attn = RealRoPESelfAttention(attn, shared_rope)
            count += 1
    if not count:
        raise ValueError('No supported fixed-layout dense RoPE attention found')
    return count
