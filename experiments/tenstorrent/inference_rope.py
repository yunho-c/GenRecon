"""Fixed-layout real RoPE candidate that avoids a tiled [..., pairs, 2] axis.

Preserves adjacent complex pairs: y = x*cos + swap_adjacent(x)*signed_sin.
Prepare after checkpoint loading on a disposable eval model with fixed phases.
The stock implementation and production default remain unchanged.
"""
import torch
from torch import nn


def compensated_gather(x, indices):
    """Near-FP32 lookup via three supported BF16 payloads.

    Intended for normal GenRecon activation ranges. BF16 multiplication keeps
    narrowing observable: the pinned compiler removes direct down/up casts.
    The small CPU approximation and native error require independent checks.
    """
    residual = x
    pieces = []
    for index in range(3):
        digit = residual.bfloat16() * 3
        pieces.append(torch.index_select(digit, -1, indices).float() * (1/3))
        if index < 2:
            residual = residual - digit.float() * (1/3)
    return (pieces[0] + pieces[1]) + pieces[2]


class RealPairRoPE(nn.Module):
    def __init__(self, phases, head_first=False, flatten=False, fp32_split=False):
        super().__init__()
        if fp32_split and not (head_first and flatten):
            raise ValueError('FP32 split lookup requires flattened head-first layout')
        if phases.device.type != 'cpu' or not phases.is_complex() or phases.ndim != 2:
            raise ValueError('Expected fixed CPU complex phases [tokens, head_dim/2]')
        tokens, pairs = phases.shape
        self.tokens, self.head_dim = tokens, pairs*2
        self.head_first = head_first
        self.flatten = flatten
        self.fp32_split = fp32_split
        ids = torch.arange(self.head_dim)
        signs = torch.where(ids % 2 == 0, -1., 1.)
        self.register_buffer('partner_ids', ids ^ 1)
        cosine = phases.real.repeat_interleave(2, -1)[None, :, None, :]
        sine = (phases.imag.repeat_interleave(2, -1)*signs)[None, :, None, :]
        if head_first:
            cosine, sine = cosine.transpose(1, 2), sine.transpose(1, 2)
        self.register_buffer('cosine', cosine.contiguous())
        self.register_buffer('signed_sine', sine.contiguous())

    def forward(self, x):
        token_axis = 2 if self.head_first else 1
        if x.ndim != 4 or x.shape[token_axis] != self.tokens or x.shape[-1] != self.head_dim:
            raise ValueError('RoPE candidate requires the prepared token/head layout')
        if self.flatten:
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            cosine = self.cosine.expand(shape).reshape(-1, shape[-1])
            sine = self.signed_sine.expand(shape).reshape(-1, shape[-1])
            partner = (compensated_gather(flat, self.partner_ids) if self.fp32_split and x.dtype == torch.float32
                       else torch.index_select(flat, -1, self.partner_ids).float())
            return (flat.float()*cosine + partner*sine).to(x.dtype).reshape(shape)
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
        if self.rope.head_first:
            from torch.nn import functional as F
            q, k, v = [value.permute(0, 2, 1, 3).contiguous() for value in (q, k, v)]
            h = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v).permute(0, 2, 1, 3)
        else:
            h = scaled_dot_product_attention(self.rope(q), self.rope(k), v)
        return attn.to_out(h.reshape(batch, tokens, channels))


class FixedSparseRealRoPE(nn.Module):
    """RoPE-only replacement inside a fixed-coordinate native sparse probe.

    The owning probe constructs every SparseTensor from its registered coords;
    this wrapper does not implement dynamic or ragged coordinate preparation.
    All sparse attention, QKV packing, normalization and LoRA stay original.
    """
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.eval()

    def forward(self, q, k=None):
        if self.training:
            raise RuntimeError('Prepared real RoPE is inference-only')
        if q.shape[0] != 1 or (k is not None and k.shape[0] != 1):
            raise ValueError('Fixed sparse RoPE requires one prepared chunk')
        q_embed = q.replace(self.rope(q.feats.unsqueeze(0))[0])
        if k is None:
            return q_embed
        return q_embed, k.replace(self.rope(k.feats.unsqueeze(0))[0])


def prepare_real_rope(model, head_first=False, flatten=False, fp32_split=False):
    from genrecon.modules.attention import MultiHeadAttention
    from genrecon.modules.sparse.attention import SparseMultiHeadAttention
    if model.training:
        raise ValueError('Real RoPE preparation requires model.eval()')
    if fp32_split and not (head_first and flatten):
        raise ValueError('FP32 split lookup requires flattened head-first layout')
    if head_first:
        from genrecon.modules.attention import config
        if config.BACKEND != 'sdpa':
            raise ValueError('Head-first RoPE requires the sdpa backend')
        if any(isinstance(module, SparseMultiHeadAttention) for module in model.modules()):
            raise ValueError('Head-first RoPE supports dense/fixed-sequence attention only')
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
        fixed_sparse = getattr(owner, 'precompute_rope', False)
        if fixed_sparse:
            # NativeSparseFlowProbe alone owns phases and its fixed coords.
            # Never infer fixed coordinates from an arbitrary sparse model.
            if not hasattr(owner, 'coords') or not hasattr(owner, 'model'):
                raise ValueError('Fixed sparse preparation requires a coordinate-owning probe')
            blocks = owner.model.blocks
        shared_rope = None
        for block in blocks:
            attn = getattr(block, 'self_attn', None)
            supported = isinstance(attn, MultiHeadAttention) or (
                fixed_sparse and isinstance(attn, SparseMultiHeadAttention))
            if not supported or not attn.use_rope:
                continue
            if attn._type != 'self' or attn.attn_mode != 'full':
                raise ValueError('Only full fixed-layout self-attention is supported')
            if shared_rope is None:
                shared_rope = RealPairRoPE(phases, head_first=head_first, flatten=flatten, fp32_split=fp32_split)
            if attn.head_dim != shared_rope.head_dim:
                raise ValueError('Attention head dimension disagrees with prepared phases')
            if fixed_sparse:
                attn.rope = FixedSparseRealRoPE(shared_rope)
            else:
                block.self_attn = RealRoPESelfAttention(attn, shared_rope)
            count += 1
    if not count:
        raise ValueError('No supported fixed-layout RoPE attention found')
    return count
