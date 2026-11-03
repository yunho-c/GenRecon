"""Separate fixed-condition prefill from the original FP32 denoiser arithmetic.

Cache tuples contain projected conditioning and either flat raw KV (default)
or normalized, head-packed key and value tensors per block (opt-in heads mode).
The experimental heads_t mode stores the key already transposed for QK.
They remain explicit graph inputs, allowing new same-shape scenes without
freezing their values into a compiled denoiser. Native validation is mandatory.
The original sampler's caller selects the positive or negative tuple; the
denoiser does not read or copy both branches on every model call.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.tenstorrent.inference_attention import ReciprocalAttention, reciprocal_attention


class ConditioningPrefill(nn.Module):
    def __init__(self,flow,cache_layout='raw'):
        super().__init__()
        self.projections=nn.ModuleList(list(flow.proj_linears))
        self.key_values=nn.ModuleList([block.cross_attn.attention.to_kv for block in flow.blocks])
        self.cache_layout=cache_layout
        self.heads=tuple(block.cross_attn.attention.num_heads for block in flow.blocks)
        self.key_norms=nn.ModuleList([
            block.cross_attn.attention.k_rms_norm if block.cross_attn.attention.qk_rms_norm else nn.Identity()
            for block in flow.blocks] if cache_layout!='raw' else [])
        self.tokens=flow.phases.shape[0]
        self.context_channels=self.key_values[0].in_features
        self.projected_channels=self.projections[0].in_features
        self.eval()

    def forward(self,context,projected):
        if self.training:raise RuntimeError('Conditioning prefill is inference-only')
        if (context.dtype!=torch.float32 or projected.dtype!=torch.float32 or
            context.ndim!=3 or context.shape[0]!=1 or context.shape[-1]!=self.context_channels or
            projected.shape!=(self.tokens,self.projected_channels)):
            raise ValueError('Expected matching fixed-chunk FP32 conditioning inputs')
        result=[]
        for index,(projection,key_value) in enumerate(zip(self.projections,self.key_values)):
            projected_value=projection(projected.unsqueeze(0))
            raw=key_value(context)
            if self.cache_layout=='raw':
                result.extend((projected_value,raw))
            else:
                key,value=raw.reshape(1,context.shape[1],2,self.heads[index],-1).unbind(2)
                key=self.key_norms[index](key)
                key_order=(0,2,3,1) if self.cache_layout=='heads_t' else (0,2,1,3)
                result.extend((projected_value,key.permute(key_order),value.permute(0,2,1,3)))
        return tuple(result)


class CachedKVInput(nn.Module):
    def __init__(self,projection):
        super().__init__()
        # Retain original learned objects/LoRA; the prefill owns the same module.
        self.projection=projection
        self.channels=projection.out_features

    def forward(self,context):
        if context.dtype!=torch.float32 or context.ndim!=3 or context.shape[0]!=1 or context.shape[-1]!=self.channels:
            raise ValueError('Expected original flat raw-KV FP32 cache')
        return context


class CachedHeadCrossAttention(nn.Module):
    """Use original queries/output projection with fixed, prepared K/V inputs."""
    def __init__(self,prepared,key_transposed=False):
        super().__init__()
        self.attention=prepared.attention
        self.key_transposed=key_transposed
        self.eval()

    def forward(self,x,context,phases=None):
        if self.training:raise RuntimeError('Cached attention is inference-only')
        attention=self.attention
        batch,tokens,channels=x.shape
        key,value=context
        if self.key_transposed:
            # The attention helper transposes K for QK. These two views cancel,
            # retaining the canonical transposed-key matmul boundary natively.
            key=key.transpose(-2,-1)
        query=attention.to_q(x).reshape(batch,tokens,attention.num_heads,-1)
        if attention.qk_rms_norm:
            query=attention.q_rms_norm(query)
        query=query.permute(0,2,1,3)
        h=reciprocal_attention(query,key,value,'fp32_reciprocal',head_first=True)
        return attention.to_out(h.reshape(batch,tokens,channels))


class CachedConditionFlow(nn.Module):
    def __init__(self,flow,cache_layout='raw'):
        super().__init__()
        self.flow=flow
        self.layers=len(flow.blocks)
        self.channels=flow.input_layer.out_features
        self.tokens=flow.phases.shape[0]
        self.cache_layout=cache_layout
        self.cache_stride=2 if cache_layout=='raw' else 3
        self.heads=tuple(block.cross_attn.attention.num_heads for block in flow.blocks)
        for block in flow.blocks:
            if cache_layout=='raw':
                block.cross_attn.attention.to_kv=CachedKVInput(block.cross_attn.attention.to_kv)
            else:
                block.cross_attn=CachedHeadCrossAttention(block.cross_attn,cache_layout=='heads_t')
        self.eval()

    def forward(self,x,t,cache):
        if self.training:raise RuntimeError('Cached conditioning is inference-only')
        if len(cache)!=self.cache_stride*self.layers:
            raise ValueError('Expected every original block conditioning cache')
        context_tokens=None
        for index in range(self.layers):
            values=cache[self.cache_stride*index:self.cache_stride*(index+1)]
            projected,key_value=values[:2]
            if any(value.dtype!=torch.float32 for value in values):
                raise ValueError('Conditioning caches must retain FP32 storage')
            if projected.shape!=(1,self.tokens,self.channels):
                raise ValueError('Projected cache disagrees with the fixed token layout')
            if self.cache_layout=='raw':
                if key_value.ndim!=3 or key_value.shape[0]!=1 or key_value.shape[-1]!=2*self.channels:
                    raise ValueError('Raw KV cache has incompatible dimensions')
                length=key_value.shape[1]
            else:
                value=values[2]
                if key_value.ndim!=4 or key_value.shape[:2]!=(1,self.heads[index]):
                    raise ValueError('Head-packed KV cache has incompatible dimensions')
                transposed=self.cache_layout=='heads_t'
                length=key_value.shape[-1] if transposed else key_value.shape[-2]
                width=key_value.shape[-2] if transposed else key_value.shape[-1]
                if (width!=self.channels//self.heads[index] or
                    value.shape!=(1,self.heads[index],length,self.channels//self.heads[index])):
                    raise ValueError('Head-packed KV cache has incompatible dimensions')
            if context_tokens is None:context_tokens=length
            elif length!=context_tokens:raise ValueError('KV context lengths must agree across blocks')
        flow=self.flow
        h=flow.input_layer(x).unsqueeze(0).to(flow.dtype)
        mod=flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
        for index,block in enumerate(flow.blocks):
            values=cache[self.cache_stride*index:self.cache_stride*(index+1)]
            projected=values[0]
            context=values[1] if self.cache_layout=='raw' else (values[1],values[2])
            h=h+projected
            h=block(h,mod,context,flow.phases)
        return flow.out_layer(F.layer_norm(h.to(x.dtype),h.shape[-1:]))[0]


def prepare_conditioning_cache(flow,cache_layout='raw'):
    """Validate the complete plan before changing any original module."""
    if cache_layout not in ('raw','heads','heads_t'):
        raise ValueError('Unknown conditioning cache layout')
    if flow.training or flow.dtype!=torch.float32 or any(
        value.is_floating_point() and value.dtype!=torch.float32 for value in flow.parameters()):
        raise ValueError('Conditioning cache requires an eval FP32 flow')
    if not len(flow.blocks) or len(flow.blocks)!=len(flow.proj_linears):
        raise ValueError('Expected matching original blocks and projections')
    context_channels=None
    projected_channels=None
    for block,projection in zip(flow.blocks,flow.proj_linears):
        attention=block.cross_attn
        if (not getattr(block,'share_mod',False) or not isinstance(attention,ReciprocalAttention) or
            attention.mode!='fp32_reciprocal' or attention.attention._type!='cross' or
            attention.attention.use_rope or not isinstance(attention.attention.to_kv,nn.Linear) or
            not isinstance(projection,nn.Linear)):
            raise ValueError('Expected original FP32 cross-attention/projection modules, prepared once')
        key_value=attention.attention.to_kv
        if projection.out_features!=flow.input_layer.out_features or key_value.out_features!=2*flow.input_layer.out_features:
            raise ValueError('Conditioning output widths disagree with the original flow')
        if attention.attention.num_heads<1 or flow.input_layer.out_features%attention.attention.num_heads:
            raise ValueError('Attention heads must divide the original flow width')
        if context_channels is None:
            context_channels,projected_channels=key_value.in_features,projection.in_features
        elif key_value.in_features!=context_channels or projection.in_features!=projected_channels:
            raise ValueError('Conditioning feature widths must agree across blocks')
    prefill=ConditioningPrefill(flow,cache_layout)
    return prefill,CachedConditionFlow(flow,cache_layout)
