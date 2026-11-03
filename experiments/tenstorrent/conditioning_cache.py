"""Separate fixed-condition prefill from the original FP32 denoiser arithmetic.

Cache tuples alternate projected-conditioning and flat raw-KV tensors per block.
They remain explicit graph inputs, allowing new same-shape scenes without
freezing their values into a compiled denoiser. Native validation is mandatory.
The original sampler's caller selects the positive or negative tuple; the
denoiser does not read or copy both branches on every model call.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.tenstorrent.inference_attention import ReciprocalAttention


class ConditioningPrefill(nn.Module):
    def __init__(self,flow):
        super().__init__()
        self.projections=nn.ModuleList(list(flow.proj_linears))
        self.key_values=nn.ModuleList([block.cross_attn.attention.to_kv for block in flow.blocks])
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
        for projection,key_value in zip(self.projections,self.key_values):
            result.extend((projection(projected.unsqueeze(0)),key_value(context)))
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


class CachedConditionFlow(nn.Module):
    def __init__(self,flow):
        super().__init__()
        self.flow=flow
        self.layers=len(flow.blocks)
        self.channels=flow.input_layer.out_features
        self.tokens=flow.phases.shape[0]
        for block in flow.blocks:
            block.cross_attn.attention.to_kv=CachedKVInput(block.cross_attn.attention.to_kv)
        self.eval()

    def forward(self,x,t,cache):
        if self.training:raise RuntimeError('Cached conditioning is inference-only')
        if len(cache)!=2*self.layers:
            raise ValueError('Expected every original block conditioning cache')
        context_tokens=None
        for index in range(self.layers):
            projected,key_value=cache[2*index:2*index+2]
            if projected.dtype!=torch.float32 or key_value.dtype!=torch.float32:
                raise ValueError('Conditioning caches must retain FP32 storage')
            if projected.shape!=(1,self.tokens,self.channels):
                raise ValueError('Projected cache disagrees with the fixed token layout')
            if key_value.ndim!=3 or key_value.shape[0]!=1 or key_value.shape[-1]!=2*self.channels:
                raise ValueError('Raw KV cache has incompatible dimensions')
            if context_tokens is None:context_tokens=key_value.shape[1]
            elif key_value.shape[1]!=context_tokens:raise ValueError('KV context lengths must agree across blocks')
        flow=self.flow
        h=flow.input_layer(x).unsqueeze(0).to(flow.dtype)
        mod=flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
        for index,block in enumerate(flow.blocks):
            projected,context=cache[2*index:2*index+2]
            h=h+projected
            h=block(h,mod,context,flow.phases)
        return flow.out_layer(F.layer_norm(h.to(x.dtype),h.shape[-1:]))[0]


def prepare_conditioning_cache(flow):
    """Validate the complete plan before changing any original module."""
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
        if context_channels is None:
            context_channels,projected_channels=key_value.in_features,projection.in_features
        elif key_value.in_features!=context_channels or projection.in_features!=projected_channels:
            raise ValueError('Conditioning feature widths must agree across blocks')
    prefill=ConditioningPrefill(flow)
    return prefill,CachedConditionFlow(flow)
