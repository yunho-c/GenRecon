"""Opt-in fixed-scene FP32 conditioning for native consteval experiments.

Readonly buffers make both conditioning branches eligible for consteval.
Actual hoisting must be checked in native IR; CPU equivalence alone is not a
cache, performance or reconstruction-quality result. Rebuild for new conditions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.tenstorrent.inference_attention import ReciprocalAttention


class FixedConditions(nn.Module):
    def __init__(self,positive,negative):
        super().__init__()
        if set(positive)!={'context','projected'} or set(negative)!=set(positive):
            raise ValueError('Expected explicit context/projected conditioning branches')
        for key,rank in (('context',3),('projected',2)):
            a,b=positive[key],negative[key]
            if (a.device.type!='cpu' or b.device.type!='cpu' or a.dtype!=torch.float32 or
                b.dtype!=torch.float32 or a.ndim!=rank or a.shape!=b.shape):
                raise ValueError('Fixed conditioning requires matching CPU FP32 branch tensors')
            if key=='context' and a.shape[0]!=1:
                raise ValueError('Only a single fixed chunk is supported')
            self.register_buffer('positive_'+key,a.detach().clone())
            self.register_buffer('negative_'+key,b.detach().clone())


class FrozenKVProjection(nn.Module):
    def __init__(self,projection,conditions):
        super().__init__()
        self.projection=projection
        self.conditions=conditions

    def forward(self,context):
        if self.training:raise RuntimeError('Frozen conditioning is inference-only')
        if context.dtype!=torch.bool or context.shape!=(1,self.conditions.positive_context.shape[1],1):
            raise ValueError('Expected a broadcast boolean conditioning selector')
        # Select the original flat projection output. The existing attention
        # keeps its original KV reshape, normalization and head layout.
        return torch.where(context,self.projection(self.conditions.positive_context),
                           self.projection(self.conditions.negative_context))


class FrozenProjection(nn.Module):
    def __init__(self,projection,conditions):
        super().__init__()
        self.projection=projection
        self.conditions=conditions

    def forward(self,positive):
        return torch.where(positive,
            self.projection(self.conditions.positive_projected.unsqueeze(0)),
            self.projection(self.conditions.negative_projected.unsqueeze(0)))


class FrozenConditionFlow(nn.Module):
    def __init__(self,flow,positive,negative):
        super().__init__()
        if flow.training or flow.dtype!=torch.float32:
            raise ValueError('Frozen conditioning requires an eval FP32 flow')
        conditions=FixedConditions(positive,negative)
        if not len(flow.blocks) or len(flow.blocks)!=len(flow.proj_linears):
            raise ValueError('Expected matching nonempty block/projection lists')
        if conditions.positive_projected.shape[0]!=flow.phases.shape[0]:
            raise ValueError('Conditioning tokens disagree with the fixed phase layout')
        planned=[]
        for block,projection in zip(flow.blocks,flow.proj_linears):
            attention=block.cross_attn
            if isinstance(projection,FrozenProjection) or isinstance(getattr(getattr(attention,'attention',None),'to_kv',None),FrozenKVProjection):
                raise ValueError('Frozen conditioning is already prepared')
            if (not isinstance(attention,ReciprocalAttention) or attention.mode!='fp32_reciprocal' or
                attention.attention._type!='cross' or attention.attention.use_rope):
                raise ValueError('Frozen conditioning requires ordinary FP32 reciprocal cross attention')
            if (attention.attention.ctx_channels!=positive['context'].shape[-1] or
                projection.in_features!=positive['projected'].shape[-1]):
                raise ValueError('Conditioning feature dimensions disagree with learned modules')
            planned.append((block,projection,attention.attention.to_kv))
        self.flow=flow
        self.conditions=conditions
        for index,(block,projection,key_value) in enumerate(planned):
            block.cross_attn.attention.to_kv=FrozenKVProjection(key_value,conditions)
            flow.proj_linears[index]=FrozenProjection(projection,conditions)
        self.eval()

    def forward(self,x,t,positive):
        if self.training:raise RuntimeError('Frozen conditioning is inference-only')
        if positive.dtype!=torch.bool or positive.shape!=(1,):
            raise ValueError('Expected one explicit boolean branch selector')
        flow=self.flow
        h=flow.input_layer(x).unsqueeze(0).to(flow.dtype)
        mod=flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
        context=positive.reshape(1,1,1).expand(1,self.conditions.positive_context.shape[1],1)
        for block,projection in zip(flow.blocks,flow.proj_linears):
            h=h+projection(positive)
            h=block(h,mod,context,flow.phases)
        return flow.out_layer(F.layer_norm(h.to(x.dtype),h.shape[-1:]))[0]
