"""Observe prepared FP32 attention; native equivalence must be checked separately."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from experiments.tenstorrent.inference_attention import ReciprocalAttention

ATTENTION_STATE_NAMES = ('output','q_raw','k_raw','v_raw','q_norm','k_norm',
                         'q_head','k_head','v_head','scores','probabilities','attended')


def reciprocal_probabilities(scores):
    exponentials=torch.exp(scores-scores.amax(dim=-1,keepdim=True))
    return exponentials*torch.reciprocal(exponentials.sum(dim=-1,keepdim=True))


class AttentionObservation(nn.Module):
    def __init__(self,attention):
        super().__init__()
        if not isinstance(attention,ReciprocalAttention) or attention.mode!='fp32_reciprocal':
            raise ValueError('Observation requires prepared FP32 reciprocal attention')
        if attention.attention.use_rope and not attention.head_first:
            raise ValueError('Observation requires fixed head-first RoPE')
        self.prepared=attention

    def forward(self,x,context):
        module=self.prepared
        base=module.attention
        batch,tokens,channels=x.shape
        if base._type=='self':
            q,k,v=base.to_qkv(x).reshape(batch,tokens,3,base.num_heads,-1).unbind(2)
        else:
            q=base.to_q(x).reshape(batch,tokens,base.num_heads,-1)
            k,v=base.to_kv(context).reshape(batch,context.shape[1],2,base.num_heads,-1).unbind(2)
        q_norm,k_norm=(base.q_rms_norm(q),base.k_rms_norm(k)) if base.qk_rms_norm else (q,k)
        if module.head_first:
            q_head,k_head,v_head=[value.permute(0,2,1,3).contiguous() for value in (q_norm,k_norm,v)]
            q_head,k_head=module.fixed_rope(q_head),module.fixed_rope(k_head)
        else:
            q_head,k_head,v_head=[value.permute(0,2,1,3) for value in (q_norm,k_norm,v)]
        scores=(q_head @ k_head.transpose(-2,-1)).float()*(1/math.sqrt(q_head.shape[-1]))
        probabilities=reciprocal_probabilities(scores)
        attended=probabilities @ v_head
        output=base.to_out(attended.permute(0,2,1,3).reshape(batch,tokens,channels))
        return output,q,k,v,q_norm,k_norm,q_head,k_head,v_head,scores,probabilities,attended


def clone_projection_parameters(attention):
    base=attention.attention
    return ({name:value.detach().cpu().clone() for name,value in base.named_parameters()},
            {name:getattr(getattr(base,name),'scaling',None)
             for name in ('to_qkv','to_q','to_kv','to_out') if hasattr(base,name)})


def cpu_attention_references(attention,parameters,scales,x,context,states,cpu_rope=None):
    """Teacher force each boundary with captured inputs and cloned CPU weights."""
    base=attention.attention
    def linear(prefix,values):
        result=F.linear(values,parameters[prefix+'.weight'],parameters.get(prefix+'.bias'))
        if scales[prefix] is not None:
            delta=parameters[prefix+'.lora_B'] @ parameters[prefix+'.lora_A']
            result=result+F.linear(values,delta)*scales[prefix]
        return result
    batch,tokens,channels=x.shape
    if base._type=='self':
        q,k,v=linear('to_qkv',x).reshape(batch,tokens,3,base.num_heads,-1).unbind(2)
    else:
        q=linear('to_q',x).reshape(batch,tokens,base.num_heads,-1)
        k,v=linear('to_kv',context).reshape(batch,context.shape[1],2,base.num_heads,-1).unbind(2)
    expected={'q_raw':q,'k_raw':k,'v_raw':v}
    for prefix in ('q','k'):
        raw=states[prefix+'_raw']
        expected[prefix+'_norm']=(F.normalize(raw.float(),dim=-1)*parameters[prefix+'_rms_norm.gamma']*base.head_dim**.5
                                  if base.qk_rms_norm else raw)
        head=states[prefix+'_norm'].permute(0,2,1,3)
        expected[prefix+'_head']=cpu_rope(head.contiguous()) if attention.head_first else head
    expected['v_head']=states['v_raw'].permute(0,2,1,3)
    expected['scores']=(states['q_head'] @ states['k_head'].transpose(-2,-1)).float()*(1/math.sqrt(base.head_dim))
    expected['probabilities']=reciprocal_probabilities(states['scores'])
    expected['attended']=states['probabilities'] @ states['v_head']
    expected['output']=linear('to_out',states['attended'].permute(0,2,1,3).reshape(batch,tokens,channels))
    return expected


class QKVAttention(nn.Module):
    def __init__(self,outer_sum=False):
        super().__init__()
        self.outer_sum=outer_sum

    def forward(self,q,k,v):
        from experiments.tenstorrent.outer_product_sum import outer_product_sum
        scores=(outer_product_sum(q,k.transpose(-2,-1)) if self.outer_sum else q @ k.transpose(-2,-1))
        scores=scores*(1/math.sqrt(q.shape[-1]))
        return reciprocal_probabilities(scores) @ v
