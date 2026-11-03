import os
os.environ.update(ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
import pytest
import torch
from torch import nn
from genrecon.modules.attention import MultiHeadAttention,RotaryPositionEmbedder
from genrecon.modules.lora import apply_lora_to_attention
from experiments.tenstorrent.inference_rope import prepare_real_rope,RealPairRoPE
from experiments.tenstorrent.inference_attention import prepare_attention
from experiments.tenstorrent.attention_observation import (AttentionObservation,ATTENTION_STATE_NAMES,
    clone_projection_parameters,cpu_attention_references,QKVAttention)


@pytest.mark.parametrize('kind',['self','cross'])
def test_observer_and_cpu_teacher_reconstruct_nonzero_lora_attention(kind):
    torch.set_num_threads(4)
    torch.manual_seed(66)
    model=nn.Module()
    attention=MultiHeadAttention(48,4,ctx_channels=24,type=kind,
                                use_rope=kind=='self',qk_rms_norm=True)
    apply_lora_to_attention(attention,4,8)
    for name,param in attention.named_parameters():
        if 'lora_B' in name:nn.init.normal_(param,std=.05)
    setattr(model,kind+'_attn',attention)
    phases=RotaryPositionEmbedder(12,3)(torch.randint(0,32,(17,3)))
    model.register_buffer('phases',phases)
    model.eval()
    if kind=='self':prepare_real_rope(model,head_first=True,flatten=True)
    prepare_attention(model,'fp32_reciprocal')
    prepared=getattr(model,kind+'_attn')
    observation=AttentionObservation(prepared).eval()
    parameters,scales=clone_projection_parameters(prepared)
    x,context=torch.randn(2,17,48),torch.randn(2,9,24)
    with torch.no_grad():
        states=dict(zip(ATTENTION_STATE_NAMES,observation(x,context),strict=True))
        torch.testing.assert_close(states['output'],prepared(x,context),rtol=0,atol=0)
        rope=RealPairRoPE(phases,head_first=True,flatten=True) if kind=='self' else None
        refs=cpu_attention_references(prepared,parameters,scales,x,context,states,rope)
        for name,value in refs.items():torch.testing.assert_close(value,states[name],rtol=0,atol=0)
        baseline=QKVAttention()(*[states[name] for name in ('q_head','k_head','v_head')])
        alternative=QKVAttention(True)(*[states[name] for name in ('q_head','k_head','v_head')])
        torch.testing.assert_close(baseline,states['attended'],rtol=0,atol=0)
        torch.testing.assert_close(alternative,baseline,rtol=2e-5,atol=2e-5)
