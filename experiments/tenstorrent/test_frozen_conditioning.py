import os
os.environ.update(ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
import pytest
import torch
from torch import nn
from experiments.tenstorrent.test_flow_observation import TinyFlow
from experiments.tenstorrent.inference_rope import prepare_real_rope
from experiments.tenstorrent.inference_attention import prepare_attention
from experiments.tenstorrent.frozen_conditioning import FrozenConditionFlow


def fixture():
    torch.set_num_threads(4)
    torch.manual_seed(71)
    flow=TinyFlow().eval()
    prepare_real_rope(flow,head_first=True,flatten=True)
    prepare_attention(flow,'fp32_reciprocal')
    positive={'context':torch.randn(1,9,24),'projected':torch.randn(17,24)}
    negative={key:torch.zeros_like(value) for key,value in positive.items()}
    return flow,positive,negative


def test_both_conditioning_branches_preserve_outputs_lora_and_parameter_ids():
    flow,pos,neg=fixture()
    identities={id(value) for value in flow.parameters()}
    weights=[value.clone() for value in flow.parameters()]
    x,t=torch.randn(17,8),torch.tensor([.6])
    with torch.no_grad():
        expected=[flow(x,t,branch['context'],branch['projected']) for branch in (pos,neg)]
        frozen=FrozenConditionFlow(flow,pos,neg)
        for flag,reference in zip((True,False),expected,strict=True):
            actual=frozen(x,t,torch.tensor([flag]))
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    assert identities=={id(value) for value in frozen.parameters()}
    for value,original in zip(frozen.parameters(),weights,strict=True):
        torch.testing.assert_close(value,original,rtol=0,atol=0)


def test_buffers_are_owned_and_invalid_policy_fails_before_mutation():
    flow,pos,neg=fixture()
    original=flow.blocks[0].cross_attn
    neg['context']=neg['context'].bfloat16()
    with pytest.raises(ValueError,match='matching CPU FP32'):
        FrozenConditionFlow(flow,pos,neg)
    assert flow.blocks[0].cross_attn is original
    neg['context']=neg['context'].float()
    frozen=FrozenConditionFlow(flow,pos,neg)
    original_context=frozen.conditions.positive_context.clone()
    pos['context'].zero_()
    torch.testing.assert_close(frozen.conditions.positive_context,original_context,rtol=0,atol=0)


def test_branch_selector_and_training_are_rejected():
    flow,pos,neg=fixture();frozen=FrozenConditionFlow(flow,pos,neg)
    x,t=torch.randn(17,8),torch.tensor([.6])
    with pytest.raises(ValueError,match='boolean'):
        frozen(x,t,torch.tensor([1.]))
    frozen.train()
    with pytest.raises(RuntimeError,match='inference-only'):
        frozen(x,t,torch.tensor([True]))


def test_fixed_token_layout_and_repeated_preparation_are_rejected():
    flow,pos,neg=fixture()
    shortened={key:(value[:-1] if key=='projected' else value) for key,value in pos.items()}
    shortened_neg={key:torch.zeros_like(value) for key,value in shortened.items()}
    with pytest.raises(ValueError,match='tokens'):
        FrozenConditionFlow(flow,shortened,shortened_neg)
    FrozenConditionFlow(flow,pos,neg)
    with pytest.raises(ValueError,match='already prepared'):
        FrozenConditionFlow(flow,pos,neg)


def test_incompatible_later_block_is_rejected_before_any_layer_is_replaced():
    flow,pos,neg=fixture()
    first=flow.blocks[0].cross_attn.attention.to_kv
    projection=flow.proj_linears[0]
    flow.blocks[1].cross_attn=nn.Identity()
    with pytest.raises(ValueError,match='ordinary FP32 reciprocal'):
        FrozenConditionFlow(flow,pos,neg)
    assert flow.blocks[0].cross_attn.attention.to_kv is first
    assert flow.proj_linears[0] is projection
