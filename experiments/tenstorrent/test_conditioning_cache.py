import os
os.environ.update(ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
import pytest
import torch
from experiments.tenstorrent.test_frozen_conditioning import fixture
from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache


@pytest.mark.parametrize('layout',['raw','heads','heads_t'])
def test_scene_noise_and_guidance_changes_preserve_original_outputs_and_lora(layout):
    flow,positive,negative=fixture()
    original_parameters=list(flow.parameters())
    values=[value.clone() for value in original_parameters]
    x,t=torch.randn(17,8),torch.tensor([.6])
    scenes=[(positive,negative),({key:value+.125 for key,value in positive.items()},negative)]
    with torch.no_grad():
        expected={(scene,noise,branch):flow(x+noise*.25,t,condition['context'],condition['projected'])
            for scene,conditions in enumerate(scenes) for noise in (0,1) for branch,condition in enumerate(conditions)}
        prefill,cached=prepare_conditioning_cache(flow,layout)
        for scene,conditions in enumerate(scenes):
            caches=[prefill(condition['context'],condition['projected']) for condition in conditions]
            for noise in (0,1):
                for branch in (0,1):
                    actual=cached(x+noise*.25,t,caches[branch])
                    torch.testing.assert_close(actual,expected[scene,noise,branch],rtol=0,atol=0)
    assert {id(value) for value in original_parameters}=={id(value) for value in cached.parameters()}
    assert {id(value) for value in prefill.parameters()}<={id(value) for value in cached.parameters()}
    for actual,original in zip(original_parameters,values,strict=True):
        torch.testing.assert_close(actual,original,rtol=0,atol=0)


@pytest.mark.parametrize('layout',['raw','heads','heads_t'])
def test_bad_cache_precision_or_shape_is_rejected(layout):
    flow,pos,neg=fixture();prefill,cached=prepare_conditioning_cache(flow,layout)
    with torch.no_grad():pc=prefill(pos['context'],pos['projected']);nc=prefill(neg['context'],neg['projected'])
    x,t=torch.randn(17,8),torch.tensor([.6])
    with pytest.raises(ValueError,match='every original block'):
        cached(x,t,pc[:-1])
    with pytest.raises(ValueError,match='FP32 storage'):
        cached(x,t,(pc[0].bfloat16(),*pc[1:]))
    with pytest.raises(ValueError,match='fixed token'):
        cached(x,t,(pc[0][:,:-1],*pc[1:]))
    if layout!='raw':
        assert len(pc)==3*len(flow.blocks)
        assert pc[1].shape==((1,4,12,9) if layout=='heads_t' else (1,4,9,12))
        with pytest.raises(ValueError,match='Head-packed'):
            cached(x,t,(pc[0],pc[1][:,:-1],*pc[2:]))
        with pytest.raises(ValueError,match='Head-packed'):
            cached(x,t,(pc[0],pc[1],pc[2][:,:,:-1],*pc[3:]))


def test_preparation_rejects_training_mixed_precision_and_repeated_wrapping():
    flow,pos,neg=fixture();flow.train()
    with pytest.raises(ValueError,match='eval FP32'):prepare_conditioning_cache(flow)
    flow.eval();flow.input_layer.weight.data=flow.input_layer.weight.data.bfloat16()
    with pytest.raises(ValueError,match='eval FP32'):prepare_conditioning_cache(flow)
    flow.input_layer.weight.data=flow.input_layer.weight.data.float()
    prepare_conditioning_cache(flow)
    with pytest.raises(ValueError,match='prepared once'):prepare_conditioning_cache(flow)


@pytest.mark.parametrize('layout',['heads','heads_t'])
def test_head_cache_without_key_normalization_and_changed_timestep(layout):
    flow,pos,neg=fixture()
    for block in flow.blocks:
        block.cross_attn.attention.qk_rms_norm=False
    x=torch.randn(17,8)
    with torch.no_grad():
        expected=[flow(x,torch.tensor([time]),pos['context'],pos['projected']) for time in (.6,.85)]
        prefill,cached=prepare_conditioning_cache(flow,layout)
        cache=prefill(pos['context'],pos['projected'])
        for time,reference in zip((.6,.85),expected):
            torch.testing.assert_close(cached(x,torch.tensor([time]),cache),reference,rtol=0,atol=0)


def test_invalid_head_plan_fails_before_mutating_any_layer():
    from torch import nn
    flow,pos,neg=fixture()
    first=flow.blocks[0].cross_attn
    with pytest.raises(ValueError,match='Unknown'):
        prepare_conditioning_cache(flow,'unknown')
    assert flow.blocks[0].cross_attn is first
    flow.blocks[1].cross_attn=nn.Identity()
    with pytest.raises(ValueError,match='prepared once'):
        prepare_conditioning_cache(flow,'heads')
    assert flow.blocks[0].cross_attn is first
