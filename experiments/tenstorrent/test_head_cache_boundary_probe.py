import os
os.environ.update(ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
import torch
import pytest
from experiments.tenstorrent.test_frozen_conditioning import fixture
from experiments.tenstorrent.head_cache_boundary_probe import stages


@pytest.mark.parametrize('transposed',[False,True])
def test_diagnostic_paths_preserve_original_attention_and_share_all_learned_objects(transposed):
    flow,positive,negative=fixture()
    original=flow.blocks[0].cross_attn
    parameters={id(value) for value in original.parameters()}
    pack,raw,head,fused=stages(original,transposed)
    for module in (pack,raw,head,fused):
        assert {id(value) for value in module.parameters()}==parameters
    query=torch.randn(1,17,48)
    with torch.no_grad():
        for condition in (positive,negative):
            context=condition['context']
            kv=original.attention.to_kv(context)
            expected=original(query,context)
            for actual in (raw(query,kv),head(query,pack(kv)),fused(query,kv)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
