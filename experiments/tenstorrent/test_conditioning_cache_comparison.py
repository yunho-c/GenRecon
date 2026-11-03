import os
os.environ.update(ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
import torch

from experiments.tenstorrent.test_frozen_conditioning import fixture
from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache
from experiments.tenstorrent.conditioning_cache_comparison import InlineConditionFlow


def test_inline_control_preserves_original_lora_values_and_recomputes_scene_conditions():
    flow, positive, negative = fixture()
    x, t = torch.randn(17, 8), torch.tensor([.6])
    branches = [positive, negative, {key: value+.25 for key, value in positive.items()}]
    original = {id(value): value.clone() for value in flow.parameters()}
    with torch.no_grad():
        expected = [flow(x, t, branch['context'], branch['projected']) for branch in branches]
        prefill, body = prepare_conditioning_cache(flow)
        inline = InlineConditionFlow(prefill, body)
        for branch, reference in zip(branches, expected, strict=True):
            torch.testing.assert_close(inline(x, t, branch['context'], branch['projected']), reference, rtol=0, atol=0)
    assert {id(value) for value in inline.parameters()} == original.keys()
    assert {id(value) for value in body.parameters()} == original.keys()
    for value in inline.parameters():
        torch.testing.assert_close(value, original[id(value)], rtol=0, atol=0)
