"""Complete cached texture schedules retain their fixed shape conditioning."""
import os
os.environ.update(ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')

import pytest
import torch
from torch import nn

from experiments.tenstorrent.cached_flow_sampler import assert_bitwise_trajectory
from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache
from experiments.tenstorrent.conditioning_cache_comparison import InlineConditionFlow
from experiments.tenstorrent.test_frozen_conditioning import fixture
from experiments.tenstorrent.trajectory import TrajectoryFixture


@pytest.mark.parametrize('layout', ['raw', 'heads', 'heads_t'])
def test_complete_texture_cache_and_inline_keep_shape_concat_and_original_schedule(layout):
    flow, positive, _ = fixture()
    flow.input_layer = nn.Linear(64, flow.input_layer.out_features)
    flow.out_layer = nn.Linear(flow.input_layer.out_features, 32)
    original_ids = {id(p) for p in flow.parameters()}
    coords = torch.cat([torch.zeros(17, 1, dtype=torch.int32),
                        torch.arange(17, dtype=torch.int32)[:, None].expand(-1, 3)], 1)
    inputs = (torch.randn(17, 64), torch.tensor([600.]),
              positive['context'], positive['projected'])
    fixtures = []
    for shift in (0., .25):
        values = inputs[0].clone()
        values[:, 32:] += shift
        fixtures.append(TrajectoryFixture('tex', (values, *inputs[1:]), coords))

    def run(sample, forward):
        seen = []

        def checked(x, t, cond):
            torch.testing.assert_close(x[:, 32:], sample.concat.feats, rtol=0, atol=0)
            assert x.shape == (17, 64)
            seen.append(t.item())
            return forward(x, t, cond)

        payload, measurements = sample.run(checked)
        assert len(seen) == len(measurements['forwards']) == 18
        assert len(measurements['steps']) == 12
        assert measurements['parameters'] == sample.params
        return payload

    with torch.no_grad():
        expected = [run(sample, lambda x, t, cond: flow(x, t, cond['context'], cond['projected']))
                    for sample in fixtures]
        prefill, body = prepare_conditioning_cache(flow, layout)
        inline = InlineConditionFlow(prefill, body)
        assert {id(p) for p in inline.parameters()} == original_ids
        for sample, reference in zip(fixtures, expected, strict=True):
            caches = {id(cond): prefill(cond['context'], cond['projected'])
                      for cond in (sample.positive, sample.negative)}
            cached = run(sample, lambda x, t, cond: body(x, t, caches[id(cond)]))
            assert_bitwise_trajectory(cached, reference)
            recomputed = run(sample, lambda x, t, cond: inline(x, t, cond['context'], cond['projected']))
            assert_bitwise_trajectory(recomputed, reference)
    assert not torch.equal(expected[0]['samples'], expected[1]['samples'])
    torch.testing.assert_close(fixtures[0].noise.feats, fixtures[1].noise.feats, rtol=0, atol=0)
