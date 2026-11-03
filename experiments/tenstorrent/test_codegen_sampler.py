import os
os.environ['SPARSE_CONV_BACKEND'] = 'none'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

import copy
import pytest
import torch
from experiments.tenstorrent.codegen_sampler import match_activation_inputs, assert_bitwise_trajectory
from experiments.tenstorrent.trajectory import TrajectoryFixture, compare_trajectories


def test_generated_inputs_match_by_values_and_reject_missing_or_ambiguous_inputs():
    inputs = (torch.randn(5, 32), torch.tensor([600.]), torch.randn(1, 9, 24), torch.randn(5, 24))
    templates = [torch.linspace(.01, 1., 128), inputs[1].clone(), inputs[2].clone(),
                 inputs[3].clone(), inputs[0].clone()]
    assert match_activation_inputs(templates, inputs) == [None, 1, 2, 3, 0]
    wrong = list(templates)
    wrong[2] = wrong[2] + .001
    with pytest.raises(ValueError, match='four exact inputs'):
        match_activation_inputs(wrong, inputs)
    with pytest.raises(ValueError, match='Ambiguous'):
        match_activation_inputs([torch.zeros(128), torch.zeros(1)], [torch.zeros(1)]*4)


def test_native_bitwise_gate_rejects_drift_below_stock_smoke_threshold():
    coords = torch.zeros(5, 4, dtype=torch.int32)
    inputs = (torch.randn(5, 32), torch.tensor([600.]), torch.randn(1, 9, 24), torch.randn(5, 24))
    fixture = TrajectoryFixture('shape', inputs, coords)
    payload, _ = fixture.run(lambda x, t, cond: .1*x)
    assert_bitwise_trajectory(payload, payload)
    changed = copy.deepcopy(payload)
    changed['pred_x_t'][4] += 1e-5
    assert compare_trajectories(changed, payload)['all_states_pass_smoke']
    with pytest.raises(ValueError, match='pred_x_t'):
        assert_bitwise_trajectory(changed, payload)
