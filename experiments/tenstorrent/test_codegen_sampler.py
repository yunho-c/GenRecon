import os
os.environ['SPARSE_CONV_BACKEND'] = 'none'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

import copy
import json
import math
from pathlib import Path
import pytest
import torch
from experiments.tenstorrent.codegen_sampler import (match_activation_inputs, assert_bitwise_trajectory,
                                                     restore_native_time_frequencies, sampler_fixture)
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


def test_fixed_frequency_restoration_is_bound_to_exact_native_control_and_values():
    evidence = Path(__file__).parent/'results/autoresearch/093-rejected-generated-sampler-boundary'
    control = json.loads((evidence/'sdk-execute.json').read_text())['sampler_native_control']['result_sha256']
    original = torch.exp(-math.log(10000)*torch.arange(128, dtype=torch.float32)/128)
    native, repair = restore_native_time_frequencies(original, control)
    assert repair['changed_indices'] == [12, 106]
    assert torch.nonzero(native != original).flatten().tolist() == [12, 106]
    assert torch.all((native.view(torch.int32)-original.view(torch.int32)).abs() <= 1)
    with pytest.raises(ValueError, match='control110'):
        restore_native_time_frequencies(original, 'different-control')
    corrupted = original.clone(); corrupted[0] += .001
    with pytest.raises(ValueError, match='two recorded'):
        restore_native_time_frequencies(corrupted, control)


def test_new_control_frequency_proof_requires_all_native_values_and_rejects_ambiguous_ir(tmp_path):
    import gzip
    import hashlib
    evidence = Path(__file__).parent/'results/autoresearch'
    path = evidence/'108-texture512-fresh-native-control/runtime-ttnn.mlir.gz'
    control = hashlib.sha256((evidence/'110-texture512-matching-source-controls/ordinary-result.json').read_bytes()).hexdigest()
    original = torch.exp(-math.log(10000)*torch.arange(128, dtype=torch.float32)/128)
    native, repair = restore_native_time_frequencies(original, control, native_ir=path)
    assert repair['changed_indices'] == [12, 106]
    assert repair['bound_native_control_result_sha256'] == control
    assert repair['native_ir_uncompressed_sha256'] == hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
    assert repair['source'] == str(path.resolve())
    wrong = native.clone()
    wrong[40] = torch.nextafter(wrong[40], torch.tensor(float('inf')))
    for label, data in [('changed', wrong), ('duplicate', native), ('missing', native)]:
        literal = 'dense<"0x'+data.numpy().tobytes().hex()+'"> : tensor<1x128xf32>'
        text = literal if label == 'changed' else literal*2 if label == 'duplicate' else ''
        target = tmp_path/(label+'.mlir')
        target.write_text(text)
        with pytest.raises(ValueError, match='exactly the sealed128-value'):
            restore_native_time_frequencies(original, control, native_ir=target)


@pytest.mark.parametrize('case,count', [('shape_flow_tensor', 21), ('tex_flow_tensor', 18)])
def test_generated_sampler_fixture_keeps_original_output_width_and_concat_identity(case, count):
    stage = case.split('_')[0]
    coords = torch.zeros(5, 4, dtype=torch.int32)
    inputs = (torch.randn(5, 64 if stage == 'tex' else 32), torch.tensor([600.]),
              torch.randn(1, 9, 24), torch.randn(5, 24))
    fixture = TrajectoryFixture(stage, inputs, coords)
    payload, _ = fixture.run(lambda x, t, cond: .1*x[:, :32])
    validated = sampler_fixture(case, inputs, payload, payload)
    _, measurements = validated.run(lambda x, t, cond: .1*x[:, :32])
    assert len(measurements['forwards']) == count
    if stage == 'tex':
        changed = inputs[0].clone()
        changed[:, 32:] += .25
        with pytest.raises(ValueError, match='conditions'):
            sampler_fixture(case, (changed, *inputs[1:]), payload, payload)
