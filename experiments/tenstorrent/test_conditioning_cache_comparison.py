import os
os.environ.update(ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
import torch
import pytest

from experiments.tenstorrent.test_frozen_conditioning import fixture
from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache
from experiments.tenstorrent.conditioning_cache_comparison import InlineConditionFlow


@pytest.mark.parametrize('layout',['raw','heads','heads_t'])
def test_inline_control_preserves_original_lora_values_and_recomputes_scene_conditions(layout):
    flow, positive, negative = fixture()
    x, t = torch.randn(17, 8), torch.tensor([.6])
    branches = [positive, negative, {key: value+.25 for key, value in positive.items()}]
    original = {id(value): value.clone() for value in flow.parameters()}
    with torch.no_grad():
        expected = [flow(x, t, branch['context'], branch['projected']) for branch in branches]
        prefill, body = prepare_conditioning_cache(flow,layout)
        inline = InlineConditionFlow(prefill, body)
        for branch, reference in zip(branches, expected, strict=True):
            torch.testing.assert_close(inline(x, t, branch['context'], branch['projected']), reference, rtol=0, atol=0)
    assert {id(value) for value in inline.parameters()} == original.keys()
    assert {id(value) for value in body.parameters()} == original.keys()
    for value in inline.parameters():
        torch.testing.assert_close(value, original[id(value)], rtol=0, atol=0)


@pytest.mark.parametrize('stage', ['shape', 'tex'])
def test_native_control_identity_keeps_prediction_width_and_rejects_changed_concat(tmp_path, stage):
    import json
    from experiments.tenstorrent.conditioning_cache_comparison import validate_native_control
    from experiments.tenstorrent.experiment_identity import sha256
    from experiments.tenstorrent.trajectory import TrajectoryFixture

    coords = torch.cat([torch.zeros(7, 1, dtype=torch.int32),
                        torch.arange(7, dtype=torch.int32)[:, None].expand(-1, 3)], 1)
    inputs = (torch.randn(7, 64 if stage == 'tex' else 32), torch.tensor([600.]),
              torch.randn(1, 9, 24), torch.randn(7, 24))
    sample = TrajectoryFixture(stage, inputs, coords)
    # This control's outputs are placeholders, but its semantic identity,
    # original schedule and 32-channel sampler prediction contract are real.
    expected, _ = sample.run(lambda x, t, cond: torch.zeros_like(x[:, :32]))
    path = tmp_path / 'trajectory-tt.pt'
    torch.save(expected, path)
    report = {'case': stage+'_flow_tensor', 'checkpoint': {'sha256': 'test-checkpoint'},
              'config': 'test-config', 'dtype': 'float32', 'versions': {'torch': torch.__version__},
              'seed': 0, 'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in inputs],
              'source_files_sha256': {'genrecon/example.py': 'same-source'}}
    source = dict(report, status='numerical_mismatch', phase='complete',
                  xla_executed_fallback_ops=[], xla_aten_counters={}, recompiled_during_trajectory=False,
                  attention_mode='fp32_reciprocal', device_precision='float32',
                  rope_mode='real_pairs_flat_head_first',
                  requested_compile_options={'math_fidelity': 'hifi4', 'optimization_level': 0},
                  trajectory_artifacts_sha256={path.name: sha256(path)})
    (tmp_path / 'result.json').write_text(json.dumps(source))
    actual, metadata = validate_native_control(tmp_path, report, sample)
    torch.testing.assert_close(actual['samples'], expected['samples'], rtol=0, atol=0)
    assert metadata['identity_verified']
    if stage == 'tex':
        changed = inputs[0].clone()
        changed[:, 32:] += .25
        other = TrajectoryFixture('tex', (changed, *inputs[1:]), coords)
        with pytest.raises(ValueError, match='conditions'):
            validate_native_control(tmp_path, report, other)
