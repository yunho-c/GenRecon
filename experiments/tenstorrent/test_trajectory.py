import os
os.environ['SPARSE_CONV_BACKEND'] = 'none'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

import pytest
import torch
from genrecon.modules.sparse import SparseTensor
from genrecon.pipelines.samplers import FlowEulerGuidanceIntervalSampler
from experiments.tenstorrent.scene_sampling import features
from experiments.tenstorrent.trajectory import TrajectoryFixture, compare_trajectories


@pytest.mark.parametrize('stage,count', [('ss',22), ('shape',21), ('tex',18)])
def test_artifact_matches_native_sampler_and_preserves_texture(stage, count):
    torch.manual_seed(27)
    coords = torch.cat([torch.zeros(7,1,dtype=torch.int32), torch.arange(7,dtype=torch.int32)[:,None].expand(-1,3)], 1)
    values = torch.randn(1,3,4) if stage == 'ss' else torch.randn(7,64 if stage == 'tex' else 32)
    inputs = (values, torch.tensor([600.]), torch.randn(1,9,24), torch.randn(7,24))
    fixture = TrajectoryFixture(stage, inputs, None if stage=='ss' else coords)
    snapshots = [x.clone() for x in inputs]
    def tensor_forward(x,t,cond):
        assert t.shape == (1,)  # Active tokens must not become sampler batches.
        base = x[:,:32] if stage=='tex' else x
        result = .1*base.sin() + .01*cond['context'].mean() + .02*t[0]/1000
        if stage=='tex':
            torch.testing.assert_close(x[:,32:], inputs[0][:,32:], rtol=0, atol=0)
            result = result + .03*x[:,32:].cos()
        return result
    def native(x,t,cond,concat_cond=None):
        result = .1*features(x).sin() + .01*cond['context'].mean() + .02*t[0]/1000
        if concat_cond is not None:
            result = result + .03*concat_cond.feats.cos()
        return result if stage=='ss' else x.replace(result)
    noise = values if stage=='ss' else SparseTensor(values[:,:32],coords)
    concat = SparseTensor(values[:,32:],coords) if stage=='tex' else None
    expected = FlowEulerGuidanceIntervalSampler(1e-5).sample(native,noise,
        cond=fixture.positive,neg_cond=fixture.negative,concat_cond=concat,verbose=False,**fixture.params)
    actual, measurements = fixture.run(tensor_forward)
    torch.testing.assert_close(actual['samples'],features(expected.samples),rtol=0,atol=0)
    for key in ('pred_x_t','pred_x_0'):
        for a,e in zip(actual[key],expected[key],strict=True):
            torch.testing.assert_close(a,features(e),rtol=0,atol=0)
    for a,e in zip(inputs,snapshots,strict=True):
        torch.testing.assert_close(a,e,rtol=0,atol=0)
    assert len(measurements['forwards']) == count
    assert compare_trajectories(actual,actual)['all_states_pass_smoke']
    corrupted = dict(actual,pred_x_t=list(actual['pred_x_t']))
    corrupted['pred_x_t'][3] = corrupted['pred_x_t'][3]+2
    assert not compare_trajectories(corrupted,actual)['all_states_pass_smoke']
    wrong_condition = dict(actual,fixture_identity=dict(actual['fixture_identity'],concat_sha256='changed'))
    with pytest.raises(ValueError,match='conditions'):
        compare_trajectories(wrong_condition,actual)


def test_legacy_artifacts_require_explicit_diagnostic_mode():
    value = torch.arange(8.)
    old = {'coords':None, 'noise':value, 'samples':value,
           'pred_x_t':[value]*12, 'pred_x_0':[value]*12}
    with pytest.raises(ValueError,match='conditions'):
        compare_trajectories(old,old)
    result = compare_trajectories(old,old,allow_legacy=True)
    assert result['all_states_pass_smoke']
    assert not result['fixture_identity_verified']
