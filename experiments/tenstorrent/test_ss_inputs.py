import pytest
import torch

from experiments.tenstorrent.prepare_ss_inputs import load_ss_inputs


def fixture(tmp_path):
    values = (torch.randn(1, 2, 2, 2, 2), torch.tensor([1000.]),
              torch.randn(1, 9, 8), torch.randn(1, 8, 8))
    payload = {'schema': 1, 'stage': 'ss', 'checkpoint_sha256': 'matched',
               'inputs': values, 'provenance': {'image_features': 'source-hash'}}
    path = tmp_path / 'inputs.pt'
    torch.save(payload, path)
    return path, payload


def test_saved_inputs_preserve_real_conditions_and_identity(tmp_path):
    path, payload = fixture(tmp_path)
    values, metadata = load_ss_inputs(path, payload['inputs'], 'matched')
    for actual, expected in zip(values, payload['inputs'], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata['flow_inputs_provenance'] == payload['provenance']
    assert len(metadata['flow_inputs_sha256']) == 64
    with pytest.raises(ValueError, match='checkpoint'):
        load_ss_inputs(path, payload['inputs'], 'different')


@pytest.mark.parametrize('corruption', ['shape', 'dtype', 'nan', 'stage'])
def test_bad_occupancy_fixture_rejected(tmp_path, corruption):
    path, payload = fixture(tmp_path)
    expected = tuple(t.clone() for t in payload['inputs'])
    if corruption == 'stage':
        payload['stage'] = 'shape'
    else:
        values = list(payload['inputs'])
        if corruption == 'shape':
            values[2] = values[2][:, :3]
        elif corruption == 'dtype':
            values[2] = values[2].bfloat16()
        else:
            values[2][0, 0, 0] = float('nan')
        payload['inputs'] = values
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_ss_inputs(path, expected, 'matched')
