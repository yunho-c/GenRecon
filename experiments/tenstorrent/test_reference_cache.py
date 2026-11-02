import copy
import json
import os
os.environ['SPARSE_CONV_BACKEND'] = 'none'

import pytest
import torch

from experiments.tenstorrent.experiment_identity import sha256
from experiments.tenstorrent.reference_cache import create_cache, load_cache
from experiments.tenstorrent.trajectory import TrajectoryFixture, compare_trajectories


def setup_cache(tmp_path):
    torch.manual_seed(37)
    inputs = (torch.randn(1, 3, 4), torch.tensor([1000.]), torch.randn(1, 9, 8), torch.randn(1, 8, 8))
    fixture = TrajectoryFixture('ss', inputs)
    payload, measurements = fixture.run(lambda x, t, cond: .1 * x + .01 * cond['context'].mean())
    source = tmp_path / 'source'; source.mkdir()
    report = {'phase': 'waiting_for_device', 'case': 'ss_flow', 'strict_loading': True,
              'reference_precision': 'float32', 'checkpoint': {'sha256': 'matched'},
              'versions': {'torch': torch.__version__}, 'seed': 37,
              'source_files_sha256': {'genrecon/model.py': 'original',
                                      'experiments/tenstorrent/flow_probes.py': 'original',
                                      'experiments/tenstorrent/inference_rope.py': 'candidate-v1'},
              'cpu_trajectory': measurements}
    (source / 'result.json').write_text(json.dumps(report))
    torch.save(payload, source / 'trajectory-reference.pt')
    cache = tmp_path / 'cache'
    create_cache(source, cache)
    return cache, report, fixture, payload


def test_cache_reuses_exact_stock_reference_across_preparation_changes(tmp_path):
    cache, report, fixture, expected = setup_cache(tmp_path)
    report['source_files_sha256']['experiments/tenstorrent/inference_rope.py'] = 'candidate-v2'
    report['flow_inputs_sha256'] = 'different-serialization-of-identical-tensors'
    actual, measurements, info = load_cache(cache, report, fixture)
    assert compare_trajectories(actual, expected)['all_states_pass_smoke']
    assert torch.equal(actual['samples'], expected['samples'])
    assert measurements == report['cpu_trajectory']
    assert info['reference_sha256'] == sha256(cache / 'reference.pt')


@pytest.mark.parametrize('change', ['checkpoint', 'reference_source', 'torch', 'precision'])
def test_cache_rejects_changed_reference_recipe(tmp_path, change):
    cache, report, fixture, _ = setup_cache(tmp_path)
    if change == 'checkpoint': report['checkpoint']['sha256'] = 'different'
    elif change == 'reference_source': report['source_files_sha256']['genrecon/model.py'] = 'changed'
    elif change == 'torch': report['versions']['torch'] = 'changed'
    else: report['reference_precision'] = 'native'
    with pytest.raises(ValueError): load_cache(cache, report, fixture)


@pytest.mark.parametrize('change', ['conditions', 'noise', 'coordinates'])
def test_cache_rejects_changed_fixture(tmp_path, change):
    cache, report, fixture, _ = setup_cache(tmp_path)
    if change == 'conditions': fixture.positive['context'] = fixture.positive['context'] + 1
    elif change == 'noise': fixture.noise = fixture.noise + 1
    else: fixture.coords = torch.zeros(3, 4, dtype=torch.int32)
    with pytest.raises(ValueError): load_cache(cache, report, fixture)


def test_cache_rejects_modified_file_and_nonfinite_states(tmp_path):
    cache, report, fixture, payload = setup_cache(tmp_path)
    with (cache / 'reference.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'): load_cache(cache, report, fixture)
    payload = copy.deepcopy(payload)
    payload['pred_x_t'][3].flatten()[0] = float('nan')
    torch.save(payload, cache / 'reference.pt')
    manifest = json.loads((cache / 'manifest.json').read_text())
    manifest['reference_sha256'] = sha256(cache / 'reference.pt')
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='finite'): load_cache(cache, report, fixture)
