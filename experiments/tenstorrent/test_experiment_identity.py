import pytest
from experiments.tenstorrent.experiment_identity import experiment_identity, validate_resume


@pytest.mark.parametrize('change', ['source', 'config', 'weights', 'settings', 'new_weights'])
def test_resume_rejects_changed_experiment(tmp_path, change):
    root = tmp_path/'repo'
    for folder in ('genrecon', 'configs', 'experiments/tenstorrent'):
        (root/folder).mkdir(parents=True)
    source, config = root/'genrecon/a.py', root/'configs/a.json'
    source.write_text('a=1')
    config.write_text('{}')
    checkpoints = tmp_path/'weights'
    checkpoints.mkdir()
    weight = checkpoints/'shape.pt'
    weight.write_bytes(b'original')
    settings = {'layout': 'sequence', 'threads': 16}
    before = experiment_identity(root, settings, checkpoints)
    validate_resume(before, experiment_identity(root, settings, checkpoints))
    if change == 'source':
        source.write_text('a=2')
    elif change == 'config':
        config.write_text('{"changed": true}')
    elif change == 'weights':
        weight.write_bytes(b'modified')  # Same byte count, different content.
    elif change == 'settings':
        settings['layout'] = 'native'
    else:
        (checkpoints/'texture.pt').write_bytes(b'new')
    with pytest.raises(ValueError, match='identity'):
        validate_resume(before, experiment_identity(root, settings, checkpoints))


def test_legacy_manifest_requires_new_run():
    with pytest.raises(ValueError, match='missing'):
        validate_resume(None, {'schema': 1})
