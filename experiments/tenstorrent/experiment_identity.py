"""Content identity for resumable staged experiments (outside timed inference)."""
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment_identity(root, settings, checkpoints):
    root, checkpoints = Path(root), Path(checkpoints)
    sources = sorted({*root.joinpath('genrecon').rglob('*.py'),
                      *root.joinpath('experiments/tenstorrent').rglob('*.py'),
                      *root.joinpath('configs').rglob('*.json')})
    weights = sorted(p for p in checkpoints.rglob('*') if p.is_file()
                     and p.suffix in ('.pt', '.safetensors', '.json'))
    return {'schema': 1, 'settings': dict(settings), 'checkpoints': str(checkpoints.resolve()),
            'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sources},
            'checkpoint_sha256': {str(p.relative_to(checkpoints)): sha256(p) for p in weights}}


def validate_resume(previous, current):
    if previous != current:
        raise ValueError('Resume experiment identity changed or is missing; use a fresh output directory')
