"""Seal and reuse completed stock FP32 CPU trajectories with strict identity.

Cache creation snapshots a trusted local experiment's completed CPU artifact;
its GPU run may still be pending. Reuse checks the sealed file hash, original
model/reference code, checkpoint, versions and every fixture input. Preparation
experiments and CLI changes do not invalidate an unchanged stock reference.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from experiments.tenstorrent.experiment_identity import sha256

REFERENCE_FILES = {'flow_probes.py', 'probes.py', 'reference.py', 'trajectory.py', 'scene_sampling.py'}


def signature(report):
    if report.get('reference_precision') != 'float32' or not report.get('strict_loading'):
        raise ValueError('Cache requires the original strictly loaded FP32 reference')
    # Serialized input files include preparation timings/provenance that can
    # change on a stateless restore. Exact tensor/config checks below establish
    # semantic identity; retain original file hashes in the source report.
    keys = ('case', 'checkpoint', 'config', 'dtype', 'versions', 'seed', 'inputs')
    return {**{key: report.get(key) for key in keys},
            'reference_sources': {key: value for key, value in report['source_files_sha256'].items()
                                  if key.startswith('genrecon/') or Path(key).name in REFERENCE_FILES}}


def create_cache(source_case, destination):
    source_case, destination = Path(source_case), Path(destination)
    report = json.loads((source_case / 'result.json').read_text())
    if not report.get('cpu_trajectory') or report.get('phase') in ('imports', 'build', 'cpu_reference', 'stock_cpu_trajectory'):
        raise ValueError('Source CPU trajectory has not completed')
    identity = signature(report)
    source = source_case / 'trajectory-reference.pt'
    digest = sha256(source)
    recorded = report.get('trajectory_artifacts_sha256', {}).get(source.name)
    if recorded is not None and recorded != digest:
        raise ValueError('Source CPU trajectory hash changed')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, destination / 'reference.pt')
    if sha256(destination / 'reference.pt') != digest:
        raise ValueError('CPU reference changed during snapshot')
    manifest = {'schema': 1, 'signature': identity, 'reference_sha256': digest,
                'source_case': str(source_case.resolve()), 'source_report': report,
                'cpu_trajectory': report['cpu_trajectory'],
                'seal_note': 'Snapshot of trusted local completed CPU reference; SHA256 sealed at cache creation. Source GPU execution may still be running.'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_cache(directory, report, fixture):
    import torch
    from experiments.tenstorrent.scene_sampling import features
    from experiments.tenstorrent.trajectory import tensor_digest
    start = time.perf_counter()
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest['signature'] != signature(report):
        raise ValueError('CPU cache model/reference/checkpoint/version/input identity changed')
    if sha256(directory / 'reference.pt') != manifest['reference_sha256']:
        raise ValueError('CPU cache artifact hash changed')
    payload = torch.load(directory / 'reference.pt', weights_only=True, map_location='cpu')
    identity = {'stage': fixture.stage, 'parameters': dict(fixture.params), 'sigma_min': fixture.sigma_min,
                'sampler_config_sha256': fixture.config_sha256,
                'conditions': {branch: {key: tensor_digest(value) for key, value in cond.items()}
                               for branch, cond in (('positive', fixture.positive), ('negative', fixture.negative))},
                'concat_sha256': None if fixture.concat is None else tensor_digest(fixture.concat.feats)}
    if payload.get('fixture_identity') != identity:
        raise ValueError('CPU cache conditions or sampler configuration differ')
    noise = features(fixture.noise)
    if not torch.equal(payload['noise'], noise):
        raise ValueError('CPU cache noise differs')
    if (payload['coords'] is None) != (fixture.coords is None) or (
            fixture.coords is not None and not torch.equal(payload['coords'], fixture.coords)):
        raise ValueError('CPU cache coordinates differ')
    if len(payload['pred_x_t']) != 12 or len(payload['pred_x_0']) != 12:
        raise ValueError('CPU cache must contain every original sampler state')
    for value in (payload['samples'], *payload['pred_x_t'], *payload['pred_x_0']):
        if value.shape != noise.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('CPU cache states must be finite FP32 tensors of the expected shape')
    return payload, manifest['cpu_trajectory'], {
        'directory': str(directory.resolve()), 'manifest_sha256': sha256(directory / 'manifest.json'),
        'reference_sha256': manifest['reference_sha256'], 'load_validate_seconds': time.perf_counter()-start,
        'note': 'Stock FP32 trajectory reused after strict identity/hash checks; stored CPU duration describes the original computation, not this run.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-case', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    create_cache(args.source_case, args.output_dir)
    print('Sealed CPU trajectory cache:', args.output_dir)
