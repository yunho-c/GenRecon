"""Matched fixed-input trajectories using the original Euler/CFG samplers."""
import hashlib
import json
from pathlib import Path
import time

import torch

from experiments.tenstorrent.scene_sampling import TensorFlowAdapter, MeasuredSampler, features
from experiments.tenstorrent.validation import compare_tensors
from genrecon.modules.sparse import SparseTensor

CONFIG = Path(__file__).resolve().parents[2]/'configs/pipelines/texture.json'


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f'{value.dtype}:{list(value.shape)}'.encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TrajectoryFixture:
    def __init__(self, stage, inputs, coords=None, *, input_source='synthetic'):
        family = {'ss': 'sparse_structure', 'shape': 'shape_slat', 'tex': 'tex_slat'}[stage]
        self.stage = stage
        self.input_source = input_source
        spec = json.loads(CONFIG.read_text())['args'][family+'_sampler']
        self.params = spec['params']
        self.sigma_min = spec['args']['sigma_min']
        self.config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.positive = {'context': inputs[2].detach().cpu().clone(),
                         'projected': inputs[3].detach().cpu().clone()}
        self.negative = {key: torch.zeros_like(value) for key, value in self.positive.items()}
        self.concat = None
        self.coords = None if coords is None else coords.detach().cpu().clone()
        values = inputs[0].detach().cpu().clone()
        if stage == 'ss':
            if values.shape[0] != 1:
                raise ValueError('Only one fixed chunk is supported')
            self.noise = values
        else:
            if self.coords is None or len(self.coords) != len(values) or torch.any(self.coords[:, 0] != 0):
                raise ValueError('Expected matching single-chunk coordinates')
            if stage == 'tex':
                if values.shape[-1] != 64:
                    raise ValueError('Expected 32 texture plus 32 normalized shape channels')
                self.concat = SparseTensor(values[:, 32:], self.coords)
                values = values[:, :32]
            self.noise = SparseTensor(values, self.coords)

    def run(self, forward):
        adapter = TensorFlowAdapter(forward)
        sampler = MeasuredSampler(self.sigma_min)
        identity = {'stage': self.stage, 'parameters': dict(self.params), 'sigma_min': self.sigma_min,
                    'sampler_config_sha256': self.config_sha256,
                    'conditions': {branch: {key: tensor_digest(value) for key, value in cond.items()}
                                   for branch, cond in (('positive', self.positive), ('negative', self.negative))},
                    'concat_sha256': None if self.concat is None else tensor_digest(self.concat.feats)}
        start = time.perf_counter()
        result = sampler.sample(adapter, self.noise, cond=self.positive, neg_cond=self.negative,
                                concat_cond=self.concat, verbose=False, **self.params)
        seconds = time.perf_counter()-start
        expected_calls = {'ss': 22, 'shape': 21, 'tex': 18}[self.stage]
        if len(adapter.calls) != expected_calls or len(sampler.steps) != 12:
            raise ValueError('Unexpected original sampler schedule')
        payload = {'fixture_identity': identity, 'coords': self.coords, 'noise': features(self.noise).clone(),
                   'samples': features(result.samples),
                   'pred_x_t': [features(x) for x in result.pred_x_t],
                   'pred_x_0': [features(x) for x in result.pred_x_0]}
        measurements = {'seconds': seconds, 'forwards': adapter.calls, 'steps': sampler.steps,
                        'parameters': self.params, 'sampler_config_sha256': self.config_sha256,
                        'scope': f'One complete original 12-step trajectory, fixed noise and {self.input_source} conditioning; no reconstruction-quality assessment'}
        return payload, measurements


def compare_trajectories(actual, expected, *, allow_legacy=False):
    identity_verified = bool(actual.get('fixture_identity'))
    legacy = not actual.get('fixture_identity') and not expected.get('fixture_identity')
    if not (allow_legacy and legacy) and (
        not identity_verified or actual['fixture_identity'] != expected.get('fixture_identity')):
        raise ValueError('Trajectory conditions or sampler configuration differ or are missing')
    if not torch.equal(actual['noise'], expected['noise']):
        raise ValueError('Trajectory noise differs')
    if (actual['coords'] is None) != (expected['coords'] is None) or (
        actual['coords'] is not None and not torch.equal(actual['coords'], expected['coords'])):
        raise ValueError('Trajectory coordinates differ')
    checks = {'fixture_identity_verified': identity_verified,
              'identity_note': 'Condition/config hashes match' if identity_verified else 'Legacy artifacts: condition/config hashes unavailable; numerical diagnostics only',
              'final': compare_tensors(actual['samples'], expected['samples'])}
    for key in ('pred_x_t', 'pred_x_0'):
        if len(actual[key]) != 12 or len(expected[key]) != 12:
            raise ValueError('Expected all twelve original sampling steps')
        checks[key] = [compare_tensors(a, e) for a, e in zip(actual[key], expected[key], strict=True)]
    checks['all_states_pass_smoke'] = checks['final']['passed'] and all(
        check['passed'] for key in ('pred_x_t', 'pred_x_0') for check in checks[key])
    return checks
