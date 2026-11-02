"""Prepare matched occupancy-flow noise and original real-image conditioning.

DINO features, projection and learned aggregation run on FP32 CPU. The saved
inputs isolate flow/sampler accuracy from upstream TT errors. No sampling or
reconstruction-quality validation is performed here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.tenstorrent.experiment_identity import sha256


def load_ss_inputs(path, expected_inputs, checkpoint_sha256):
    import torch
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if payload.get('schema') != 1 or payload.get('stage') != 'ss':
        raise ValueError('Expected schema-1 occupancy-flow inputs')
    if payload.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('Flow-input conditioning checkpoint differs from loaded weights')
    inputs = payload.get('inputs')
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 4:
        raise ValueError('Expected noise, timestep, context and projected conditioning')
    for value, expected in zip(inputs, expected_inputs, strict=True):
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('Flow-input tensor shape/dtype differs from supported occupancy layout')
        if not torch.isfinite(value).all():
            raise ValueError('Flow inputs must be finite')
    return tuple(inputs), {'input_kind': 'real posed-image conditioning; FP32 CPU projection/aggregation',
                           'flow_inputs_sha256': sha256(path),
                           'flow_inputs_provenance': payload['provenance']}


def main():
    import torch
    from experiments.tenstorrent.flow_probes import load_model
    from experiments.tenstorrent.generate_decoder_fixture import conditioning
    from experiments.tenstorrent.reference import float32_reference
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditioning-dir', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, default=Path.home() / 'models/genrecon')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    source = args.conditioning_dir / 'conditioning.pt'
    provenance_path = args.conditioning_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if sha256(source) != provenance.get('conditioning_sha256'):
        raise ValueError('Conditioning artifact hash differs from provenance')
    data = torch.load(source, weights_only=True, map_location='cpu')
    model, _, metadata = load_model('ss', args.checkpoints)
    start = time.perf_counter()
    with torch.no_grad(), float32_reference(model):
        cond = conditioning(model, data['features'], data['extrinsics'], data['intrinsics'])['cond']
    seconds = time.perf_counter() - start
    generator = torch.Generator().manual_seed(args.seed)
    inputs = (torch.randn(1, model.in_channels, *([model.resolution] * 3), generator=generator),
              torch.tensor([1000.]), cond['cond_2D'].clone(), cond['cond_3D'].clone())
    origin = {'conditioning_sha256': sha256(source),
              'conditioning_provenance_sha256': sha256(provenance_path),
              'dino_sha256': provenance['dino_sha256'],
              'dino_conversion': provenance.get('dino_conversion'),
              'script_sha256': sha256(Path(__file__)), 'noise_seed': args.seed,
              'projection_aggregation_precision': 'stock FP32 CPU',
              'projection_aggregation_seconds': seconds}
    payload = {'schema': 1, 'stage': 'ss', 'checkpoint_sha256': metadata['checkpoint']['sha256'],
               'inputs': inputs, 'provenance': origin}
    path = args.output_dir / 'inputs.pt'
    torch.save(payload, path)
    load_ss_inputs(path, inputs, payload['checkpoint_sha256'])
    report = {'status': 'prepared_flow_inputs', 'stage': 'ss',
              'checkpoint': metadata['checkpoint'], 'provenance': origin,
              'inputs_sha256': sha256(path),
              'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in inputs],
              'scope': __doc__}
    (args.output_dir / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Prepared matched occupancy-flow inputs:', path, flush=True)


if __name__ == '__main__':
    import os
    import sys
    os.environ.update(ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Keep script-style flow probe dependencies available.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
