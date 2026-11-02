"""Decode matched sampler latents with the same FP32 CPU occupancy decoder.

Isolates sampler differences from TT decoder differences on one scene crop.
Reports numerical failures alongside occupancy overlap; no mesh-quality claim
or automatic quality promotion from an IoU value.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def overlap(actual, reference):
    import torch
    if actual.shape != reference.shape or actual.dtype != torch.bool or reference.dtype != torch.bool:
        raise ValueError('Expected matching boolean occupancy grids')
    intersection = int((actual & reference).sum())
    union = int((actual | reference).sum())
    a, b = int(actual.sum()), int(reference.sum())
    return {'actual_occupied': a, 'reference_occupied': b,
            'intersection': intersection, 'union': union,
            'iou': intersection / union if union else 1.,
            'dice': 2 * intersection / (a + b) if a + b else 1.,
            'false_positive': int((actual & ~reference).sum()),
            'false_negative': int((~actual & reference).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='Stock FP32 CPU trajectory payload')
    p.add_argument('--candidates', nargs='+', required=True, help='NAME=/path/to/trajectory.pt')
    p.add_argument('--decoder', type=Path, default=Path.home() / 'models/genrecon/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='reference')
    import torch
    from torch.nn import functional as F
    from genrecon import models
    from experiments.tenstorrent.experiment_identity import sha256
    from experiments.tenstorrent.trajectory import compare_trajectories
    torch.set_num_threads(args.threads)
    paths = {'cpu_fp32': args.reference}
    for item in args.candidates:
        name, path = item.split('=', 1)
        if not name or name in paths:
            raise ValueError('Candidate names must be unique and cannot be cpu_fp32')
        paths[name] = Path(path)
    payloads = {name: torch.load(path, weights_only=True, map_location='cpu') for name, path in paths.items()}
    reference = payloads['cpu_fp32']
    if reference['fixture_identity']['stage'] != 'ss':
        raise ValueError('Expected occupancy-flow trajectories')
    report = {'status': 'running', 'scope': __doc__, 'torch': torch.__version__,
              'script_sha256': sha256(Path(__file__)),
              'source_artifacts_sha256': {name: sha256(path) for name, path in paths.items()},
              'decoder_sha256': {ext: sha256(args.decoder.with_suffix(ext)) for ext in ('.json', '.safetensors')},
              'precision': 'Same official decoder, FP32 CPU for every latent; fixed zero threshold and original max-pool 2',
              'trajectory_checks': {name: compare_trajectories(payload, reference)
                                    for name, payload in payloads.items() if name != 'cpu_fp32'}}
    def save():
        (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    decoder = models.from_pretrained(str(args.decoder)).float().eval()
    decoder.convert_to_fp32()
    occupied, report['decode_seconds'] = {}, {}
    with torch.no_grad():
        for name, payload in payloads.items():
            start = time.perf_counter()
            logits = decoder(payload['samples'].float())
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite occupancy-decoder output')
            occupied[name] = F.max_pool3d((logits > 0).float(), 2, 2) > 0
            report['decode_seconds'][name] = time.perf_counter() - start
    report['grid_shape'] = list(occupied['cpu_fp32'].shape)
    report['overlap_vs_cpu_fp32'] = {name: overlap(value, occupied['cpu_fp32'])
                                     for name, value in occupied.items() if name != 'cpu_fp32'}
    report['pairwise_overlap'] = {a + '_vs_' + b: overlap(occupied[a], occupied[b])
                                 for a, b in itertools.combinations(occupied, 2)}
    torch.save(occupied, args.output_dir / 'occupancy.pt')
    report['occupancy_sha256'] = sha256(args.output_dir / 'occupancy.pt')
    report['status'] = 'completed_occupancy_diagnostic'
    save()
    print(json.dumps(report['overlap_vs_cpu_fp32'], indent=2), flush=True)


if __name__ == '__main__':
    main()
