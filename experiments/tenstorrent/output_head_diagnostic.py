"""Teacher-force the original CPU output head with accepted captured TT states.

Combined normalization/projection error is observable from the captured hidden
state and prediction, without changing device lowering. CPU stage comparisons
measure propagation/amplification; these errors are not additive components.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-directory', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path.home() / 'models/genrecon/shape_slat.pt')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    from experiments.tenstorrent.experiment_identity import sha256
    from experiments.tenstorrent.validation import compare_tensors
    origin = json.loads((args.capture_directory / 'result.json').read_text())
    if not (origin['status'] == 'completed_numerical_diagnostic' and
            origin['captured_final_bitwise_matches_ordinary'] and
            origin['repeated_all_states_bitwise'] and not origin['executed_fallback_ops'] and
            not origin['aten_counters']):
        raise ValueError('Source capture was not observationally accepted')
    if origin['model']['config'] != 'configs/gen/slat_flow_img2shape/genrecon_512.json':
        raise ValueError('Expected original shape-flow head')
    torch.set_num_threads(origin['threads'])
    digest = sha256(args.checkpoint)
    if digest != origin['model']['checkpoint']['sha256']:
        raise ValueError('Physical checkpoint differs from source capture')
    weights = torch.load(args.checkpoint, weights_only=True, mmap=True)
    states = {}
    for key, file in [('fp32', 'cpu-fp32.pt'), ('cpu_precision', 'cpu-precision-diagnostic.pt'),
                      ('tt', 'captured-tt.pt')]:
        path = args.capture_directory / file
        if sha256(path) != origin['artifacts_sha256'][file]:
            raise ValueError('Captured tensor artifact changed')
        states[key] = torch.load(path, weights_only=True, mmap=True)
    norms = {k: F.layer_norm(v[-1].float(), v[-1].shape[-1:]) for k, v in states.items()}
    heads = {k: F.linear(v, weights['out_layer.weight'], weights['out_layer.bias'])[0]
             for k, v in norms.items()}
    for key in ('fp32', 'cpu_precision'):
        torch.testing.assert_close(heads[key], states[key][0], rtol=0, atol=0)
    report = {'status': 'completed_teacher_forced_head_diagnostic', 'scope': __doc__,
              'script_sha256': sha256(Path(__file__)), 'torch': torch.__version__,
              'checkpoint_sha256': digest,
              'source_capture_sha256': sha256(args.capture_directory / 'result.json'),
              'source_artifacts_sha256': origin['artifacts_sha256'],
              'original_cpu_head_reconstruction_bitwise': True, 'measurements': {}}
    for name in ('cpu_precision', 'tt'):
        report['measurements'][name] = {
            'hidden_vs_stock_fp32': compare_tensors(states[name][-1].float(), states['fp32'][-1]),
            'cpu_normalized_hidden_vs_stock': compare_tensors(norms[name], norms['fp32']),
            'cpu_head_on_hidden_vs_stock_prediction': compare_tensors(heads[name], states['fp32'][0]),
            'observed_prediction_vs_stock_prediction': compare_tensors(states[name][0], states['fp32'][0]),
            'observed_prediction_vs_cpu_head_on_same_hidden': compare_tensors(states[name][0], heads[name])}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({'hidden': {k: v[-1].clone() for k, v in states.items()},
                'observed_prediction': {k: v[0].clone() for k, v in states.items()},
                'cpu_head': heads}, args.output_dir / 'states.pt')
    report['states_sha256'] = sha256(args.output_dir / 'states.pt')
    (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['measurements']['tt'], indent=2))


if __name__ == '__main__':
    main()
