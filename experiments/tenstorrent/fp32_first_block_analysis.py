"""Compare accepted native boundaries with CPU operations on identical inputs.

These local errors are not additive components of model/sampler error. CPU
reconstruction, physical checkpoint and source/artifact identity are mandatory.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))


def first_block_references(flow, inputs, states):
    """CPU arithmetic consumes observed TT intermediates, never stock substitutes."""
    block = flow.blocks[0]
    shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation+states['modulation']).type(states['modulation'].dtype).chunk(6,dim=1)
    expected = {
        'input_layer':flow.input_layer(inputs[0]).unsqueeze(0).to(flow.dtype),
        'modulation':flow.adaLN_modulation(flow.t_embedder(inputs[1])).to(flow.dtype),
        'first_projected':flow.proj_linears[0](inputs[3].unsqueeze(0).to(flow.dtype)),
        'first_input':states['input_layer']+states['first_projected'],
        'first_norm1':block.norm1(states['first_input']),
        'first_self_input':states['first_norm1']*(1+scale.unsqueeze(1))+shift.unsqueeze(1),
        'first_self_output':block.self_attn(states['first_self_input'],phases=flow.phases),
        'first_after_self':states['first_input']+states['first_self_output']*gate.unsqueeze(1),
        'first_cross_input':block.norm2(states['first_after_self']),
        'first_cross_output':block.cross_attn(states['first_cross_input'],inputs[2].to(flow.dtype)),
        'first_after_cross':states['first_after_self']+states['first_cross_output'],
        'first_norm3':block.norm3(states['first_after_cross']),
        'first_mlp_input':states['first_norm3']*(1+scale_mlp.unsqueeze(1))+shift_mlp.unsqueeze(1),
        'first_mlp_output':block.mlp(states['first_mlp_input']),
        'block_00':states['first_after_cross']+states['first_mlp_output']*gate_mlp.unsqueeze(1),
    }
    last = states[f'block_{len(flow.blocks)-1:02d}']
    expected['final'] = flow.out_layer(F.layer_norm(last.float(),last.shape[-1:]))[0]
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-directory',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,default=Path.home()/'models/genrecon')
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    os.environ.update(ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    from experiments.tenstorrent.flow_probes import build_flow_case
    from experiments.tenstorrent.flow_observation import ShapeFlowObservation
    from experiments.tenstorrent.inference_rope import prepare_real_rope
    from experiments.tenstorrent.inference_attention import prepare_attention
    from experiments.tenstorrent.inference_precision import prepare_float32_device
    from experiments.tenstorrent.trajectory import tensor_digest
    from experiments.tenstorrent.experiment_identity import sha256
    from experiments.tenstorrent.validation import compare_tensors
    origin = json.loads((args.capture_directory/'result.json').read_text())
    if not (origin['status']=='completed_numerical_diagnostic' and origin['first_block'] and
            origin['captured_final_bitwise_matches_ordinary'] and origin['repeated_all_states_bitwise'] and
            not origin['executed_fallback_ops'] and not origin['aten_counters']):
        raise ValueError('Detailed source capture was not observationally accepted')
    source_proof = {key:sha256(ROOT/key)==value for key,value in origin['source_sha256'].items()
                    if key.startswith('genrecon/') or Path(key).name in
                    {'flow_probes.py','flow_observation.py','reference.py','inference_rope.py',
                     'inference_attention.py','inference_precision.py'}}
    if not all(source_proof.values()):
        raise ValueError('Captured model/helper source changed')
    checkpoint_sha = sha256(args.checkpoints/'shape_slat.pt')
    if checkpoint_sha!=origin['verified_checkpoint_sha256']:
        raise ValueError('Physical checkpoint changed')
    states = {}
    for label,name in [('tt','captured-tt.pt'),('prepared','cpu-prepared.pt'),('stock','cpu-stock.pt')]:
        path = args.capture_directory/name
        if sha256(path)!=origin['artifacts_sha256'][name]:
            raise ValueError('Captured tensor artifact changed')
        states[label] = torch.load(path,weights_only=True,map_location='cpu')
    torch.set_num_threads(origin['threads'])
    torch.manual_seed(origin['seed'])
    flow,inputs,metadata = build_flow_case('shape_flow_tensor',args.checkpoints)
    if [tensor_digest(value) for value in inputs]!=origin['input_digests']:
        raise ValueError('Original fixture changed')
    prepare_real_rope(flow,head_first=True,flatten=True)
    prepare_attention(flow,'fp32_reciprocal')
    prepare_float32_device(flow)
    observation = ShapeFlowObservation(flow,first_block=True).eval()
    with torch.no_grad():
        reproduced = dict(zip(observation.names,observation(*inputs),strict=True))
        for name,value in reproduced.items():
            torch.testing.assert_close(value,states['prepared'][name],rtol=0,atol=0)
        references = first_block_references(flow,inputs,states['tt'])
        cpu_references = first_block_references(flow,inputs,states['prepared'])
        for name,value in cpu_references.items():
            torch.testing.assert_close(value,states['prepared'][name],rtol=0,atol=0)
    local = {name:{'native_vs_cpu_on_same_inputs':compare_tensors(states['tt'][name],reference),
                   'native_bitwise_cpu_on_same_inputs':torch.equal(states['tt'][name],reference),
                   'native_vs_prepared_flow':compare_tensors(states['tt'][name],states['prepared'][name])}
             for name,reference in references.items()}
    report = {'status':'completed_teacher_forced_first_block', 'scope':__doc__,
              'script_sha256':sha256(Path(__file__)), 'torch':torch.__version__,
              'source_capture_sha256':sha256(args.capture_directory/'result.json'),
              'verified_checkpoint_sha256':checkpoint_sha, 'source_proof':source_proof,
              'source_artifacts_sha256':origin['artifacts_sha256'],
              'complete_prepared_cpu_reconstruction_bitwise':True,
              'cpu_boundaries_reconstruct_bitwise':True,'measurements':local,
              'interpretation':'Each CPU expectation consumes the same observed TT input values. Modulated boundaries combine parameter addition, scaling and shifts. Attention/MLP/head are combined local operations. Ranking local relative errors does not rank contributions to final sampler drift.'}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:row['native_vs_cpu_on_same_inputs']['relative_l2'] for name,row in local.items()},indent=2))


if __name__=='__main__':
    main()
