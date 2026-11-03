"""CPU diagnostic for pipeline BF16 boundaries with FP32 GEMM accumulation.

Avoids this host's prohibitively slow native BF16 matrix products. This is an
explicit emulation of tensor rounding, not a CUDA/TT kernel oracle or a reason
to weaken the original FP32 numerical gates. Optional prepared FP32 execution
controls whether helper algebra changes the complete original trajectory.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

ROOT = Path(__file__).resolve().parents[2]


class PipelinePrecisionMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.promoted_ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        tensors = [value for value in tree_flatten((args, kwargs))[0] if isinstance(value, torch.Tensor)]
        bf16 = [value for value in tensors if value.dtype == torch.bfloat16]
        gemm = name in ('aten::mm', 'aten::bmm', 'aten::addmm', 'aten::baddbmm')
        attention = name == 'aten::_scaled_dot_product_flash_attention_for_cpu'
        if not bf16 or not (gemm or attention):
            return func(*args, **kwargs)
        if any(value.device.type != 'cpu' for value in tensors) or kwargs.get('out') is not None:
            raise ValueError('Precision diagnostic supports CPU inference without out= only')
        self.promoted_ops[name] += 1
        promote = lambda value: value.float() if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16 else value
        result = func(*tree_map(promote, args), **tree_map(promote, kwargs))
        if attention:
            # The auxiliary CPU flash-attention logsumexp remains FP32.
            return (result[0].to(torch.bfloat16), *result[1:])
        return result.to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('ss','shape','tex'), default='shape')
    p.add_argument('--active-voxels', type=int, default=512)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoints', type=Path, default=Path.home()/'models/genrecon')
    p.add_argument('--flow-inputs', type=Path, help='Prepared real-image inputs; currently supports --stage ss')
    p.add_argument('--precision', choices=('pipeline', 'float32'), default='pipeline')
    p.add_argument('--rope-mode', choices=('stock', 'real_pairs', 'real_pairs_head_first', 'real_pairs_flat_head_first', 'real_pairs_fp32_split'), default='stock')
    p.add_argument('--attention-mode', choices=('stock', 'fp32_reciprocal', 'fp32_scores_bf16_av', 'bf16_qk_fp32_softmax'), default='stock')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.flow_inputs is not None and args.stage != 'ss':
        p.error('flow-inputs currently supports ss only')
    os.environ.update(ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.tenstorrent.probes import build_case
    from experiments.tenstorrent.trajectory import TrajectoryFixture
    from experiments.tenstorrent.experiment_identity import sha256
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'phase':'build', 'seed':args.seed, 'threads':args.threads,
              'torch':torch.__version__, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'precision':args.precision, 'rope_mode':args.rope_mode, 'attention_mode':args.attention_mode,
              'source_files_sha256':{str(path.relative_to(ROOT)):sha256(path)
                  for folder in (ROOT/'genrecon', ROOT/'experiments/tenstorrent') for path in sorted(folder.rglob('*.py'))},
              'scope':('Prepared model, FP32 CPU complete trajectory control; no native BF16, CUDA or TT execution. Does not replace the original stock FP32 acceptance reference.' if args.precision=='float32' else
                       'CPU BF16-boundary/FP32-accumulation diagnostic, not native BF16, CUDA or TT execution; does not replace FP32 acceptance gates')}
    def save():
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        case = {'ss':'ss_flow','shape':'shape_flow_tensor','tex':'tex_flow_tensor'}[args.stage]
        model, inputs, metadata = build_case(case,args.checkpoints,
            active_voxels=None if args.stage=='ss' else args.active_voxels)
        if args.flow_inputs is not None:
            from experiments.tenstorrent.prepare_ss_inputs import load_ss_inputs
            inputs, fixture_metadata = load_ss_inputs(args.flow_inputs, inputs, metadata['checkpoint']['sha256'])
            metadata.update(fixture_metadata)
        model.eval()
        report['metadata'] = metadata
        stem = {'ss':'sparse_structure', 'shape':'shape_slat', 'tex':'texture_slat'}[args.stage]
        checkpoint_digest = sha256(args.checkpoints/(stem+'.pt'))
        if checkpoint_digest != metadata['checkpoint']['sha256']:
            raise ValueError('Physical checkpoint differs from recorded provenance')
        report['verified_checkpoint_sha256'] = checkpoint_digest
        fixture = TrajectoryFixture(args.stage,inputs,getattr(model,'coords',None),
            input_source='real posed-image' if args.flow_inputs is not None else 'synthetic')
        if args.rope_mode != 'stock':
            from experiments.tenstorrent.inference_rope import prepare_real_rope
            report['prepared_rope_blocks'] = prepare_real_rope(model,
                head_first=args.rope_mode in ('real_pairs_head_first', 'real_pairs_flat_head_first', 'real_pairs_fp32_split'),
                flatten=args.rope_mode in ('real_pairs_flat_head_first', 'real_pairs_fp32_split'),
                fp32_split=args.rope_mode=='real_pairs_fp32_split')
        if args.attention_mode != 'stock':
            from experiments.tenstorrent.inference_attention import prepare_attention
            report['prepared_attention_modules'] = prepare_attention(model,args.attention_mode)
        if args.precision == 'float32':
            from experiments.tenstorrent.inference_precision import prepare_float32_device
            report['precision_preparation'] = prepare_float32_device(model)
            inputs = tuple(value.float() if value.is_floating_point() else value for value in inputs)
        report['phase'] = 'prepared_fp32_forward_and_trajectory' if args.precision=='float32' else 'emulated_pipeline_forward_and_trajectory'
        save()
        mode = PipelinePrecisionMode()
        context = mode if args.precision=='pipeline' else nullcontext()
        with torch.no_grad(), context:
            start = time.perf_counter()
            forward = model(*inputs)
            report['forward_seconds'] = time.perf_counter()-start
            payload,report['trajectory'] = fixture.run(
                lambda x,t,cond:model(x,t,cond['context'],cond['projected']))
        torch.save(forward,args.output_dir/'forward.pt')
        torch.save(payload,args.output_dir/'trajectory.pt')
        report.update(status='completed_diagnostic',phase='complete',promoted_ops=dict(mode.promoted_ops),
                      artifact_sha256={name:hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest()
                                       for name in ('forward.pt','trajectory.pt')})
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
