"""CPU diagnostic for pipeline BF16 boundaries with FP32 GEMM accumulation.

Avoids this host's prohibitively slow native BF16 matrix products. This is an
explicit emulation of tensor rounding, not a CUDA/TT kernel oracle or a reason
to weaken the original FP32 numerical gates.
"""
import argparse
from collections import Counter
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
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    os.environ.update(ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments.tenstorrent.probes import build_case
    from experiments.tenstorrent.trajectory import TrajectoryFixture
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'phase':'build', 'seed':args.seed, 'threads':args.threads,
              'torch':torch.__version__, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope':'CPU BF16-boundary/FP32-accumulation diagnostic, not native BF16, CUDA or TT execution; does not replace FP32 acceptance gates'}
    def save():
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        case = {'ss':'ss_flow','shape':'shape_flow_tensor','tex':'tex_flow_tensor'}[args.stage]
        model, inputs, metadata = build_case(case,args.checkpoints,
            active_voxels=None if args.stage=='ss' else args.active_voxels)
        model.eval()
        report['metadata'] = metadata
        fixture = TrajectoryFixture(args.stage,inputs,getattr(model,'coords',None))
        report['phase'] = 'emulated_pipeline_forward_and_trajectory'
        save()
        mode = PipelinePrecisionMode()
        with torch.no_grad(), mode:
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
