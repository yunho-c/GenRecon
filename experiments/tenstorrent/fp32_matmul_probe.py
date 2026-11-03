"""Test native FP32 matmul preservation and explicit rounding counterfactuals.

Seeded synthetic matrices, not learned attention or complete-model attribution.
Identity operands remain dynamic native inputs. All inputs and outputs use
FP32 storage; some cases intentionally contain exactly BF16-representable values.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def quantize_tf32(x, rounded=False):
    """CPU counterfactual for normal finite FP32 inputs; retain ten fraction bits."""
    bits = x.contiguous().view(torch.int32)
    if rounded:
        bits = bits + 0xfff + ((bits >> 13) & 1)
    return (bits & -8192).view(torch.float32)


class MatrixProduct(nn.Module):
    def forward(self,a,b):
        return a @ b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--math-fidelities',nargs='+',choices=('hifi4','hifi3'),default=['hifi4'])
    args = parser.parse_args()
    if args.runs<1:
        parser.error('runs must be positive')
    if len(args.math_fidelities)!=1:
        parser.error('Use a fresh process for each fidelity: the native graph cache can reuse prior settings despite torch.compiler.reset()')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in ('LD_LIBRARY_PATH','PYTHONPATH','TT_METAL_HOME','TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(name,None)
    os.environ.update(PJRT_DEVICE='TT',ATTN_BACKEND='sdpa',
                      SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    report={'status':'running','scope':__doc__,'seed':62,'math_fidelities':args.math_fidelities,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'cases':{}}
    def phase(name):
        report['phase']=name
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,flush=True)
    phase('imports')
    try:
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(4)
        torch.manual_seed(report['seed'])
        report['versions']={name:importlib.metadata.version(name)
                            for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['script_sha256']=sha256(Path(__file__))
        a=torch.randn(512,128)
        b=torch.randn(128,512)
        identity=torch.eye(128)
        cancel_a=torch.zeros(512,128)
        cancel_b=torch.zeros(128,128)
        cancel_a[:,0],cancel_a[:,1]=1+2**-12,1.
        cancel_b[0,:],cancel_b[1,:]=4096.,-4096.
        bf16_cancel_a=torch.zeros(512,128)
        bf16_cancel_b=torch.zeros(128,128)
        bf16_cancel_a[:,0],bf16_cancel_a[:,1]=1+2**-7,1.
        bf16_cancel_b[0,:],bf16_cancel_b[1,:]=1+2**-7,-(1+2**-6)
        cases={'identity_fp32':(a,identity),
               'identity_bf16_values_fp32_storage':(a.bfloat16().float(),identity),
               'product_fp32':(a,b),
               'product_bf16_values_fp32_storage':(a.bfloat16().float(),b.bfloat16().float()),
               'product_bf16_values_k32':(a[:,:32].contiguous().bfloat16().float(),b[:32,:].contiguous().bfloat16().float()),
               'product_bf16_values_k1':(a[:,:1].contiguous().bfloat16().float(),b[:1,:].contiguous().bfloat16().float()),
               'cancellation_fp32':(cancel_a,cancel_b),
               'cancellation_bf16_values':(bf16_cancel_a,bf16_cancel_b)}
        torch.testing.assert_close(cancel_a @ cancel_b,torch.ones(512,128),rtol=0,atol=0)
        torch.testing.assert_close(bf16_cancel_a @ bf16_cancel_b,torch.full((512,128),2**-14),rtol=0,atol=0)
        cases={((name if args.math_fidelities==['hifi4'] else fidelity+'_'+name)):(fidelity,left,right)
               for fidelity in args.math_fidelities for name,(left,right) in cases.items()}
        phase('waiting_for_device')
        lock=os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX)
        device=torch_xla.device()
        outputs={}
        for name,(fidelity,left,right) in cases.items():
            phase(name)
            reference=left @ right
            counterfactuals={'tf32_truncated_inputs':quantize_tf32(left) @ quantize_tf32(right),
                            'tf32_rounded_inputs':quantize_tf32(left,True) @ quantize_tf32(right,True),
                            'tf32_truncated_output':quantize_tf32(reference),
                            'tf32_rounded_output':quantize_tf32(reference,True),
                            'bf16_rounded_output':reference.bfloat16().float()}
            if 'identity' in name:
                torch.testing.assert_close(reference,left,rtol=0,atol=0)
            torch.compiler.reset()
            options={'optimization_level':0,'math_fidelity':fidelity,'dry_run':'false',
                     'export_path':str(args.output_dir/(name+'-native-ir'))}
            torch_xla.set_custom_compile_options(options)
            compiled=torch.compile(MatrixProduct().eval(),backend='tt',fullgraph=True)
            resident=(left.to(device),right.to(device))
            readback=[value.cpu() for value in resident]
            input_roundtrip=[torch.equal(value,original) for value,original in zip(readback,(left,right),strict=True)]
            if not all(input_roundtrip):
                raise ValueError('Input device roundtrip changed FP32 values')
            metrics.clear_all()
            with torch.no_grad():
                actual=compiled(*resident).cpu()
                times=[]
                for _ in range(args.runs):
                    start=time.perf_counter();repeated=compiled(*resident).cpu()
                    times.append(time.perf_counter()-start)
                    torch.testing.assert_close(actual,repeated,rtol=0,atol=0)
            row={'requested_compile_options':options,
                 'input_digests':[tensor_digest(left),tensor_digest(right)],
                 'input_shapes':[list(left.shape),list(right.shape)],
                 'input_roundtrip_bitwise':input_roundtrip,
                 'output_dtype':str(actual.dtype),'native_vs_cpu':compare_tensors(actual,reference),
                 'reference_minmax':[reference.min().item(),reference.max().item()],
                 'native_minmax':[actual.min().item(),actual.max().item()],
                 'native_bitwise_cpu':torch.equal(actual,reference),'warm_seconds':times,
                 'counterfactuals':{key:{'native_vs_cpu_counterfactual':compare_tensors(actual,value),
                                         'native_bitwise_cpu_counterfactual':torch.equal(actual,value)}
                                    for key,value in counterfactuals.items()},
                 'executed_fallback_ops':list(metrics.executed_fallback_ops()),
                 'aten_counters':{key:metrics.counter_value(key) for key in metrics.counter_names() if key.startswith('aten::')}}
            if row['executed_fallback_ops'] or row['aten_counters']:
                raise ValueError('Native matrix probe used fallback')
            report['cases'][name]=row
            outputs[name]=actual
            (args.output_dir/(name+'-metrics.txt')).write_text(metrics.metrics_report())
        torch.save(outputs,args.output_dir/'outputs.pt')
        report['outputs_sha256']=sha256(args.output_dir/'outputs.pt')
        report['status']='completed_native_matmul_probe'
        phase('complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        phase(report.get('phase','initialization'))
        raise


if __name__=='__main__':
    main()
