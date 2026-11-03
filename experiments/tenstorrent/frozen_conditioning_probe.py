"""Seeded tiny native smoke and consteval-placement control, not model performance.

Uses the shared two-block nonzero-LoRA test fixture. Separate processes for
dynamic and fixed conditioning; all branch outputs and policy metadata retained.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def summarize_functions(text):
    functions={}
    current=None
    for line in text.splitlines():
        match=re.match(r'\s*func\.func\s+(?:(?:public|private)\s+)?@([^\s(]+)',line)
        if match:
            current=match[1]
            functions[current]={'linear_matmul_ops':0,'conditioning_products':0}
        if current and re.search(r'"ttnn\.(linear|matmul)"',line):
            functions[current]['linear_matmul_ops']+=1
            operand=re.search(r': \(tensor<([^,>]+)',line)
            if operand and 'xf32' in operand[1]:
                dimensions=operand[1].split('xf32')[0].split('x')
                if dimensions[-2:] in (['9','24'],['17','24']):
                    functions[current]['conditioning_products']+=1
    return functions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('dynamic','frozen'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','TT_METAL_HOME','TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key,None)
    os.environ.update(PJRT_DEVICE='TT',ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    report={'status':'running','scope':__doc__,'mode':args.mode,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def phase(name):
        report['phase']=name
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,flush=True)
    phase('imports')
    try:
        import torch
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from experiments.tenstorrent.test_frozen_conditioning import fixture
        from experiments.tenstorrent.frozen_conditioning import FrozenConditionFlow
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['source_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in
            [Path(__file__),Path(__file__).with_name('frozen_conditioning.py'),
             Path(__file__).with_name('test_frozen_conditioning.py'),Path(__file__).with_name('test_flow_observation.py')]}
        flow,positive,negative=fixture()
        x,t=torch.randn(17,8),torch.tensor([.6])
        original_parameters=dict(flow.named_parameters())
        identities={id(value) for value in original_parameters.values()}
        report['original_parameter_digests']={name:tensor_digest(value) for name,value in original_parameters.items()}
        report['input_digests']={'x':tensor_digest(x),'t':tensor_digest(t),
            'positive':{key:tensor_digest(value) for key,value in positive.items()},
            'negative':{key:tensor_digest(value) for key,value in negative.items()}}
        with torch.no_grad():
            reference={name:flow(x,t,branch['context'],branch['projected']) for name,branch in [('positive',positive),('negative',negative)]}
            changed_x=x+.125
            changed_reference=flow(changed_x,t,positive['context'],positive['projected'])
            assert not torch.equal(changed_reference,reference['positive'])
        model=FrozenConditionFlow(flow,positive,negative) if args.mode=='frozen' else flow
        assert identities=={id(value) for value in model.parameters()}
        report['parameter_identities_preserved']=True
        with torch.no_grad():
            for name,branch in [('positive',positive),('negative',negative)]:
                output=(model(x,t,torch.tensor([name=='positive'])) if args.mode=='frozen'
                        else model(x,t,branch['context'],branch['projected']))
                torch.testing.assert_close(output,reference[name],rtol=0,atol=0)
        report['cpu_both_branches_bitwise_original']=True
        phase('waiting_for_device')
        lock=os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX)
        device=torch_xla.device()
        model.to(device)
        resident={name:((x.to(device),t.to(device),torch.tensor([name=='positive'],device=device))
                  if args.mode=='frozen' else
                  (x.to(device),t.to(device),branch['context'].to(device),branch['projected'].to(device)))
                  for name,branch in [('positive',positive),('negative',negative)]}
        torch_xla.set_custom_compile_options({'optimization_level':0,'math_fidelity':'hifi4','dry_run':'false',
            'export_path':str(args.output_dir/'native-ir')})
        compiled=torch.compile(model,backend='tt',fullgraph=True)
        metrics.clear_all()
        phase('native_both_branches')
        outputs={};report['branches']={}
        with torch.no_grad():
            for name in ('positive','negative'):
                outputs[name]=compiled(*resident[name]).cpu()
            before=dict(torch._dynamo.utils.counters['stats'])
            for name in ('positive','negative','positive'):
                start=time.perf_counter();value=compiled(*resident[name]).cpu();seconds=time.perf_counter()-start
                torch.testing.assert_close(value,outputs[name],rtol=0,atol=0)
                report['branches'][name]={'native_vs_original_cpu':compare_tensors(value,reference[name]),
                    'resident_call_seconds':seconds,'repeated_bitwise':True}
            changed_args=(changed_x.to(device),*resident['positive'][1:])
            changed_output=compiled(*changed_args).cpu()
            report['changed_noise_vs_original_cpu']=compare_tensors(changed_output,changed_reference)
            outputs['changed_noise']=changed_output
        report['recompiled_after_branch_warmup']=before!=dict(torch._dynamo.utils.counters['stats'])
        report['executed_fallback_ops']=list(metrics.executed_fallback_ops())
        report['aten_counters']={name:metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
        ir=next((args.output_dir/'native-ir/irs').glob('ttnn_runtime_[0-9]*.mlir'))
        report['native_functions']=summarize_functions(ir.read_text())
        report['native_ir_sha256']=sha256(ir)
        torch.save(outputs,args.output_dir/'outputs.pt')
        report['outputs_sha256']=sha256(args.output_dir/'outputs.pt')
        (args.output_dir/'xla_metrics.txt').write_text(metrics.metrics_report())
        if report['executed_fallback_ops'] or report['aten_counters'] or report['recompiled_after_branch_warmup']:
            raise ValueError('Frozen-conditioning smoke has fallback or recompile')
        report['all_branches_pass_smoke']=(all(row['native_vs_original_cpu']['passed'] for row in report['branches'].values())
                                           and report['changed_noise_vs_original_cpu']['passed'])
        report['status']='completed_tiny_consteval_control' if report['all_branches_pass_smoke'] else 'numerical_mismatch'
        phase('complete')
        return 0 if report['all_branches_pass_smoke'] else 1
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        phase(report.get('phase','initialization'))
        raise


if __name__=='__main__':raise SystemExit(main())
