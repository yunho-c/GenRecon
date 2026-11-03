"""Guard isolated learned attention against accepted full-model FP32 boundaries.

Only accepted ordinary/observer comparisons support component attribution.
QKV-only alternatives consume identical observed native Q/K/V values. Timings
are isolated resident-input calls with output transfer, not whole-flow latency.
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-directory',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,default=Path.home()/'models/genrecon')
    parser.add_argument('--runs',type=int,default=3)
    args=parser.parse_args()
    if args.runs<1:parser.error('runs must be positive')
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','TT_METAL_HOME','TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key,None)
    os.environ.update(PJRT_DEVICE='TT',ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    report={'status':'running','scope':__doc__,'cases':{},
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
        from experiments.tenstorrent.flow_probes import build_flow_case
        from experiments.tenstorrent.inference_rope import prepare_real_rope,RealPairRoPE
        from experiments.tenstorrent.inference_attention import prepare_attention
        from experiments.tenstorrent.inference_precision import prepare_float32_device
        from experiments.tenstorrent.attention_observation import (ATTENTION_STATE_NAMES,
            AttentionObservation,clone_projection_parameters,cpu_attention_references,QKVAttention)
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        origin=json.loads((args.capture_directory/'result.json').read_text())
        if not (origin['status']=='completed_numerical_diagnostic' and origin['first_block'] and
                origin['captured_final_bitwise_matches_ordinary'] and origin['repeated_all_states_bitwise'] and
                not origin['executed_fallback_ops'] and not origin['aten_counters']):
            raise ValueError('Source full-model observation was not accepted')
        source_proof={name:sha256(ROOT/name)==value for name,value in origin['source_sha256'].items()
                      if name.startswith('genrecon/') or Path(name).name in
                      {'flow_probes.py','inference_rope.py','inference_attention.py','inference_precision.py'}}
        if not all(source_proof.values()):raise ValueError('Source model/helper files changed')
        path=args.capture_directory/'captured-tt.pt'
        if sha256(path)!=origin['artifacts_sha256'][path.name]:raise ValueError('Source native tensors changed')
        source=torch.load(path,weights_only=True,map_location='cpu')
        torch.set_num_threads(origin['threads'])
        torch.manual_seed(origin['seed'])
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['source_sha256']={str(path.relative_to(ROOT)):sha256(path)
            for folder in (ROOT/'genrecon',ROOT/'experiments/tenstorrent') for path in sorted(folder.rglob('*.py'))}
        report['source_capture_sha256']=sha256(args.capture_directory/'result.json')
        report['source_model_proof']=source_proof
        phase('build_verified_fp32_model')
        flow,inputs,report['model']=build_flow_case('shape_flow_tensor',args.checkpoints)
        report['verified_checkpoint_sha256']=sha256(args.checkpoints/'shape_slat.pt')
        if report['verified_checkpoint_sha256']!=origin['verified_checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if [tensor_digest(value) for value in inputs]!=origin['input_digests']:raise ValueError('Original fixture changed')
        prepare_real_rope(flow,head_first=True,flatten=True)
        prepare_attention(flow,'fp32_reciprocal')
        prepare_float32_device(flow)
        cpu_rope=RealPairRoPE(flow.phases.clone(),head_first=True,flatten=True)
        fixtures={}
        phase('cpu_attention_controls')
        for kind in ('self','cross'):
            prepared=getattr(flow.blocks[0],kind+'_attn')
            observation=AttentionObservation(prepared).eval()
            values=source['first_'+kind+'_input'].clone()
            context=inputs[2].clone()
            parameters,scales=clone_projection_parameters(prepared)
            with torch.no_grad():
                cpu_states=dict(zip(ATTENTION_STATE_NAMES,observation(values,context),strict=True))
                torch.testing.assert_close(cpu_states['output'],prepared(values,context),rtol=0,atol=0)
                refs=cpu_attention_references(prepared,parameters,scales,values,context,cpu_states,cpu_rope)
                for name,value in refs.items():torch.testing.assert_close(value,cpu_states[name],rtol=0,atol=0)
            fixtures[kind]=(prepared,observation,values,context,parameters,scales,cpu_states)
            report['cases'][kind]={'cpu_observer_bitwise_ordinary':True,'cpu_teacher_boundaries_bitwise':True,
                'input_digest':tensor_digest(values),'context_digest':tensor_digest(context),
                'full_model_boundary_vs_cpu':compare_tensors(source['first_'+kind+'_output'],cpu_states['output'])}
        phase('waiting_for_device')
        lock=os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX)
        device=torch_xla.device()
        outputs={}
        def compile_module(module,label):
            torch.compiler.reset()
            torch_xla.set_custom_compile_options({'optimization_level':0,'math_fidelity':'hifi4','dry_run':'false',
                'export_path':str(args.output_dir/(label+'-native-ir'))})
            return torch.compile(module,backend='tt',fullgraph=True)
        for kind,(prepared,observation,values,context,parameters,scales,cpu_states) in fixtures.items():
            row=report['cases'][kind]
            prepared.to(device)
            resident=values.to(device),context.to(device)
            metrics.clear_all()
            phase(kind+'_ordinary_attention')
            ordinary=compile_module(prepared,kind+'_ordinary')
            with torch.no_grad():plain=ordinary(*resident).cpu()
            row['isolated_vs_full_model']=compare_tensors(plain,source['first_'+kind+'_output'])
            row['isolated_bitwise_full_model']=torch.equal(plain,source['first_'+kind+'_output'])
            phase(kind+'_observed_attention')
            observer=compile_module(observation,kind+'_observed')
            with torch.no_grad():
                actual=dict(zip(ATTENTION_STATE_NAMES,[value.cpu() for value in observer(*resident)],strict=True))
                repeated=dict(zip(ATTENTION_STATE_NAMES,[value.cpu() for value in observer(*resident)],strict=True))
            row['observer_vs_ordinary']=compare_tensors(actual['output'],plain)
            row['observer_bitwise_ordinary']=torch.equal(actual['output'],plain)
            row['all_states_repeat_bitwise']=all(torch.equal(value,repeated[name]) for name,value in actual.items())
            row['executed_fallback_ops']=list(metrics.executed_fallback_ops())
            row['aten_counters']={name:metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
            row['attribution_accepted']=(row['isolated_bitwise_full_model'] and row['observer_bitwise_ordinary'] and
                row['all_states_repeat_bitwise'] and not row['executed_fallback_ops'] and not row['aten_counters'])
            outputs[kind]={'ordinary':plain,'observed':actual}
            torch.save(outputs[kind],args.output_dir/(kind+'-attention-states.pt'))
            if row['attribution_accepted']:
                with torch.no_grad():refs=cpu_attention_references(prepared,parameters,scales,values,context,actual,cpu_rope)
                row['local_errors']={name:compare_tensors(actual[name],ref) for name,ref in refs.items()}
                phase(kind+'_qkv_only_controls')
                qkv=tuple(actual[name] for name in ('q_head','k_head','v_head'))
                resident_qkv=tuple(value.to(device) for value in qkv)
                with torch.no_grad():reference=QKVAttention()(*qkv)
                row['qkv_cases']={}
                for mode,outer in [('native',False),('outer_sum',True)]:
                    phase(kind+'_qkv_'+mode)
                    module=QKVAttention(outer).eval()
                    with torch.no_grad():cpu_formula=module(*qkv)
                    cpu_check=compare_tensors(cpu_formula,reference,max_relative_l2=1e-3,min_pcc=.999999)
                    if not cpu_check['passed']:raise ValueError('CPU QKV alternative changes arithmetic beyond tolerance')
                    compiled=compile_module(module,kind+'_qkv_'+mode)
                    metrics.clear_all()
                    with torch.no_grad():
                        result=compiled(*resident_qkv).cpu()
                        times=[]
                        for _ in range(args.runs):
                            start=time.perf_counter();repeat=compiled(*resident_qkv).cpu();times.append(time.perf_counter()-start)
                            torch.testing.assert_close(result,repeat,rtol=0,atol=0)
                    check={'cpu_formula_vs_reference':cpu_check,'native_vs_cpu':compare_tensors(result,reference),
                        'native_vs_cpu_formula':compare_tensors(result,cpu_formula),'warm_seconds':times,
                        'native_vs_captured_attended':compare_tensors(result,actual['attended']),
                        'native_bitwise_captured_attended':torch.equal(result,actual['attended']),
                        'executed_fallback_ops':list(metrics.executed_fallback_ops()),
                        'aten_counters':{name:metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}}
                    if check['executed_fallback_ops'] or check['aten_counters']:raise ValueError('QKV alternative used fallback')
                    row['qkv_cases'][mode]=check
                    outputs[kind]['qkv_'+mode]=result
                    (args.output_dir/(kind+'-qkv-'+mode+'-metrics.txt')).write_text(metrics.metrics_report())
                row['qkv_source_attribution_accepted']=row['qkv_cases']['native']['native_bitwise_captured_attended']
            torch.save(outputs[kind],args.output_dir/(kind+'-attention-states.pt'))
        report['artifacts_sha256']={path.name:sha256(path) for path in args.output_dir.glob('*.pt')}
        report['status']='completed_learned_attention_diagnostic'
        phase('complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        phase(report.get('phase','initialization'))
        raise


if __name__=='__main__':main()
