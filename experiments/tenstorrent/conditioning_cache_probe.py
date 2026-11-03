"""Tiny two-graph conditioning cache control across scenes, branches and noise.

Nonzero LoRA and all original learned objects are retained. Prefill is dynamic
native work with explicit timing and synchronization; the denoiser receives one
selected resident cache tuple. This smoke does not establish pretrained speed.
"""
import argparse
import fcntl
import gc
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


def main():
    from experiments.tenstorrent.native_environment import reexec_clean_native_process
    reexec_clean_native_process()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-layout',choices=('raw','heads','heads_t'),default='raw')
    args=parser.parse_args()
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=False)
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','TT_METAL_HOME','TT_METAL_RUNTIME_ROOT'):os.environ.pop(key,None)
    os.environ.update(PJRT_DEVICE='TT',ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    report={'status':'running','scope':__doc__,'cache_layout':args.cache_layout,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def phase(name):
        report['phase']=name;(args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(name,flush=True)
    phase('imports')
    try:
        import torch
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from experiments.tenstorrent.test_frozen_conditioning import fixture
        from experiments.tenstorrent.conditioning_cache import prepare_conditioning_cache
        from experiments.tenstorrent.frozen_conditioning_probe import summarize_functions
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['source_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in
            [Path(__file__),Path(__file__).with_name('conditioning_cache.py'),Path(__file__).with_name('test_flow_observation.py'),
             Path(__file__).with_name('test_frozen_conditioning.py')]}
        flow,pos,neg=fixture();x,t=torch.randn(17,8),torch.tensor([.6])
        parameters=dict(flow.named_parameters());identities={id(value) for value in parameters.values()}
        report['original_parameter_digests']={name:tensor_digest(value) for name,value in parameters.items()}
        scenes=[(pos,neg),({key:value+.125 for key,value in pos.items()},neg)]
        report['conditioning_digests']=[{name:{key:tensor_digest(value) for key,value in branch.items()}
            for name,branch in zip(('positive','negative'),scene,strict=True)} for scene in scenes]
        cases=[('positive',0,x),('negative',1,x),('changed_noise',0,x+.125)]
        with torch.no_grad():
            reference={(scene,name):flow(values,t,branches[index]['context'],branches[index]['projected'])
                for scene,branches in enumerate(scenes) for name,index,values in cases}
        prefill,body=prepare_conditioning_cache(flow,args.cache_layout)
        assert identities=={id(value) for value in body.parameters()}
        report['parameter_identities_preserved']=True
        with torch.no_grad():
            cpu_caches={(scene,index):prefill(branch['context'],branch['projected'])
                for scene,branches in enumerate(scenes) for index,branch in enumerate(branches)}
            for scene in range(2):
                for name,index,values in cases:
                    torch.testing.assert_close(body(values,t,cpu_caches[scene,index]),reference[scene,name],rtol=0,atol=0)
        report['cpu_all_scenes_branches_noise_bitwise_original']=True
        phase('waiting_for_device')
        lock=os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX)
        device=torch_xla.device();body.to(device);prefill.to(device)
        resident={(scene,index):tuple(branch[key].to(device) for key in ('context','projected'))
            for scene,branches in enumerate(scenes) for index,branch in enumerate(branches)}
        def compile_module(module,name):
            torch_xla.set_custom_compile_options({'optimization_level':0,'math_fidelity':'hifi4','dry_run':'false',
                'export_path':str(args.output_dir/(name+'-native-ir'))})
            return torch.compile(module,backend='tt',fullgraph=True)
        torch.compiler.reset();metrics.clear_all();phase('compile_prefill')
        compiled_prefill=compile_module(prefill,'prefill')
        with torch.no_grad():warm=compiled_prefill(*resident[0,0])
        torch_xla.sync(wait=True);del warm;gc.collect()
        caches={};outputs={};report['scenes']={}
        compiled_body=compile_module(body,'body')
        for scene in range(2):
            phase('scene_'+str(scene)+'_prefill')
            start=time.perf_counter()
            with torch.no_grad():
                for index in range(2):caches[scene,index]=tuple(compiled_prefill(*resident[scene,index]))
            torch_xla.sync(wait=True)
            row={'prefill_pair_seconds':time.perf_counter()-start,'outputs':{}}
            if scene==1:before_scene=dict(torch._dynamo.utils.counters['stats'])
            phase('scene_'+str(scene)+'_cached_body')
            with torch.no_grad():
                for name,index,values in cases:
                    arguments=values.to(device),t.to(device),caches[scene,index]
                    value=compiled_body(*arguments).cpu()
                    start=time.perf_counter();repeat=compiled_body(*arguments).cpu();elapsed=time.perf_counter()-start
                    torch.testing.assert_close(value,repeat,rtol=0,atol=0)
                    outputs[scene,name]=value
                    row['outputs'][name]={'native_vs_original_cpu':compare_tensors(value,reference[scene,name]),
                        'resident_body_seconds':elapsed,'repeat_bitwise':True}
            if scene==0:after_first_scene=dict(torch._dynamo.utils.counters['stats'])
            else:report['recompiled_for_new_scene_or_noise']=after_first_scene!=dict(torch._dynamo.utils.counters['stats'])
            report['scenes'][str(scene)]=row
        with torch.no_grad():old_scene=compiled_body(x.to(device),t.to(device),caches[0,0]).cpu()
        report['old_scene_cache_still_bitwise']=torch.equal(old_scene,outputs[0,'positive'])
        report['executed_fallback_ops']=list(metrics.executed_fallback_ops())
        report['aten_counters']={name:metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
        report['native_functions']={}
        for name in ('prefill','body'):
            path=next((args.output_dir/(name+'-native-ir')/'irs').glob('ttnn_runtime_[0-9]*.mlir'))
            report['native_functions'][name]=summarize_functions(path.read_text())
        torch.save({'outputs':outputs,'caches':{key:tuple(value.cpu() for value in cache) for key,cache in caches.items()}},args.output_dir/'outputs.pt')
        report['outputs_sha256']=sha256(args.output_dir/'outputs.pt');(args.output_dir/'xla_metrics.txt').write_text(metrics.metrics_report())
        report['all_outputs_pass_smoke']=all(result['native_vs_original_cpu']['passed'] for row in report['scenes'].values() for result in row['outputs'].values())
        if report['executed_fallback_ops'] or report['aten_counters'] or report['recompiled_for_new_scene_or_noise'] or not report['old_scene_cache_still_bitwise']:
            raise ValueError('Cache smoke has fallback, recompile or stale scene state')
        report['status']='completed_dynamic_prefill_cache_control' if report['all_outputs_pass_smoke'] else 'numerical_mismatch'
        phase('complete');return 0 if report['all_outputs_pass_smoke'] else 1
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());phase(report.get('phase','initialization'));raise


if __name__=='__main__':raise SystemExit(main())
