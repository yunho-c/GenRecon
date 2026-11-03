"""Isolate the rejected head-cache boundary on pretrained first-block attention.

All stage graphs share original learned objects/LoRA. The query is teacher-forced
from the identical prepared CPU first block; both original condition branches
are exercised. This diagnostic is not complete-sampler quality or speed evidence.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))


def stages(attention,key_transposed=False):
    import torch
    from torch import nn
    from experiments.tenstorrent.inference_attention import reciprocal_attention
    from experiments.tenstorrent.conditioning_cache import CachedHeadCrossAttention

    class PackKV(nn.Module):
        def __init__(self):
            super().__init__();self.attention=attention.attention;self.eval()

        def forward(self,raw):
            attn=self.attention
            key,value=raw.reshape(1,raw.shape[1],2,attn.num_heads,-1).unbind(2)
            if attn.qk_rms_norm:key=attn.k_rms_norm(key)
            order=(0,2,3,1) if key_transposed else (0,2,1,3)
            return key.permute(order),value.permute(0,2,1,3)

    class RawCross(nn.Module):
        def __init__(self):
            super().__init__();self.attention=attention.attention;self.eval()

        def forward(self,x,raw):
            attn=self.attention
            batch,tokens,channels=x.shape
            query=attn.to_q(x).reshape(batch,tokens,attn.num_heads,-1)
            key,value=raw.reshape(batch,raw.shape[1],2,attn.num_heads,-1).unbind(2)
            if attn.qk_rms_norm:query,key=attn.q_rms_norm(query),attn.k_rms_norm(key)
            h=reciprocal_attention(query,key,value,'fp32_reciprocal')
            return attn.to_out(h.reshape(batch,tokens,channels))

    class FusedPackCross(nn.Module):
        def __init__(self,pack,head):
            super().__init__();self.pack=pack;self.head=head;self.eval()

        def forward(self,x,raw):
            return self.head(x,self.pack(raw))

    pack=PackKV();raw=RawCross();head=CachedHeadCrossAttention(attention,key_transposed)
    return pack,raw,head,FusedPackCross(pack,head)


def main():
    from experiments.tenstorrent.native_environment import reexec_clean_native_process
    reexec_clean_native_process()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-dir',type=Path,default=Path.home()/'models/genrecon')
    parser.add_argument('--cache-layout',choices=('heads','heads_t'),default='heads')
    args=parser.parse_args();args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ.update(PJRT_DEVICE='TT',ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',SPARSE_CONV_BACKEND='none')
    from experiments.tenstorrent.experiment_identity import sha256
    report={'status':'running','scope':__doc__,'seed':0,'script_sha256':sha256(Path(__file__)),
            'cache_layout':args.cache_layout,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def phase(name):
        report['phase']=name;(args.output_dir/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(name,flush=True)
    phase('imports')
    try:
        import torch
        from experiments.tenstorrent.probes import build_case
        from experiments.tenstorrent.inference_rope import prepare_real_rope
        from experiments.tenstorrent.inference_attention import prepare_attention
        from experiments.tenstorrent.inference_precision import prepare_float32_device
        from experiments.tenstorrent.conditioning_cache import ConditioningPrefill,prepare_conditioning_cache
        from experiments.tenstorrent.flow_observation import ShapeFlowObservation
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(16);torch.manual_seed(0)
        phase('build_pretrained_cpu_stages')
        model,inputs,metadata=build_case('shape_flow_tensor',args.checkpoint_dir)
        model.eval();report.update(metadata)
        if sha256(args.checkpoint_dir/'shape_slat.pt')!=report['checkpoint']['sha256']:
            raise ValueError('Physical checkpoint changed')
        prepare_real_rope(model,head_first=True,flatten=True);prepare_attention(model,'fp32_reciprocal');prepare_float32_device(model)
        with torch.no_grad():
            observation=ShapeFlowObservation(model,first_block=True).eval()
            values=dict(zip(observation.names,observation(*inputs),strict=True))
            query=values['first_cross_input'].clone()
            del values,observation
        original_cross=model.blocks[0].cross_attn
        headed_prefill,body=prepare_conditioning_cache(model,args.cache_layout)
        raw_prefill=ConditioningPrefill(model,'raw')
        pack,raw_cross,head_cross,fused=stages(original_cross,args.cache_layout=='heads_t')
        branches={'positive':(inputs[2],inputs[3]),
                  'negative':(torch.zeros_like(inputs[2]),torch.zeros_like(inputs[3]))}
        cpu={}
        with torch.no_grad():
            for name,(context,projected) in branches.items():
                raw=raw_prefill(context,projected);heads=headed_prefill(context,projected)
                expected=original_cross(query,context)
                for candidate in (raw_cross(query,raw[1]),head_cross(query,(heads[1],heads[2])),fused(query,raw[1])):
                    torch.testing.assert_close(candidate,expected,rtol=0,atol=0)
                cpu[name]={'raw':tuple(v.clone() for v in raw[:2]),'heads':tuple(v.clone() for v in heads[:3]),
                           'expected':expected.clone()}
                del raw,heads
        report['cpu_all_paths_bitwise_original_cross']=True
        report['query_digest']=tensor_digest(query)
        report['conditioning_digests']={name:[tensor_digest(v) for v in pair] for name,pair in branches.items()}
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        report['versions']={name:importlib.metadata.version(name) for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['helper_sha256']={name:sha256(Path(__file__).with_name(name)) for name in
            ('conditioning_cache.py','inference_attention.py','flow_observation.py','flow_probes.py')}
        phase('waiting_for_device')
        lock=os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX)
        device=torch_xla.device()
        for module in (raw_prefill,headed_prefill,pack,raw_cross,head_cross,fused):module.to(device)
        resident={name:tuple(v.to(device) for v in pair) for name,pair in branches.items()}
        q=query.to(device);torch_xla.sync(wait=True);torch.compiler.reset();metrics.clear_all()
        modules={'raw_prefill':raw_prefill,'heads_prefill':headed_prefill,'pack':pack,
                 'raw_cross':raw_cross,'heads_cross':head_cross,'fused_cross':fused}
        compiled={}
        def graph(name):
            if name not in compiled:
                torch_xla.set_custom_compile_options({'optimization_level':0,'math_fidelity':'hifi4','dry_run':'false',
                    'export_path':str(args.output_dir/(name+'-native-ir'))})
                compiled[name]=torch.compile(modules[name],backend='tt',fullgraph=True)
            return compiled[name]
        outputs={};report['branches']={}
        def comparison(a,b):
            return {'bitwise':torch.equal(a,b),'check':compare_tensors(a,b)}
        with torch.no_grad():
            for name in branches:
                phase(name+'_native_stages')
                raw=tuple(graph('raw_prefill')(*resident[name]));torch_xla.sync(wait=True)
                heads=tuple(graph('heads_prefill')(*resident[name]));torch_xla.sync(wait=True)
                packed=tuple(graph('pack')(raw[1]));torch_xla.sync(wait=True)
                native={}
                for label,args_for_graph in [('raw_cross',(q,raw[1])),('heads_cross',(q,(heads[1],heads[2]))),
                                             ('split_pack_cross',(q,packed)),('fused_cross',(q,raw[1]))]:
                    target='heads_cross' if label=='split_pack_cross' else label
                    value=graph(target)(*args_for_graph).cpu().float()
                    repeated=graph(target)(*args_for_graph).cpu().float()
                    torch.testing.assert_close(value,repeated,rtol=0,atol=0)
                    native[label]=value
                native.update(raw=tuple(v.cpu().float() for v in raw[:2]),
                    heads=tuple(v.cpu().float() for v in heads[:3]),
                    pack_after_raw=tuple(v.cpu().float() for v in packed))
                report['branches'][name]={
                    'raw_kv_vs_cpu':comparison(native['raw'][1],cpu[name]['raw'][1]),
                    'heads_key_vs_cpu':comparison(native['heads'][1],cpu[name]['heads'][1]),
                    'heads_value_vs_cpu':comparison(native['heads'][2],cpu[name]['heads'][2]),
                    'heads_key_vs_pack_after_raw':comparison(native['heads'][1],native['pack_after_raw'][0]),
                    'heads_value_vs_pack_after_raw':comparison(native['heads'][2],native['pack_after_raw'][1]),
                    'fused_vs_raw_cross':comparison(native['fused_cross'],native['raw_cross']),
                    'split_vs_raw_cross':comparison(native['split_pack_cross'],native['raw_cross']),
                    'heads_prefill_vs_raw_cross':comparison(native['heads_cross'],native['raw_cross']),
                    'heads_prefill_vs_split_cross':comparison(native['heads_cross'],native['split_pack_cross']),
                    'raw_cross_vs_original_cpu':comparison(native['raw_cross'],cpu[name]['expected']),
                    'all_cross_repeats_bitwise':True}
                outputs[name]={'cpu':cpu[name],'native':native}
                torch.save(outputs,args.output_dir/'outputs.pt');phase(name+'_recorded')
                del raw,heads,packed,native
        report['executed_fallback_ops']=list(metrics.executed_fallback_ops())
        report['aten_counters']={n:metrics.counter_value(n) for n in metrics.counter_names() if n.startswith('aten::')}
        (args.output_dir/'xla_metrics.txt').write_text(metrics.metrics_report())
        report['outputs_sha256']=sha256(args.output_dir/'outputs.pt')
        if report['executed_fallback_ops'] or report['aten_counters']:
            raise ValueError('Fallback invalidates native boundary attribution')
        report['status']='completed_pretrained_boundary_diagnostic';phase('complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());phase(report.get('phase','imports'));raise


if __name__=='__main__':main()
