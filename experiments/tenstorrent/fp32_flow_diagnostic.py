"""Observe the full prepared FP32 shape flow and its first block.

Attribution is accepted only when captured final output matches an ordinary
native forward bitwise, all captured outputs repeat exactly, and no fallback
occurs. This is a numerical diagnostic, not sampler quality or latency evidence.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path.home()/'models/genrecon')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--first-block', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa',
                      SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status':'running', 'scope':__doc__, 'seed':args.seed,
              'threads':args.threads, 'first_block':args.first_block,
              'recipe':{'rope_mode':'real_pairs_flat_head_first',
                        'attention_mode':'fp32_reciprocal', 'device_precision':'float32',
                        'optimization_level':0, 'math_fidelity':'hifi4'},
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,flush=True)

    phase('imports')
    try:
        import torch
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        from experiments.tenstorrent.flow_probes import build_flow_case
        from experiments.tenstorrent.flow_observation import ShapeFlowObservation
        from experiments.tenstorrent.reference import float32_reference
        from experiments.tenstorrent.inference_rope import prepare_real_rope
        from experiments.tenstorrent.inference_attention import prepare_attention
        from experiments.tenstorrent.inference_precision import prepare_float32_device
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        from experiments.tenstorrent.validation import compare_tensors
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        report['versions'] = {name:importlib.metadata.version(name)
                              for name in ('torch','torch-xla','tt-forge','pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)):sha256(path)
            for folder in (ROOT/'genrecon',ROOT/'experiments/tenstorrent')
            for path in sorted(folder.rglob('*.py'))}
        phase('build')
        flow, inputs, report['model'] = build_flow_case('shape_flow_tensor',args.checkpoints)
        report['verified_checkpoint_sha256'] = sha256(args.checkpoints/'shape_slat.pt')
        if report['verified_checkpoint_sha256'] != report['model']['checkpoint']['sha256']:
            raise ValueError('Physical checkpoint changed')
        report['input_digests'] = [tensor_digest(value) for value in inputs]
        observation = ShapeFlowObservation(flow,first_block=args.first_block).eval()
        names = observation.names
        report['state_names'] = names
        phase('stock_cpu_reference')
        with torch.no_grad(),float32_reference(flow):
            stock = tuple(value.clone() for value in observation(*inputs))
            torch.testing.assert_close(stock[0],flow(*inputs),rtol=0,atol=0)
        report['stock_cpu_capture_final_bitwise'] = True
        phase('prepare_fp32_model')
        report['prepared_rope_blocks'] = prepare_real_rope(flow,head_first=True,flatten=True)
        report['prepared_attention_modules'] = prepare_attention(flow,'fp32_reciprocal')
        report['precision_preparation'] = prepare_float32_device(flow)
        inputs = tuple(value.float() if value.is_floating_point() else value for value in inputs)
        with torch.no_grad():
            prepared = tuple(value.clone() for value in observation(*inputs))
            torch.testing.assert_close(prepared[0],flow(*inputs),rtol=0,atol=0)
        report['prepared_cpu_capture_final_bitwise'] = True
        for name,values in (('cpu-stock.pt',stock),('cpu-prepared.pt',prepared)):
            torch.save(dict(zip(names,values,strict=True)),args.output_dir/name)
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX)
        device = torch_xla.device()
        flow.to(device)
        device_inputs = tuple(value.to(device) for value in inputs)
        metrics.clear_all()
        options = {'optimization_level':0,'math_fidelity':'hifi4','dry_run':'false'}
        phase('ordinary_compiled_forward')
        torch_xla.set_custom_compile_options({**options,'export_path':str(args.output_dir/'ordinary-native-ir')})
        ordinary = torch.compile(flow,backend='tt',fullgraph=True)
        with torch.no_grad():
            plain = ordinary(*device_inputs).cpu()
        torch.save(plain,args.output_dir/'ordinary-tt.pt')
        phase('captured_compiled_forward')
        torch_xla.set_custom_compile_options({**options,'export_path':str(args.output_dir/'captured-native-ir')})
        observer = torch.compile(observation,backend='tt',fullgraph=True)
        with torch.no_grad():
            actual = tuple(value.cpu() for value in observer(*device_inputs))
            repeated = tuple(value.cpu() for value in observer(*device_inputs))
        report['captured_final_vs_ordinary'] = compare_tensors(actual[0],plain)
        report['captured_final_bitwise_matches_ordinary'] = torch.equal(actual[0],plain)
        report['repeated_all_states_bitwise'] = all(torch.equal(a,b) for a,b in zip(actual,repeated,strict=True))
        report['states'] = {name:{'tt_vs_stock_cpu':compare_tensors(a,s),
                                 'tt_vs_prepared_cpu':compare_tensors(a,p),
                                 'prepared_cpu_vs_stock_cpu':compare_tensors(p,s)}
                            for name,a,s,p in zip(names,actual,stock,prepared,strict=True)}
        report['executed_fallback_ops'] = list(metrics.executed_fallback_ops())
        report['aten_counters'] = {name:metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
        (args.output_dir/'xla_metrics.txt').write_text(metrics.metrics_report())
        torch.save(dict(zip(names,actual,strict=True)),args.output_dir/'captured-tt.pt')
        report['artifacts_sha256'] = {path.name:sha256(path) for path in args.output_dir.glob('*.pt')}
        accepted = (report['captured_final_bitwise_matches_ordinary'] and
                    report['repeated_all_states_bitwise'] and not report['executed_fallback_ops'] and
                    not report['aten_counters'])
        report['status'] = 'completed_numerical_diagnostic' if accepted else 'rejected_observation_changed_execution'
        phase('complete')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        phase(report.get('phase','initialization'))
        raise


if __name__ == '__main__':
    main()
