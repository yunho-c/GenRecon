"""Localize full shape-flow drift without changing acceptance gates.

Capturing intermediates can change compiler lowering. Attribution is accepted
only when the captured final output exactly matches an ordinary compiled
forward with the same model/inputs. No latency or reconstruction-quality claim.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, default=Path.home() / 'models/genrecon')
    p.add_argument('--rope-mode', choices=('stock', 'real_pairs'), default='real_pairs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('LD_LIBRARY_PATH', 'PYTHONPATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'rope_mode': args.rope_mode,
              'seed': args.seed, 'threads': args.threads,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, flush=True)

    phase('imports')
    try:
        import torch
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as xr
        from torch import nn
        from torch.nn import functional as F
        from experiments.tenstorrent.flow_probes import build_flow_case
        from experiments.tenstorrent.reference import float32_reference
        from experiments.tenstorrent.precision_reference import PipelinePrecisionMode
        from experiments.tenstorrent.validation import compare_tensors
        from experiments.tenstorrent.experiment_identity import sha256
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path)
            for folder in (ROOT / 'genrecon', ROOT / 'experiments/tenstorrent')
            for path in sorted(folder.rglob('*.py'))}
        phase('build')
        model, inputs, report['model'] = build_flow_case('shape_flow_tensor', args.checkpoints)
        model.eval()

        class Capture(nn.Module):
            def __init__(self, flow):
                super().__init__()
                self.flow = flow

            def forward(self, x, t, context, projected):
                flow = self.flow
                h = flow.input_layer(x).unsqueeze(0).to(flow.dtype)
                states = [h]
                mod = flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
                context = context.to(flow.dtype)
                projected = projected.unsqueeze(0).to(flow.dtype)
                for block, projection in zip(flow.blocks, flow.proj_linears):
                    h = h + projection(projected)
                    h = block(h, mod, context, flow.phases)
                    states.append(h)
                output = flow.out_layer(F.layer_norm(h.to(x.dtype), h.shape[-1:]))[0]
                return (output, *states)

        captured = Capture(model).eval()
        names = ['final', 'input_layer'] + [f'block_{i:02d}' for i in range(len(model.blocks))]
        phase('stock_fp32_references')
        with torch.no_grad(), float32_reference(model):
            stock = model(*inputs)
            fp32 = tuple(x.clone() for x in captured(*inputs))
            torch.testing.assert_close(fp32[0], stock, rtol=0, atol=0)
        report['cpu_capture_final_bitwise_matches_stock'] = True
        phase('stock_pipeline_precision_diagnostic')
        with torch.no_grad(), PipelinePrecisionMode() as precision:
            bf16 = tuple(x.clone() for x in captured(*inputs))
        report['pipeline_precision_operations'] = dict(precision.promoted_ops)
        if args.rope_mode == 'real_pairs':
            from experiments.tenstorrent.inference_rope import prepare_real_rope
            report['prepared_rope_blocks'] = prepare_real_rope(model)
            with torch.no_grad(), float32_reference(model):
                prepared = captured(*inputs)
            report['prepared_cpu_bitwise_by_state'] = {
                name: torch.equal(a, b) for name, a, b in zip(names, prepared, fp32, strict=True)}
            if not all(report['prepared_cpu_bitwise_by_state'].values()):
                raise ValueError('Prepared CPU intermediate differs from original')
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert xr.device_type() == 'TT'
        torch_xla.set_custom_compile_options({'optimization_level': 0})
        model.to(device)
        device_inputs = tuple(x.to(device) for x in inputs)
        metrics.clear_all()
        phase('ordinary_compiled_forward')
        ordinary = torch.compile(model, backend='tt', fullgraph=True)
        with torch.no_grad():
            plain = ordinary(*device_inputs).cpu()
        phase('captured_compiled_forward')
        observer = torch.compile(captured, backend='tt', fullgraph=True)
        with torch.no_grad():
            actual = tuple(x.cpu() for x in observer(*device_inputs))
            repeated = tuple(x.cpu() for x in observer(*device_inputs))
        report['captured_final_vs_ordinary'] = compare_tensors(actual[0], plain)
        report['captured_final_bitwise_matches_ordinary'] = torch.equal(actual[0], plain)
        report['repeated_all_states_bitwise'] = all(torch.equal(a, b) for a, b in zip(actual, repeated, strict=True))
        report['states'] = {name: {'tt_vs_stock_fp32': compare_tensors(a, f),
                                    'tt_vs_cpu_precision_diagnostic': compare_tensors(a, b),
                                    'cpu_precision_diagnostic_vs_stock_fp32': compare_tensors(b, f)}
                            for name, a, f, b in zip(names, actual, fp32, bf16, strict=True)}
        report['executed_fallback_ops'] = list(metrics.executed_fallback_ops())
        report['aten_counters'] = {name: metrics.counter_value(name) for name in metrics.counter_names() if name.startswith('aten::')}
        (args.output_dir / 'xla_metrics.txt').write_text(metrics.metrics_report())
        for name, value in (('cpu-fp32.pt', fp32), ('cpu-precision-diagnostic.pt', bf16),
                            ('ordinary-tt.pt', plain), ('captured-tt.pt', actual)):
            torch.save(value, args.output_dir / name)
        report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output_dir.glob('*.pt')}
        if (not report['captured_final_bitwise_matches_ordinary'] or
                not report['repeated_all_states_bitwise'] or report['executed_fallback_ops'] or report['aten_counters']):
            report['status'] = 'rejected_observation_changed_execution'
        else:
            report['status'] = 'completed_numerical_diagnostic'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
