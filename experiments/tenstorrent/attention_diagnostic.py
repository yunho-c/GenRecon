"""Compare attention arithmetic on identical pretrained first-block Q/K/V.

Isolated self/cross attention, not a whole-flow latency or quality claim. The
FP32 candidate promotes SDPA inputs/intermediates and restores the input dtype;
weights, masks and attention scale are unchanged. Native CPU BF16 and a CPU
precision diagnostic are both retained as references.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


class AttentionMath(nn.Module):
    def __init__(self, fp32=False, postscale=False):
        super().__init__()
        self.fp32 = fp32
        self.postscale = postscale
        if postscale and not fp32:
            raise ValueError('Post-matmul scaling experiment requires FP32 intermediates')

    def forward(self, q, k, v):
        from genrecon.modules.attention.full_attn import scaled_dot_product_attention
        if self.postscale:
            from genrecon.modules.attention.full_attn import _naive_sdpa
            return _naive_sdpa(q.float(), k.float(), v.float()).to(q.dtype)
        if self.fp32:
            return scaled_dot_product_attention(q.float(), k.float(), v.float()).to(q.dtype)
        return scaled_dot_product_attention(q, k, v)


def extract_qkv(attention, x, *, phases=None, context=None):
    """Use original learned projections/LoRA, normalization and complex RoPE."""
    from genrecon.modules.attention import RotaryPositionEmbedder
    batch, tokens, _ = x.shape
    if attention._type == 'self':
        q, k, v = attention.to_qkv(x).reshape(batch, tokens, 3, attention.num_heads, -1).unbind(2)
    else:
        q = attention.to_q(x).reshape(batch, tokens, attention.num_heads, -1)
        k, v = attention.to_kv(context).reshape(batch, context.shape[1], 2, attention.num_heads, -1).unbind(2)
    if attention.qk_rms_norm:
        q, k = attention.q_rms_norm(q), attention.k_rms_norm(k)
    if attention.use_rope:
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
    return q, k, v


def first_block_fixtures(flow, inputs):
    from experiments.tenstorrent.precision_reference import PipelinePrecisionMode
    x, t, context, projected = inputs
    block = flow.blocks[0]
    with torch.no_grad(), PipelinePrecisionMode():
        h = flow.input_layer(x).unsqueeze(0).to(flow.dtype)
        h = h + flow.proj_linears[0](projected.unsqueeze(0).to(flow.dtype))
        mod = flow.adaLN_modulation(flow.t_embedder(t)).to(flow.dtype)
        shift, scale, gate, *_ = (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        msa_input = block.norm1(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        qkv_self = extract_qkv(block.self_attn, msa_input, phases=flow.phases)
        self_output = block.self_attn(msa_input, phases=flow.phases)
        reconstructed = block.self_attn.to_out(AttentionMath()(*qkv_self).reshape_as(msa_input))
        torch.testing.assert_close(reconstructed, self_output, rtol=0, atol=0)
        cross_input = block.norm2(h + self_output * gate.unsqueeze(1))
        context = context.to(flow.dtype)
        qkv_cross = extract_qkv(block.cross_attn, cross_input, context=context)
        cross_output = block.cross_attn(cross_input, context)
        reconstructed = block.cross_attn.to_out(AttentionMath()(*qkv_cross).reshape_as(cross_input))
        torch.testing.assert_close(reconstructed, cross_output, rtol=0, atol=0)
    return {name: tuple(value.detach().contiguous().clone() for value in values)
            for name, values in (('self', qkv_self), ('cross', qkv_cross))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, default=Path.home() / 'models/genrecon')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--export-ir', action='store_true')
    args = p.parse_args()
    if min(args.threads, args.runs) < 1:
        p.error('threads and runs must be positive')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for key in ('TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'LD_LIBRARY_PATH', 'PYTHONPATH'):
        os.environ.pop(key, None)
    os.environ.update(PJRT_DEVICE='TT', ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    report = {'status': 'running', 'scope': __doc__, 'seed': args.seed,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}

    def phase(name):
        report['phase'] = name
        (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, flush=True)

    phase('imports')
    try:
        import tt_torch
        import torch_xla
        import torch_xla.debug.metrics as metrics
        import torch_xla.runtime as xr
        from experiments.tenstorrent.flow_probes import build_flow_case
        from experiments.tenstorrent.precision_reference import PipelinePrecisionMode
        from experiments.tenstorrent.validation import compare_tensors
        from experiments.tenstorrent.experiment_identity import sha256
        from experiments.tenstorrent.trajectory import tensor_digest
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        report['versions'] = {key: importlib.metadata.version(key)
                              for key in ('torch', 'torch-xla', 'tt-forge', 'pjrt-plugin-tt')}
        report['source_sha256'] = {str(path.relative_to(ROOT)): sha256(path)
            for folder in (ROOT / 'genrecon', ROOT / 'experiments/tenstorrent')
            for path in sorted(folder.rglob('*.py'))}
        phase('pretrained_fixtures')
        flow, inputs, report['model'] = build_flow_case('shape_flow_tensor', args.checkpoints)
        actual_sha = sha256(args.checkpoints / 'shape_slat.pt')
        if actual_sha != report['model']['checkpoint']['sha256']:
            raise ValueError('Physical checkpoint hash mismatch')
        report['verified_checkpoint_sha256'] = actual_sha
        fixtures = first_block_fixtures(flow, inputs)
        report['cpu_attention_reconstruction_exact'] = True
        report['fixture_digests'] = {name: [tensor_digest(value) for value in qkv] for name, qkv in fixtures.items()}
        torch.save(fixtures, args.output_dir / 'fixtures.pt')
        del flow, inputs
        phase('cpu_references')
        references, report['cases'] = {}, {}
        variants = (('stock', False, False), ('fp32_sdpa', True, False), ('fp32_postscale', True, True))
        for name, qkv in fixtures.items():
            with torch.no_grad():
                native = AttentionMath()(*qkv)
                with PipelinePrecisionMode():
                    diagnostic = AttentionMath()(*qkv)
                fp32 = AttentionMath(True)(*qkv)
                postscale = AttentionMath(True, True)(*qkv)
            report['cases'][name] = {'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in qkv],
                                    'native_cpu_vs_precision_diagnostic': compare_tensors(native, diagnostic),
                                    'fp32_cpu_vs_precision_diagnostic': compare_tensors(fp32, diagnostic),
                                    'postscale_cpu_vs_precision_diagnostic': compare_tensors(postscale, diagnostic)}
            references[name] = {'native': native, 'diagnostic': diagnostic, 'fp32': fp32}
        torch.save(references, args.output_dir / 'references.pt')
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        device = torch_xla.device()
        assert xr.device_type() == 'TT'
        torch_xla.set_custom_compile_options({'optimization_level': 0})
        os.chdir(args.output_dir)
        outputs = {}
        for name, qkv in fixtures.items():
            device_inputs = tuple(value.to(device) for value in qkv)
            for variant, promote, postscale in variants:
                label = name + '_' + variant
                phase(label)
                torch_xla.set_custom_compile_options({'optimization_level': 0})
                module = AttentionMath(promote, postscale).eval()
                compiled = torch.compile(module, backend='tt', fullgraph=True)
                metrics.clear_all()
                with torch.no_grad():
                    start = time.perf_counter()
                    actual = compiled(*device_inputs).cpu()
                    first_seconds = time.perf_counter() - start
                    compiled(*device_inputs).cpu()
                    warm = []
                    for _ in range(args.runs):
                        start = time.perf_counter()
                        repeated = compiled(*device_inputs).cpu()
                        warm.append(time.perf_counter() - start)
                        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
                result = {'first_seconds': first_seconds, 'warm_seconds': warm,
                          'timing_scope': 'Isolated attention; preuploaded Q/K/V, output D2H and synchronization included',
                          'vs_native_cpu_bf16': compare_tensors(actual, references[name]['native']),
                          'vs_cpu_precision_diagnostic': compare_tensors(actual, references[name]['diagnostic']),
                          'vs_fp32_sdpa_cpu': compare_tensors(actual, references[name]['fp32']),
                          'fallback_ops': list(metrics.executed_fallback_ops()),
                          'aten_counters': {key: metrics.counter_value(key) for key in metrics.counter_names() if key.startswith('aten::')}}
                report['cases'][name][variant] = result
                (args.output_dir / (label + '-metrics.txt')).write_text(metrics.metrics_report())
                if result['fallback_ops'] or result['aten_counters']:
                    raise ValueError('Unexpected CPU fallback in attention experiment')
                outputs[label] = actual
        torch.save(outputs, args.output_dir / 'outputs.pt')
        # codegen_py sets process-global custom compile options and mutates
        # module.compile(). It returns placeholder outputs, not inference.
        # Never measure after export or reuse an exported module for inference.
        if args.export_ir:
            report['codegen_ordering'] = 'All native measurements finish before any exports; fresh modules are exported and never measured.'
            report['codegen_helper_sha256'] = sha256(Path(tt_torch.codegen_py.__code__.co_filename))
            for name, qkv in fixtures.items():
                for variant, promote, postscale in variants:
                    label = name + '_' + variant
                    phase('export_ir_' + label)
                    with torch.no_grad():
                        tt_torch.codegen_py(AttentionMath(promote, postscale).eval(), *qkv, compiler_options={'optimization_level': 0},
                                            export_path=str(args.output_dir / (label + '-codegen')))
        report['artifacts_sha256'] = {path.name: sha256(path) for path in args.output_dir.glob('*.pt')}
        report['status'] = 'completed_isolated_attention_diagnostic'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise


if __name__ == '__main__':
    main()
