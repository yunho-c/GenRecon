"""Original shape Euler/CFG sampler through a sealed generated TTNN graph.

Changing inputs are freshly uploaded on every call. Learned weights and the
compiler const-eval cache remain resident. Require a sealed canonical forward
and every state of the canonical native trajectory to match bitwise. Original
stock CPU trajectory gates are reported unchanged, including inherited failures.
"""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def match_activation_inputs(templates, inputs):
    """Map actual generated activation values to the four public inputs."""
    import torch
    mapping = []
    for template in templates:
        matches = [i for i, value in enumerate(inputs)
                   if value.shape == template.shape and torch.equal(value, template)]
        if len(matches) > 1:
            raise ValueError('Ambiguous generated activation')
        mapping.append(matches[0] if matches else None)
    if (len(mapping) != 5 or sorted(i for i in mapping if i is not None) != list(range(4)) or
            mapping.count(None) != 1 or templates[mapping.index(None)].shape != (128,)):
        raise ValueError('Expected four exact inputs and the fixed128 timestep frequencies')
    return mapping


def assert_bitwise_trajectory(actual, expected):
    import torch
    from experiments.tenstorrent.trajectory import compare_trajectories
    compare_trajectories(actual, expected)
    for key in ('samples', 'pred_x_t', 'pred_x_0'):
        av = actual[key] if isinstance(actual[key], list) else [actual[key]]
        ev = expected[key] if isinstance(expected[key], list) else [expected[key]]
        if len(av) != len(ev) or not all(torch.equal(a, e) for a, e in zip(av, ev)):
            raise ValueError('Generated sampler differs from native control at '+key)


def restore_native_time_frequencies(template, control_sha256):
    """Restore only the two fixed constants sealed by native control083/110."""
    import torch
    evidence = ROOT/'experiments/tenstorrent/results/autoresearch'
    previous = json.loads((evidence/'093-rejected-generated-sampler-boundary/sdk-execute.json').read_text())
    if control_sha256 != previous['sampler_native_control']['result_sha256']:
        raise ValueError('Native frequency repair requires the sealed original control110')
    record = json.loads((evidence/'093-rejected-generated-sampler-boundary/frequency-differences.json').read_text())
    path = evidence/'083-clean-start-sampler-control/runtime-ttnn.mlir.gz'
    data = gzip.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != record['native_ir_uncompressed_sha256']:
        raise ValueError('Native frequency IR seal changed')
    body = data.decode().split('func.func private @main_const_eval_133()', 1)[1].split('return', 1)[0]
    literal = re.search(r'dense<"0x([0-9A-Fa-f]+)"\s*>\s*:\s*tensor<1x128xf32>', body)
    if literal is None:
        raise ValueError('Expected fixed timestep frequency constant is absent')
    native = torch.frombuffer(bytearray.fromhex(literal[1]), dtype=torch.float32).clone()
    if template.dtype != torch.float32 or template.shape != (128,):
        raise ValueError('Expected original FP32 timestep-frequency vector')
    changed = torch.nonzero(native != template).flatten().tolist()
    if (changed != [item['index'] for item in record['differences']] or
            any(template[item['index']].item() != item['eager_cpu'] or
                native[item['index']].item() != item['native'] for item in record['differences'])):
        raise ValueError('Only the two recorded one-ULP timestep constants may be restored')
    return native, {'kind': 'restore_native_timestep_frequency_constant',
        'source': str(path.relative_to(ROOT)), 'native_ir_uncompressed_sha256': record['native_ir_uncompressed_sha256'],
        'changed_indices': changed, 'differences': record['differences']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--native-control-result', type=Path, required=True)
    parser.add_argument('--sampler-native-control-dir', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, default=Path.home()/'models/genrecon')
    parser.add_argument('--repair-embedding-indices', action='store_true')
    parser.add_argument('--repair-native-time-frequencies', action='store_true',
        help='Restore only the two fixed timestep frequencies from sealed native control083/110')
    parser.add_argument('--profile-markers', action='store_true')
    parser.add_argument('--flush-device-profiler', action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('runs must be positive')
    directory, result = args.directory.resolve(), args.result.resolve()
    if result.exists():
        raise FileExistsError(result)
    os.environ.update(ATTN_BACKEND='sdpa', SPARSE_ATTN_BACKEND='sdpa', SPARSE_CONV_BACKEND='none')
    from experiments.tenstorrent.experiment_identity import sha256
    report = {'status': 'running', 'scope': __doc__, 'script_sha256': sha256(Path(__file__)),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'instrumented': os.environ.get('TT_METAL_DEVICE_PROFILER') == '1'}

    def phase(name):
        report['phase'] = name
        temporary = result.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(result)
        print(name, flush=True)

    device = None
    phase('validate_export_and_controls')
    try:
        import torch
        torch.set_num_threads(16)
        from experiments.tenstorrent.trajectory import TrajectoryFixture, compare_trajectories
        export = json.loads((directory/'export.json').read_text())
        if (export.get('status') != 'exported_unvalidated' or export.get('case') != 'shape_flow_tensor' or
                export.get('device_precision') != 'float32' or export.get('attention_mode') != 'fp32_reciprocal' or
                export.get('rope_mode') != 'real_pairs_flat_head_first' or
                export['compiler_options'].get('math_fidelity') != 'hifi4' or
                export['compiler_options'].get('optimization_level') != 0):
            raise ValueError('Expected the calibrated current FP32 shape export')
        if sha256(directory/'reference.pt') != export['reference_sha256']:
            raise ValueError('Export reference changed')
        for name, digest in export['artifact_sha256'].items():
            if sha256(directory/name) != digest:
                raise ValueError('Export artifact changed: '+name)
        report['export_sha256'] = sha256(directory/'export.json')
        control_dir = args.sampler_native_control_dir.resolve()
        control = json.loads((control_dir/'result.json').read_text())
        report['torch'] = torch.__version__
        if report['torch'] != control['versions']['torch']:
            raise ValueError('Sampler CPU arithmetic Torch version differs from native control')
        if (control.get('phase') != 'complete' or control.get('status') not in ('passed', 'numerical_mismatch') or
                control.get('xla_executed_fallback_ops') or control.get('xla_aten_counters') or
                control.get('recompiled_during_trajectory')):
            raise ValueError('Native sampler control is incomplete or has fallback/recompilation')
        for key in ('case', 'device_precision', 'attention_mode', 'rope_mode', 'seed'):
            if control.get(key) != export.get(key):
                raise ValueError('Sampler/export recipe mismatch: '+key)
        reference_files = {'flow_probes.py', 'probes.py', 'reference.py', 'trajectory.py', 'scene_sampling.py',
                           'inference_attention.py', 'inference_precision.py', 'inference_rope.py'}
        shared = {name: digest for name, digest in control['source_files_sha256'].items()
                  if name.startswith(('genrecon/', 'configs/')) or Path(name).name in reference_files}
        if not shared or any(sha256(ROOT/name) != digest for name, digest in shared.items()):
            raise ValueError('Original model/reference/preparation sources changed')
        checkpoint = args.checkpoints/(control['checkpoint']['url'].rsplit('/', 1)[-1])
        if sha256(checkpoint) != control['checkpoint']['sha256']:
            raise ValueError('Physical checkpoint changed')
        payloads = {}
        for name in ('trajectory-tt.pt', 'trajectory-reference.pt'):
            digest = control['trajectory_artifacts_sha256'].get(name)
            if not digest or sha256(control_dir/name) != digest:
                raise ValueError('Native sampler artifact seal differs: '+name)
            payloads[name] = torch.load(control_dir/name, weights_only=True, map_location='cpu')
        expected = payloads['trajectory-tt.pt']
        reference = torch.load(directory/'reference.pt', weights_only=True, map_location='cpu')
        inputs = reference['inputs']
        fixture = TrajectoryFixture('shape', inputs, expected['coords'])
        identity, _ = fixture.run(lambda x, t, cond: torch.zeros_like(x))
        compare_trajectories(identity, expected)
        compare_trajectories(identity, payloads['trajectory-reference.pt'])
        report['sampler_native_control'] = {'directory': str(control_dir),
            'result_sha256': sha256(control_dir/'result.json'), 'source_hashes_checked': len(shared),
            'checkpoint_sha256': control['checkpoint']['sha256'],
            'artifact_sha256': control['trajectory_artifacts_sha256']}
        forward_path = args.native_control_result.resolve()
        forward = json.loads(forward_path.read_text())
        if (forward.get('status') != 'passed_smoke' or forward.get('export_sha256') != report['export_sha256'] or
                sha256(forward_path.with_suffix('.pt')) != forward.get('output_sha256')):
            raise ValueError('Canonical generated forward control changed or uses another export')
        forward_expected = torch.load(forward_path.with_suffix('.pt'), weights_only=True, map_location='cpu')
        phase('waiting_for_device')
        lock = os.open('/tmp/genrecon-tt-pjrt.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        import ttnn
        paths = sorted({Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
            if '/' in line and any(key in line for key in ('/_ttnn.so', '/_ttnncpp.so', 'libtt_metal.so'))})
        if not paths:
            raise ValueError('Cannot identify loaded native libraries')
        report['native_library_sha256'] = {str(path): sha256(path) for path in paths}
        graphs = list((directory/'codegen').glob('graph_*/main.py'))
        if len(graphs) != 1:
            raise ValueError('Expected one generated graph')
        graph = graphs[0].parent
        sys.path.insert(0, str(graph)); os.chdir(graph)
        spec = importlib.util.spec_from_file_location('generated_sampler_flow', graph/'main.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if args.repair_embedding_indices:
            symbol = 'cpu_hoisted_const_eval_b6969c39'
            original = getattr(module, symbol, None)
            ir = '\n'.join(p.read_text() for p in (graph/'irs').glob('ttnn_*.mlir'))
            if original is None or f'call @{symbol}' not in ir or 'tensor<128xui32' not in ir:
                raise ValueError('Expected declared uint32 index cast is absent')
            def corrected_indices(arg):
                output = original(arg)
                if not torch.equal(ttnn.to_torch(output).reshape(-1).long(), torch.arange(128)^1):
                    raise ValueError('Unexpected adjacent-pair index values')
                return ttnn.to_dtype(output, ttnn.uint32)
            setattr(module, symbol, corrected_indices)
            report['compatibility_repairs'] = list(forward.get('compatibility_repairs', []))
        if bool(forward.get('compatibility_repairs')) != args.repair_embedding_indices:
            raise ValueError('Canonical forward repair differs')
        phase('prepare_weights_and_dynamic_inputs')
        weights = module.load_weights_for__main()
        device = module.utils.DeviceGetter._instance
        initial = module.load_activations_for__main()
        templates = [ttnn.to_torch(value).clone() for value in initial]
        for value in initial:
            ttnn.deallocate(value)
        mapping = match_activation_inputs(templates, inputs)
        report['activation_input_mapping'] = mapping
        if args.flush_device_profiler and not report['instrumented']:
            raise ValueError('Profiler flushing requires an instrumented SDK')
        if args.profile_markers:
            from tracy import signpost
        else:
            def signpost(label):
                pass
        calls = []

        def infer(x, t, cond, label):
            values = (x, t, cond['context'], cond['projected'])
            signpost(label+'_TRANSFER_BEGIN')
            start = time.perf_counter()
            activations = [ttnn.from_torch(templates[i] if index is None else values[index],
                dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG) for i, index in enumerate(mapping)]
            ttnn.synchronize_device(device)
            preparation = time.perf_counter()-start
            signpost(label+'_TRANSFER_END')
            signpost(label+'_BEGIN')
            start = time.perf_counter()
            outputs = module._main(activations, weights)
            if len(outputs) != 1:
                raise ValueError('Expected one denoiser output')
            actual = ttnn.to_torch(outputs[0]).float()
            seconds = time.perf_counter()-start
            signpost(label+'_END')
            ttnn.deallocate(outputs[0])
            flush = 0.0
            if args.flush_device_profiler:
                start = time.perf_counter(); ttnn.ReadDeviceProfiler(device)
                flush = time.perf_counter()-start
            calls.append({'label': label, 't': t.tolist(), 'positive': cond is fixture.positive,
                          'preparation_seconds': preparation, 'model_seconds': seconds,
                          'profiler_flush_seconds': flush})
            return actual

        phase('first_forward_and_warmup')
        with torch.no_grad():
            actual = infer(inputs[0], inputs[1], fixture.positive, 'GENRECON_FIRST')
            torch.testing.assert_close(actual, forward_expected, rtol=0, atol=0)
            report['native_control'] = {'result_sha256': sha256(forward_path), 'bitwise_equal': True}
            for index, cond in enumerate((fixture.positive, fixture.negative)):
                infer(inputs[0], inputs[1], cond, 'GENRECON_WARMUP_'+str(index))
            if args.repair_native_time_frequencies:
                index = mapping.index(None)
                templates[index], repair = restore_native_time_frequencies(
                    templates[index], report['sampler_native_control']['result_sha256'])
                report.setdefault('compatibility_repairs', []).append(repair)
            for trial in range(args.runs):
                phase('sampler_trial_'+str(trial))
                begin = len(calls)
                def dynamic_forward(x, t, cond):
                    return infer(x, t, cond, f'GENRECON_SAMPLER_{trial}_CALL_{len(calls)-begin:02d}')
                trajectory, measurements = fixture.run(dynamic_forward)
                torch.save(trajectory, result.parent/f'trajectory-{trial}.pt')
                checks = compare_trajectories(trajectory, payloads['trajectory-reference.pt'])
                row = {'sampler': measurements, 'calls': calls[begin:],
                    'all_states_bitwise_native': False, 'vs_stock_cpu': checks,
                    'vs_native': compare_trajectories(trajectory, expected),
                    'artifact_sha256': sha256(result.parent/f'trajectory-{trial}.pt')}
                report.setdefault('trials', []).append(row)
                phase('check_trial_'+str(trial))
                assert_bitwise_trajectory(trajectory, expected)
                row['all_states_bitwise_native'] = True
                phase('validated_trial_'+str(trial))
        report['timing_scope'] = ('Fresh five-activation H2D/layout preparation and synchronization timed separately. '
            'Marked model calls include generated denoiser and output D2H. Original CPU Euler/CFG and all12steps/21calls retained. '
            'Instrumented sampler wall includes per-call profiler flushing; marked model/transfer windows exclude flushing. '
            'Resident original weights and const-eval cache; compilation and warmups excluded from sampler trials.')
        report['warmup_calls'] = calls[:3]
        report['status'] = 'completed_native_matched_sampler'
        phase('complete')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        phase(report.get('phase', 'initialization'))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)
        # The device lock stays live through native atexit teardown.
    return 0


if __name__ == '__main__':
    sys.exit(main())
