"""Alternating complete sampler controls sharing original learned objects."""
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn


class InlineConditionFlow(nn.Module):
    """Recompute original conditioning inside each compiled denoiser call."""
    def __init__(self, prefill, body):
        super().__init__()
        self.prefill = prefill
        self.body = body
        self.eval()

    def forward(self, x, t, context, projected):
        return self.body(x, t, self.prefill(context, projected))


def validate_native_control(directory, report, fixture):
    from experiments.tenstorrent.experiment_identity import sha256
    from experiments.tenstorrent.trajectory import compare_trajectories
    directory = Path(directory)
    source = json.loads((directory/'result.json').read_text())
    if (source.get('phase') != 'complete' or source.get('status') not in ('passed', 'numerical_mismatch') or
        source.get('xla_executed_fallback_ops') or source.get('xla_aten_counters') or source.get('recompiled_during_trajectory')):
        raise ValueError('Native control is incomplete or has fallback/recompilation')
    if (source.get('attention_mode') != 'fp32_reciprocal' or source.get('device_precision') != 'float32' or
        source.get('rope_mode') != 'real_pairs_flat_head_first' or
        source.get('requested_compile_options', {}).get('math_fidelity') != 'hifi4' or
        source.get('requested_compile_options', {}).get('optimization_level') != 0):
        raise ValueError('Native control must use the same prepared FP32 recipe')
    for key in ('case', 'checkpoint', 'config', 'dtype', 'versions', 'seed', 'inputs'):
        if source.get(key) != report.get(key):
            raise ValueError('Native control identity differs: '+key)
    common = source['source_files_sha256'].keys() & report['source_files_sha256'].keys()
    # CLI changes do not change the original model/reference arithmetic. The
    # cache helper and all shared inference/model sources remain hash-checked.
    changed = [key for key in common if Path(key).name != 'cached_flow_sampler.py' and
               source['source_files_sha256'][key] != report['source_files_sha256'][key]]
    if changed:
        raise ValueError('Native control shared sources differ: '+repr(changed))
    path = directory/'trajectory-tt.pt'
    recorded = source.get('trajectory_artifacts_sha256', {}).get(path.name)
    if recorded is None or sha256(path) != recorded:
        raise ValueError('Native control trajectory seal differs or is missing')
    expected = torch.load(path, weights_only=True, map_location='cpu')
    # Validate semantic fixture identity independently of the eventual candidate.
    identity, _ = fixture.run(lambda x, t, cond: torch.zeros_like(x))
    compare_trajectories(identity, expected)
    return expected, {'directory': str(directory.resolve()), 'artifact_sha256': recorded,
                      'shared_source_hashes_checked': len(common), 'identity_verified': True}


def paired_trials(*, rounds, fixture, prefill, body, compiled_prefill, compiled_body,
                  resident, device, inputs, compile_module, expected, first_cached,
                  assert_bitwise, phase, save, output_dir):
    """ABBA rounds with fresh, used and explicitly timed caches for each B."""
    import torch_xla
    import torch_xla.debug.metrics as metrics
    assert_bitwise(first_cached, expected)
    inline = InlineConditionFlow(prefill, body)
    if {id(p) for p in inline.parameters()} != {id(p) for p in body.parameters()}:
        raise ValueError('Inline control must share every original parameter')
    phase('compile_inline_control_and_warm_both_branches')
    start = time.perf_counter()
    compiled_inline = compile_module(inline, 'inline-control')

    def ordinary_forward(x, t, cond):
        return compiled_inline(x.to(device), t.to(device), *resident[id(cond)]).cpu().float()

    with torch.no_grad():
        for cond in (fixture.positive, fixture.negative):
            ordinary_forward(inputs[0], inputs[1], cond)
        warm, _ = fixture.run(ordinary_forward)
    assert_bitwise(warm, expected)
    torch.save(warm, output_dir/'trajectory-inline-control.pt')
    # Exercise fresh cache inputs after every graph exists, outside measurement.
    # Keep the first compiler's input references and this temporary set bounded.
    with torch.no_grad():
        for cond in (fixture.positive, fixture.negative):
            warm_cache = tuple(compiled_prefill(*resident[id(cond)]))
            torch_xla.sync(wait=True)
            compiled_body(inputs[0].to(device), inputs[1].to(device), warm_cache).cpu()
            del warm_cache
    result = {'order': ['ordinary', 'cached', 'cached', 'ordinary']*rounds,
              'inline_control_compile_and_warm_seconds': time.perf_counter()-start,
              'first_cached_and_inline_all_states_bitwise_native_control': True,
              'trials': [], 'scope': 'ABBA rounds in one process, shared original weights. Each cached trial computes and uses a fresh positive/negative native prefill pair; both paths retain all original12steps/21calls. Explicit synchronization, no compilation/warmup/artifact I/O in measured windows.'}

    def compile_state():
        return {'dynamo': dict(torch._dynamo.utils.counters['stats']),
                'xla_compile_samples': {name: metrics.metric_data(name)[0] for name in metrics.metric_names()
                                        if 'Compile' in name and 'Cached' not in name}}

    for index, mode in enumerate(result['order']):
        phase('paired_trial_'+str(index)+'_'+mode)
        torch_xla.sync(wait=True)
        before = compile_state()
        prefill_seconds = 0.0
        with torch.no_grad():
            if mode == 'cached':
                start = time.perf_counter()
                cache = {id(cond): tuple(compiled_prefill(*resident[id(cond)]))
                         for cond in (fixture.positive, fixture.negative)}
                torch_xla.sync(wait=True)
                prefill_seconds = time.perf_counter()-start

                def forward(x, t, cond):
                    return compiled_body(x.to(device), t.to(device), cache[id(cond)]).cpu().float()
            else:
                forward = ordinary_forward
            actual, measurements = fixture.run(forward)
        after = compile_state()
        assert_bitwise(actual, expected)
        row = {'mode': mode, 'prefill_pair_seconds': prefill_seconds,
               'sampler': measurements, 'total_seconds': prefill_seconds+measurements['seconds'],
               'all_states_bitwise_native_control': True,
               'compile_before': before, 'compile_after': after}
        result['trials'].append(row)
        save(result)
        if before != after:
            raise ValueError('Compilation occurred during paired measurement')
        if mode == 'cached':
            del cache
    result['summary'] = {}
    for mode in ('ordinary', 'cached'):
        values = [row['total_seconds'] for row in result['trials'] if row['mode'] == mode]
        result['summary'][mode] = {'seconds': values, 'median_seconds': statistics.median(values),
                                   'min_seconds': min(values), 'max_seconds': max(values)}
    result['summary']['cached_reduction_percent'] = 100*(1-result['summary']['cached']['median_seconds']/
                                                       result['summary']['ordinary']['median_seconds'])
    save(result)
    return result
