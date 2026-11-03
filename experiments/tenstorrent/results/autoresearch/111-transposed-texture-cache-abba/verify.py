"""Audit retained paired measurements and exact trajectory anchors without hardware.

Run with the recorded Forge environment's Python. Gzip artifacts are read in
memory; no checkpoint download, compiler initialization or device access.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

import torch


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / '110-texture512-matching-source-controls'
NATIVE = HERE.parent / '108-texture512-fresh-native-control'
CPU = HERE.parent / '109-texture-cache-cpu-and-identity-fix'


def exact(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        assert actual == expected


def tape(path, expected_sha):
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == expected_sha, path.name
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)


def main():
    seals = json.loads((HERE / 'artifact-sha256.json').read_text())
    for name, digest in seals.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    r = json.loads((HERE / 'result.json').read_text())
    anchor = json.loads((CONTROL / 'ordinary-result.json').read_text())
    assert r['phase'] == 'complete' and r['status'] == 'numerical_mismatch'
    assert r['cache_layout'] == 'heads_t' and r['case'] == 'tex_flow_tensor'
    assert r['parameter_identities_preserved']
    assert r['cached_cpu_all_states_bitwise_ordinary']
    assert r['paired_native_control']['identity_verified']
    assert not r['xla_executed_fallback_ops'] and not r['xla_aten_counters']
    assert not r['prefill_recompiled'] and not r['recompiled_during_trajectory']
    common = anchor['source_files_sha256'].keys() & r['source_files_sha256'].keys()
    for name in common:
        if Path(name).name not in ('cached_flow_sampler.py', 'run.py'):
            assert anchor['source_files_sha256'][name] == r['source_files_sha256'][name]
    native = tape(NATIVE / 'trajectory-tt.pt.gz',
                  anchor['trajectory_artifacts_sha256']['trajectory-tt.pt'])
    exact(tape(NATIVE / 'trajectory-tt.pt.gz', r['trajectory_artifacts_sha256']['trajectory-tt.pt']), native)
    exact(tape(HERE / 'trajectory-inline-control.pt.gz',
               r['trajectory_artifacts_sha256']['trajectory-inline-control.pt']), native)
    ordinary_cpu = tape(CPU / 'trajectory-ordinary-cpu.pt.gz',
                        r['trajectory_artifacts_sha256']['trajectory-ordinary-cpu.pt'])
    exact(tape(CPU / 'trajectory-cached-cpu.pt.gz',
               r['trajectory_artifacts_sha256']['trajectory-cached-cpu.pt']), ordinary_cpu)

    paired = r['paired']
    assert paired['first_cached_and_inline_all_states_bitwise_native_control']
    assert paired['order'] == ['ordinary', 'cached', 'cached', 'ordinary'] * 3
    assert len(paired['trials']) == 12
    for mode, trial in zip(paired['order'], paired['trials']):
        assert trial['mode'] == mode
        assert trial['all_states_bitwise_native_control']
        assert trial['compile_before'] == trial['compile_after']
        assert len(trial['sampler']['steps']) == 12
        assert len(trial['sampler']['forwards']) == 18
        assert trial['sampler']['parameters'] == ordinary_cpu['fixture_identity']['parameters']
        assert trial['total_seconds'] == trial['prefill_pair_seconds'] + trial['sampler']['seconds']
        assert trial['prefill_pair_seconds'] > 0 if mode == 'cached' else trial['prefill_pair_seconds'] == 0
    medians = {}
    for mode in ('ordinary', 'cached'):
        values = [t['total_seconds'] for t in paired['trials'] if t['mode'] == mode]
        assert values == paired['summary'][mode]['seconds']
        medians[mode] = statistics.median(values)
        assert medians[mode] == paired['summary'][mode]['median_seconds']
    reduction = 100 * (1 - medians['cached'] / medians['ordinary'])
    assert reduction == paired['summary']['cached_reduction_percent']
    counts = {}
    for mode, expected in [('prefill', (32, 30)), ('body', (425, 0)), ('inline-control', (457, 30))]:
        text = gzip.decompress((HERE / (mode + '-runtime-ttnn.mlir.gz')).read_bytes()).decode()
        main_ir = text.split('func.func @main(', 1)[1]
        matrices = len(re.findall(r'"ttnn\.(?:matmul|linear)"', main_ir))
        packs = sum('"ttnn.reshape"' in line and 'tensor<1029x3072xf32' in line
                    and 'tensor<1x1029x2x12x128xf32' in line for line in main_ir.splitlines())
        assert (matrices, packs) == expected
        counts[mode] = {'main_matrix_ops': matrices, 'cross_kv_head_packs': packs}
    final = r['trajectory_vs_stock_cpu_fp32']['final']
    assert final == anchor['trajectory_vs_stock_cpu_fp32']['final']
    assert not final['passed'] and final['relative_l2'] == 0.10517075593740585
    print(json.dumps({'validated': True, 'trials': 12, 'paired_medians_seconds': medians,
                      'cached_reduction_percent': reduction, 'native_main_counts': counts,
                      'stock_cpu_quality_gate': 'unchanged_fail'}, indent=2))


if __name__ == '__main__':
    main()
