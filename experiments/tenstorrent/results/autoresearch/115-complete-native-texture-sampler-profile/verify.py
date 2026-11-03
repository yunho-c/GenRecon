"""Audit retained complete device records and native trajectory without hardware."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tenstorrent.cpp_sampler_profile_report import summarize_cpp_sampler


def exact(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        assert actual == expected


def main():
    for name, digest in json.loads((HERE/'artifact-sha256.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name
    result = json.loads((HERE/'execute.json').read_text())
    saved = json.loads((HERE/'profile-report.json').read_text())
    data = gzip.decompress((HERE/'cpp-device-perf.csv.gz').read_bytes())
    assert hashlib.sha256(data).hexdigest() == saved['cpp_csv_sha256']
    assert hashlib.sha256((HERE/'execute.json').read_bytes()).hexdigest() == saved['sampler_result_sha256']
    report = summarize_cpp_sampler(data, result)
    for key, value in report.items():
        assert value == saved[key], key
    assert report['stage'] == 'tex' and report['original_calls_per_trial'] == 18
    assert report['device_operations'] == 18*4219
    assert report['matrix_operations'] == 18*457
    assert report['total_native_rows_including_first_and_warmup'] == 21*4219
    assert report['all_native_byte_ranges_contiguous_and_complete']
    assert not report['original_stock_cpu_gates_pass']
    trial = result['trials'][0]
    raw = gzip.decompress((HERE/'trajectory-0.pt.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == trial['artifact_sha256']
    actual = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    anchor = HERE.parent/'110-texture512-matching-source-controls/ordinary-result.json'
    anchor_report = json.loads(anchor.read_text())
    native_raw = gzip.decompress((HERE.parent/'108-texture512-fresh-native-control/trajectory-tt.pt.gz').read_bytes())
    assert hashlib.sha256(native_raw).hexdigest() == anchor_report['trajectory_artifacts_sha256']['trajectory-tt.pt']
    exact(actual, torch.load(io.BytesIO(native_raw), map_location='cpu', weights_only=True))
    assert trial['vs_stock_cpu']['final'] == anchor_report['trajectory_vs_stock_cpu_fp32']['final']
    repair = result['compatibility_repairs'][-1]
    assert repair['changed_indices'] == [12, 106]
    assert repair['bound_native_control_result_sha256'] == hashlib.sha256(anchor.read_bytes()).hexdigest()
    print(json.dumps({'validated': True, 'native_rows_including_first_warm': 21*4219,
                      'sampler_device_operations': 18*4219, 'sampler_matrix_operations': 18*457,
                      'every_native_state_exact': True, 'stock_CPU_gate': 'unchanged_fail'}, indent=2))


if __name__ == '__main__':
    main()
