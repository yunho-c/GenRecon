"""Per-call native C++ device timings using recorded profiler-read byte ranges.

This report does not depend on Tracy host collection. Operator names/durations
come from actual device measurements; input shapes and per-core timelines are
not present in this C++ report. Host times come from the worker's Python timers.
"""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.tenstorrent.profile_report import number


def summarize_cpp_sampler(data, result, *, expected_operations=4219, expected_matrices=457):
    if (result.get('status') != 'completed_native_matched_sampler' or
            not result.get('native_control', {}).get('bitwise_equal') or not result.get('trials') or
            len(result.get('warmup_calls', [])) != 3):
        raise ValueError('Expected a complete native-matched sampler and three first/warm calls')
    stage = result.get('stage', 'shape')
    if stage not in ('shape', 'tex') or result.get('case', stage+'_flow_tensor') != stage+'_flow_tensor':
        raise ValueError('Expected a matching shape/texture sampler stage and case')
    expected_calls = {'shape': 21, 'tex': 18}[stage]
    header = next(csv.reader(io.StringIO(data.split(b'\n', 1)[0].decode())))
    required = {'GLOBAL CALL COUNT', 'OP NAME', 'DEVICE KERNEL DURATION [ns]'}
    if not required.issubset(header):
        raise ValueError('Native C++ device timing columns are absent')
    output = {'scope': __doc__, 'stage': stage, 'original_calls_per_trial': expected_calls,
        'all_states_bitwise_native': True,
        'original_stock_cpu_gates_pass': all(t['vs_stock_cpu']['all_states_pass_smoke'] for t in result['trials']),
        'device_operations': 0, 'matrix_operations': 0, 'kernel_sum_ns': 0.,
        'host_model_seconds_sum': 0., 'upload_preparation_seconds_sum': 0.,
        'profiler_flush_seconds_sum': 0., 'trials': [], 'by_operation': {}}
    previous_end = 0
    ids = set()
    sequence = None

    def read(call):
        nonlocal previous_end, sequence
        region = call.get('cpp_profile_range')
        if not region or region['begin'] != previous_end or not region['begin'] < region['end'] <= len(data):
            raise ValueError('Missing, overlapping or noncontiguous native profiler byte range')
        segment = data[region['begin']:region['end']]
        if not segment.endswith(b'\n'):
            raise ValueError('Profiler range ends inside a native CSV row')
        reader = csv.DictReader(io.StringIO(segment.decode()), fieldnames=None if region['begin'] == 0 else header)
        rows = list(reader)
        names = [r['OP NAME'] for r in rows]
        counts = Counter(names)
        if len(rows) != expected_operations or counts['MatmulDeviceOperation'] != expected_matrices:
            raise ValueError('Missing/different native device operations in '+call['label'])
        if sequence is None:
            sequence = names
        elif names != sequence:
            raise ValueError('Native operation sequence differs in '+call['label'])
        row_ids = [int(r['GLOBAL CALL COUNT']) for r in rows]
        if len(set(row_ids)) != len(rows) or any(i in ids for i in row_ids):
            raise ValueError('Native operation IDs overlap or repeat')
        ids.update(row_ids)
        previous_end = region['end']
        duration = sum(number(r, 'DEVICE KERNEL DURATION [ns]') for r in rows)
        return rows, {'label': call['label'], 't': call['t'], 'positive': call['positive'],
            'device_operations': len(rows), 'matrix_operations': counts['MatmulDeviceOperation'],
            'kernel_sum_ns': duration, 'global_call_id_min': min(row_ids), 'global_call_id_max': max(row_ids),
            'cpp_profile_range': region, 'host_model_seconds': call['model_seconds'],
            'upload_preparation_seconds': call['preparation_seconds'],
            'profiler_flush_seconds': call['profiler_flush_seconds']}

    for call in result['warmup_calls']:
        read(call)
    labels = set()
    for trial in result['trials']:
        calls = trial['calls']
        if (not trial.get('all_states_bitwise_native') or len(calls) != expected_calls or
                len(trial['sampler']['steps']) != 12 or len(trial['sampler']['forwards']) != expected_calls or
                sum(c['positive'] for c in calls) != 12):
            raise ValueError(f'Original native-matched12-step/{expected_calls}-call schedule is missing')
        current = {'instrumented_sampler_wall_seconds': trial['sampler']['seconds'],
            'model_adapter_seconds_sum': sum(c['seconds'] for c in trial['sampler']['forwards']),
            'outside_model_adapter_seconds': trial['sampler']['seconds']-
                sum(c['seconds'] for c in trial['sampler']['forwards']), 'calls': []}
        for call in calls:
            if call['label'] in labels:
                raise ValueError('Duplicate sampler call label')
            labels.add(call['label'])
            rows, measured = read(call)
            current['calls'].append(measured)
            output['device_operations'] += measured['device_operations']
            output['matrix_operations'] += measured['matrix_operations']
            output['kernel_sum_ns'] += measured['kernel_sum_ns']
            for source, target in [('model_seconds', 'host_model_seconds_sum'),
                                   ('preparation_seconds', 'upload_preparation_seconds_sum'),
                                   ('profiler_flush_seconds', 'profiler_flush_seconds_sum')]:
                output[target] += number(call, source)
            for row in rows:
                item = output['by_operation'].setdefault(row['OP NAME'],
                    {'operation': row['OP NAME'], 'calls': 0, 'kernel_sum_ns': 0.})
                item['calls'] += 1
                item['kernel_sum_ns'] += number(row, 'DEVICE KERNEL DURATION [ns]')
        output['trials'].append(current)
    if previous_end != len(data):
        raise ValueError('Unaccounted device rows after the final recorded profiler read')
    output['by_operation'] = sorted(output['by_operation'].values(), key=lambda x: x['kernel_sum_ns'], reverse=True)
    for item in output['by_operation']:
        item['share_of_kernel_sum'] = item['kernel_sum_ns']/output['kernel_sum_ns']
    output['total_native_rows_including_first_and_warmup'] = len(ids)
    output['all_native_byte_ranges_contiguous_and_complete'] = True
    output['interpretation'] = ('Kernel sums are actual operation durations, not critical-path wall attribution. '
        'First/warm calls excluded from sampler aggregates. Host model timers exclude upload preparation and '
        'profiler flushing; instrumented sampler wall includes flushing and is not ordinary latency. '
        'This report does not infer input shapes or depend on optional Tracy host visualization.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpp-csv', type=Path, required=True)
    parser.add_argument('--sampler-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.sampler_result.read_text())
    expected = args.cpp_csv.resolve().relative_to(args.sampler_result.resolve().parent)
    all_calls = result.get('warmup_calls', [])+[c for t in result.get('trials', []) for c in t['calls']]
    if any(c.get('cpp_profile_range', {}).get('file') != str(expected) for c in all_calls):
        raise ValueError('Recorded profiler ranges refer to another file')
    report = summarize_cpp_sampler(args.cpp_csv.read_bytes(), result)
    from experiments.tenstorrent.experiment_identity import sha256
    report.update(cpp_csv_sha256=sha256(args.cpp_csv), sampler_result_sha256=sha256(args.sampler_result),
                  source_sha256=sha256(Path(__file__)))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
