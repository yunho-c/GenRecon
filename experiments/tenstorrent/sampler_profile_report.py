"""Summarize complete marked original-schedule generated sampler captures.

Model windows exclude uploads and profiler flushing. Device kernel sums are
operation-duration sums, not a critical-path decomposition of sampler wall time.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.tenstorrent.profile_report import summarize


def signatures(report):
    return {json.dumps([x['operation'], x['inputs']], sort_keys=True): x['calls']
            for x in report['by_operation_and_shape']}


def summarize_sampler(rows, result, *, expected_operations=4219, expected_matrices=457):
    if (result.get('status') != 'completed_native_matched_sampler' or
            not result.get('native_control', {}).get('bitwise_equal') or not result.get('trials')):
        raise ValueError('Sampler execution has not passed the complete native bitwise gate')
    baseline = summarize(rows, 'GENRECON_FIRST')
    if baseline['device_operations'] != expected_operations:
        raise ValueError('Unexpected complete denoiser operation count')
    baseline_signatures = signatures(baseline)
    output = {'scope': __doc__, 'trials': [], 'all_states_bitwise_native': True,
        'original_stock_cpu_gates_pass': all(t['vs_stock_cpu']['all_states_pass_smoke'] for t in result['trials']),
        'device_operations': 0, 'matrix_operations': 0, 'kernel_sum_ns': 0.,
        'model_window_wall_sum_ns': 0., 'by_operation': {}, 'by_operation_and_shape': {}}
    for trial in result['trials']:
        calls = trial['calls']
        if (not trial.get('all_states_bitwise_native') or len(calls) != 21 or
                len(trial['sampler']['steps']) != 12 or len(trial['sampler']['forwards']) != 21 or
                len({call['label'] for call in calls}) != 21 or
                sum(call['positive'] for call in calls) != 12):
            raise ValueError('Expected the native-matched original12steps/21calls and12positive/9negative branches')
        report = {'calls': [], 'instrumented_sampler_wall_seconds': trial['sampler']['seconds'],
                  'upload_preparation_seconds': sum(c['preparation_seconds'] for c in calls),
                  'profiler_flush_seconds': sum(c['profiler_flush_seconds'] for c in calls)}
        for call in calls:
            current = summarize(rows, call['label'])
            matrices = sum(x['calls'] for x in current['by_operation'] if x['operation'] == 'MatmulDeviceOperation')
            if (current['device_operations'] != expected_operations or matrices != expected_matrices or
                    signatures(current) != baseline_signatures):
                raise ValueError('Missing or different device operations/shapes in '+call['label'])
            report['calls'].append({'label': call['label'], 't': call['t'], 'positive': call['positive'],
                'device_operations': current['device_operations'], 'matrix_operations': matrices,
                'wall_ns': current['signpost_wall_ns'], 'kernel_sum_ns': current['kernel_sum_ns']})
            output['device_operations'] += current['device_operations']
            output['matrix_operations'] += matrices
            output['kernel_sum_ns'] += current['kernel_sum_ns']
            output['model_window_wall_sum_ns'] += current['signpost_wall_ns']
            for key, items in [('by_operation', current['by_operation']),
                               ('by_operation_and_shape', current['by_operation_and_shape'])]:
                for item in items:
                    signature = item['operation'] if key == 'by_operation' else json.dumps(
                        [item['operation'], item['inputs']], sort_keys=True)
                    target = output[key].setdefault(signature, dict(item, calls=0, kernel_sum_ns=0., host_sum_ns=0.))
                    for metric in ('calls', 'kernel_sum_ns', 'host_sum_ns'):
                        target[metric] += item[metric]
        output['trials'].append(report)
    for key in ('by_operation', 'by_operation_and_shape'):
        output[key] = sorted(output[key].values(), key=lambda x: x['kernel_sum_ns'], reverse=True)
        for item in output[key]:
            item['share_of_kernel_sum'] = item['kernel_sum_ns']/output['kernel_sum_ns']
    output['interpretation'] = ('Original CPU Euler/CFG and strict quality gates retained. '
        'Kernel sums/model-window sums exclude uploads and profiler flushing. '
        'Instrumented sampler wall includes flushing and is not an uninstrumented latency estimate. '
        'First forward, warmups and their device rows are excluded from sampler aggregates.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--sampler-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.csv.open() as stream:
        rows = list(csv.DictReader(stream))
    result = json.loads(args.sampler_result.read_text())
    report = summarize_sampler(rows, result)
    from experiments.tenstorrent.experiment_identity import sha256
    report.update(csv_sha256=sha256(args.csv), sampler_result_sha256=sha256(args.sampler_result),
                  source_sha256=sha256(Path(__file__)))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
