"""Summarize one marked warm interval from a TT-Metal Tracy operations CSV.

Kernel durations are sums across operations, not a critical-path decomposition
of host wall time. Preserve shapes and operation names to avoid guessing whether
a matmul implements attention, a projection, or an MLP.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {key}: {row[key]}')
    return value


def summarize(rows, window='GENRECON_MEASURED_0'):
    active = False
    selected = []
    start, end = None, None
    for row in rows:
        if row['OP TYPE'] == 'signpost':
            if row['OP CODE'] == window+'_BEGIN':
                if start is not None:
                    raise ValueError('Duplicate window begin')
                start = number(row, 'HOST START TS')
                active = True
            elif row['OP CODE'] == window+'_END':
                if not active:
                    raise ValueError('End without begin')
                end = number(row, 'HOST START TS')
                active = False
            continue
        if active:
            selected.append(row)
    if start is None or end is None or end <= start:
        raise ValueError('Missing or invalid profiling window')
    groups, shapes = {}, {}
    device_count = 0
    for row in selected:
        if row['OP TYPE'] != 'tt_dnn_device':
            continue
        # Reject incomplete device data rather than ranking a partial capture.
        duration = number(row, 'DEVICE KERNEL DURATION [ns]')
        host = number(row, 'HOST DURATION [ns]')
        code = row['OP CODE']
        shape = {key: value for key, value in row.items()
                 if key.startswith('INPUT_') and ('PAD[LOGICAL]' in key or key.endswith('_DATATYPE')) and value}
        signature = json.dumps([code, shape], sort_keys=True)
        for table, key in ((groups, code), (shapes, signature)):
            item = table.setdefault(key, {'operation': code, 'calls': 0, 'kernel_sum_ns': 0., 'host_sum_ns': 0.})
            item['calls'] += 1
            item['kernel_sum_ns'] += duration
            item['host_sum_ns'] += host
            if table is shapes:
                item['inputs'] = shape
        device_count += 1
    if not device_count:
        raise ValueError('No profiled device operations in selected interval')
    total = sum(item['kernel_sum_ns'] for item in groups.values())
    ranked = sorted(groups.values(), key=lambda item: item['kernel_sum_ns'], reverse=True)
    for item in ranked:
        item['share_of_kernel_sum'] = item['kernel_sum_ns']/total if total else 0.
    return {'window': window, 'signpost_wall_ns': end-start, 'device_operations': device_count,
            'kernel_sum_ns': total, 'by_operation': ranked,
            'by_operation_and_shape': sorted(shapes.values(), key=lambda item: item['kernel_sum_ns'], reverse=True),
            'interpretation': 'Operation kernel duration sums and their shares, not wall-time attribution. Compilation and warmups excluded by signposts; transfers/dispatch may not appear as device-op rows. Shapes preserved; no inferred attention/MLP attribution.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--window', default='GENRECON_MEASURED_0')
    args = p.parse_args()
    with args.csv.open() as stream:
        report = summarize(list(csv.DictReader(stream)), args.window)
    report.update(csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(), source_csv=str(args.csv))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
