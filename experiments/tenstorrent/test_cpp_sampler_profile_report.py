import copy
import csv
import io
import pytest
from experiments.tenstorrent.cpp_sampler_profile_report import summarize_cpp_sampler


def capture(*, missing_row=False, blank_names=False, nonfinite=False, stage='shape'):
    count = {'shape': 21, 'tex': 18}[stage]
    header = ['GLOBAL CALL COUNT', 'OP NAME', 'DEVICE KERNEL DURATION [ns]']
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    calls = []
    end = 0
    for index in range(count+3):
        begin = end
        names = ['MatmulDeviceOperation', 'ReshapeViewDeviceOperation']
        if missing_row and index == 7:
            names.pop()
        for offset, name in enumerate(names):
            writer.writerow([index*2+offset, '' if blank_names else name,
                             'nan' if nonfinite and index == 6 else '20'])
        end = len(stream.getvalue().encode())
        calls.append({'label': f'CALL_{index}', 't': [1000-index],
            'positive': index < 15, 'model_seconds': .1, 'preparation_seconds': .001,
            'profiler_flush_seconds': 2.,
            'cpp_profile_range': {'file': 'profiler/.logs/cpp_device_perf_report.csv', 'begin': begin, 'end': end}})
    result = {'status': 'completed_native_matched_sampler', 'stage': stage, 'case': stage+'_flow_tensor',
        'native_control': {'bitwise_equal': True},
        'warmup_calls': calls[:3], 'trials': [{'all_states_bitwise_native': True,
            'calls': calls[3:], 'vs_stock_cpu': {'all_states_pass_smoke': False},
            'sampler': {'steps': [{}]*12, 'forwards': [{'seconds': 2.101}]*count, 'seconds': 45.}}]}
    return stream.getvalue().encode(), result


def test_actual_cpp_ranges_cover_every_call_and_preserve_quality_failure():
    data, result = capture()
    actual = summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)
    assert actual['device_operations'] == 42
    assert actual['matrix_operations'] == 21
    assert actual['total_native_rows_including_first_and_warmup'] == 48
    assert actual['kernel_sum_ns'] == 840
    assert actual['host_model_seconds_sum'] == pytest.approx(2.1)
    assert actual['profiler_flush_seconds_sum'] == 42
    assert actual['all_native_byte_ranges_contiguous_and_complete']
    assert not actual['original_stock_cpu_gates_pass']


def test_complete_texture_ranges_cover_original18calls_and_reject_shape_schedule():
    data, result = capture(stage='tex')
    actual = summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)
    assert actual['stage'] == 'tex' and actual['original_calls_per_trial'] == 18
    assert actual['device_operations'] == 36 and actual['matrix_operations'] == 18
    assert actual['total_native_rows_including_first_and_warmup'] == 42
    result['stage'] = 'shape'
    with pytest.raises(ValueError, match='matching shape/texture'):
        summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)
    result['case'] = 'shape_flow_tensor'
    with pytest.raises(ValueError, match='21-call'):
        summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)


def test_historical_shape_report_without_stage_metadata_keeps_original_contract():
    data, result = capture()
    del result['stage'], result['case']
    actual = summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)
    assert actual['stage'] == 'shape' and actual['original_calls_per_trial'] == 21


@pytest.mark.parametrize('corruption', ['missing_row', 'blank_names', 'nonfinite', 'overlap', 'trailing_rows', 'failed_native'])
def test_reject_incomplete_cpp_device_evidence(corruption):
    data, result = capture(**{corruption: True} if corruption in ('missing_row', 'blank_names', 'nonfinite') else {})
    if corruption == 'overlap':
        result['trials'][0]['calls'][1]['cpp_profile_range'] = copy.deepcopy(result['trials'][0]['calls'][0]['cpp_profile_range'])
    elif corruption == 'trailing_rows':
        data += b'10000,MatmulDeviceOperation,20\n'
    elif corruption == 'failed_native':
        result['trials'][0]['all_states_bitwise_native'] = False
    with pytest.raises(ValueError):
        summarize_cpp_sampler(data, result, expected_operations=2, expected_matrices=1)
