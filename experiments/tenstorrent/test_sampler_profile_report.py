import copy
import pytest
from experiments.tenstorrent.sampler_profile_report import summarize_sampler


def fixture():
    rows = []
    def window(label, ts):
        rows.append({'OP TYPE': 'signpost', 'OP CODE': label+'_BEGIN', 'HOST START TS': str(ts)})
        for code in ('MatmulDeviceOperation', 'ReshapeViewDeviceOperation'):
            rows.append({'OP TYPE': 'tt_dnn_device', 'OP CODE': code,
                'DEVICE KERNEL DURATION [ns]': '20', 'HOST DURATION [ns]': '2',
                'INPUT_0_Y_PAD[LOGICAL]': '32[17]'})
        rows.append({'OP TYPE': 'signpost', 'OP CODE': label+'_END', 'HOST START TS': str(ts+100)})
    window('GENRECON_FIRST', 100)
    calls = []
    for index in range(21):
        label = f'CALL_{index}'
        window(label, 1000+index*200)
        calls.append({'label': label, 't': [1000-index], 'positive': index < 12,
            'preparation_seconds': .001, 'profiler_flush_seconds': 2.})
    result = {'status': 'completed_native_matched_sampler', 'native_control': {'bitwise_equal': True},
        'trials': [{'all_states_bitwise_native': True, 'calls': calls,
            'vs_stock_cpu': {'all_states_pass_smoke': False},
            'sampler': {'steps': [{}]*12, 'forwards': [{}]*21, 'seconds': 50.}}]}
    return rows, result


def test_complete_sampler_aggregates_only_model_windows_and_preserves_failed_quality_gate():
    rows, result = fixture()
    actual = summarize_sampler(rows, result, expected_operations=2, expected_matrices=1)
    assert actual['device_operations'] == 42
    assert actual['matrix_operations'] == 21
    assert actual['kernel_sum_ns'] == 840
    assert actual['model_window_wall_sum_ns'] == 2100
    assert actual['all_states_bitwise_native']
    assert not actual['original_stock_cpu_gates_pass']
    assert actual['trials'][0]['profiler_flush_seconds'] == 42
    assert actual['by_operation_and_shape'][0]['inputs'] == {'INPUT_0_Y_PAD[LOGICAL]': '32[17]'}


@pytest.mark.parametrize('corruption', ['missing_device_row', 'missing_window_end', 'changed_shape', 'duplicate_call', 'failed_native'])
def test_reject_incomplete_or_unmatched_full_sampler_capture(corruption):
    rows, result = fixture()
    if corruption == 'missing_device_row':
        del rows[6]
    elif corruption == 'missing_window_end':
        rows[:] = [r for r in rows if r['OP CODE'] != 'CALL_2_END']
    elif corruption == 'changed_shape':
        rows[6]['INPUT_0_Y_PAD[LOGICAL]'] = '64[37]'
    elif corruption == 'duplicate_call':
        result['trials'][0]['calls'][3] = copy.deepcopy(result['trials'][0]['calls'][2])
    else:
        result['trials'][0]['all_states_bitwise_native'] = False
    with pytest.raises(ValueError):
        summarize_sampler(rows, result, expected_operations=2, expected_matrices=1)
