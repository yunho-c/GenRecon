import pytest
from experiments.tenstorrent.profile_report import summarize


def marker(suffix, ts):
    return {'OP TYPE': 'signpost', 'OP CODE': 'GENRECON_MEASURED_0_'+suffix, 'HOST START TS': str(ts)}


def operation(duration='20'):
    return {'OP TYPE': 'tt_dnn_device', 'OP CODE': 'MatmulDeviceOperation',
            'DEVICE KERNEL DURATION [ns]': duration, 'HOST DURATION [ns]': '2',
            'INPUT_0_Y_PAD[LOGICAL]': '32[17]'}


def test_only_measured_interval_and_preserve_shapes():
    result = summarize([operation('999'), marker('BEGIN', 100), operation(), operation(), marker('END', 200), operation('999')])
    assert result['device_operations'] == 2
    assert result['kernel_sum_ns'] == 40
    assert result['signpost_wall_ns'] == 100
    assert result['by_operation_and_shape'][0]['inputs'] == {'INPUT_0_Y_PAD[LOGICAL]': '32[17]'}


@pytest.mark.parametrize('rows', [
    [marker('BEGIN', 100), operation()],
    [marker('END', 100)],
    [marker('BEGIN', 100), marker('END', 200)],
    [marker('BEGIN', 100), operation('nan'), marker('END', 200)],
    [marker('BEGIN', 100), operation(''), marker('END', 200)],
    [marker('BEGIN', 100), marker('BEGIN', 110), operation(), marker('END', 200)],
])
def test_incomplete_capture_is_rejected(rows):
    with pytest.raises(ValueError):
        summarize(rows)
