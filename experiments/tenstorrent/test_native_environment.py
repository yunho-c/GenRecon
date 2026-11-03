import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from experiments.tenstorrent.native_environment import SDK_ENV_KEYS, clean_native_environment


def test_clean_environment_preserves_unrelated_settings_and_original_mapping():
    original = {'PATH': '/example/bin', 'HF_HUB_CACHE': '/example/cache', **{key: '/other/sdk' for key in SDK_ENV_KEYS}}
    result = clean_native_environment(original)
    assert result == {'PATH': '/example/bin', 'HF_HUB_CACHE': '/example/cache'}
    assert all(key in original for key in SDK_ENV_KEYS)


@pytest.mark.parametrize('launcher', ['wrapper', 'self_reexec'])
def test_process_start_removes_python_and_native_loader_paths(tmp_path, launcher):
    root = Path(__file__).resolve().parents[2]
    rogue = tmp_path/'other_sdk'; rogue.mkdir()
    script = tmp_path/'observe.py'
    prelude = '' if launcher == 'wrapper' else (
        f'import sys\nsys.path.insert(0, {str(root)!r})\n'
        'from experiments.tenstorrent.native_environment import reexec_clean_native_process\n'
        'reexec_clean_native_process()\n')
    script.write_text(prelude + 'import os, sys, json\n' +
        f'print(json.dumps({{"sdk": {{k: os.environ.get(k) for k in {SDK_ENV_KEYS!r}}}, '
        f'"rogue_path": {str(rogue)!r} in sys.path, "preserved": os.environ["GENRECON_TEST_PRESERVED"]}}))\n')
    environment = dict(os.environ, **{key: str(rogue) for key in SDK_ENV_KEYS}, GENRECON_TEST_PRESERVED='retained')
    wrapper = [str(Path(__file__).with_name('native_environment.py'))] if launcher == 'wrapper' else []
    result = subprocess.run([sys.executable, *wrapper, str(script)], env=environment,
                            capture_output=True, text=True, check=True, timeout=10)
    observed = json.loads(result.stdout)
    assert observed['sdk'] == {key: None for key in SDK_ENV_KEYS}
    assert not observed['rogue_path']
    assert observed['preserved'] == 'retained'
