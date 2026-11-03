"""Start native experiments without inheriting another Tenstorrent SDK.

Dynamic loader and Python search paths are initialized before a script's main
function. Removing variables after imports cannot reliably undo that state.
This module uses only the standard library and can also launch any Python
experiment: native_environment.py <script.py or -m module> [arguments...].
"""
import os
import sys

SDK_ENV_KEYS = ('PYTHONPATH', 'LD_LIBRARY_PATH', 'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'TT_PJRT_PLUGIN_DIR')


def clean_native_environment(source=None):
    result = dict(os.environ if source is None else source)
    for name in SDK_ENV_KEYS:
        result.pop(name, None)
    return result


def reexec_clean_native_process():
    if any(name in os.environ for name in SDK_ENV_KEYS):
        os.execve(sys.executable, [sys.executable, *sys.argv], clean_native_environment())


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise SystemExit(__doc__)
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], clean_native_environment())
