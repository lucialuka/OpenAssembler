"""Install experimental modules in-process; production files remain frozen."""
import builtins
import importlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(prefix='candidate'):
    for suffix, name in json.loads((HERE/'inputs.json').read_text()).items():
        module = importlib.import_module(name[:-3].replace('/', '.'))
        path = HERE/(prefix+'-'+suffix+'.py')
        exec(builtins.compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    # Imported aliases of the parser and verifier keep their original module dict.
    # Refresh direct selector aliases used by the baseline A/B module.
    naive = importlib.import_module('openassembler.backend.naive')
    backend = importlib.import_module('openassembler.backend.compile')
    naive.Compiler = backend.Compiler
    naive.NotSupported = backend.NotSupported
