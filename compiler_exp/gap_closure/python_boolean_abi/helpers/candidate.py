"""Install experiment snapshots only inside this Python process."""
import builtins
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(prefix='candidate'):
    for name, suffix in (('compile', 'pyfront'), ('helpers', 'helpers'), ('modules', 'modules')):
        module = importlib.import_module('openassembler.pyfront.'+name)
        path = HERE/(prefix+'-'+suffix+'.py')
        exec(builtins.compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    package = importlib.import_module('openassembler.pyfront')
    module = importlib.import_module('openassembler.pyfront.compile')
    for name in ('NotSupported', 'KernelRef', 'compile_file', 'compile_kernel', 'find_kernel'):
        setattr(package, name, getattr(module, name))
