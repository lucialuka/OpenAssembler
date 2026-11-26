"""Install only the isolated allocator; production selector/build stay intact."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent


def install():
    name='openassembler.backend.spill'
    spec=importlib.util.spec_from_file_location(name,HERE/'spill.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
