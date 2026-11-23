"""Isolated candidate overlay used only in experiments."""
from pathlib import Path
import importlib
HERE=Path(__file__).resolve().parent

def install():
    for name in ('types','compile','helpers','__init__'):
        module=importlib.import_module('openassembler.pyfront'+('' if name=='__init__' else '.'+name))
        path=HERE/'candidate'/(name+'.py')
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
