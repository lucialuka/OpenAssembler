from pathlib import Path
import importlib,sys,types
HERE=Path(__file__).resolve().parent

def install():
    name='openassembler.pyfront.buffers';module=types.ModuleType(name);module.__file__=str(HERE/'candidate/buffers.py');module.__package__='openassembler.pyfront';sys.modules[name]=module
    exec(compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)
    for name in ('compile','__init__'):
        module=importlib.import_module('openassembler.pyfront'+('' if name=='__init__' else '.'+name));path=HERE/'candidate'/(name+'.py')
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
