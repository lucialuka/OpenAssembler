"""Install exact experimental modules without editing production."""
import builtins,importlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def install(prefix='candidate'):
    for suffix,path in json.loads((HERE/'inputs.json').read_text()).items():
        module=importlib.import_module(path[:-3].replace('/','.'))
        source=HERE/(prefix+'-'+suffix+'.py')
        exec(builtins.compile(source.read_bytes(),str(source),'exec'),module.__dict__)
    package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
    for name in ('NotSupported','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
