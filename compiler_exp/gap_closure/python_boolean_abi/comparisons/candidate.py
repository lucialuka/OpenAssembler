import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def install(prefix='candidate'):
 module=importlib.import_module('openassembler.pyfront.compile');source=HERE/(prefix+'-pyfront.py');exec(builtins.compile(source.read_bytes(),str(source),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront')
 for name in('NotSupported','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
