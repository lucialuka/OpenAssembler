"""Install isolated original-source static-value candidate without source mutation."""
import builtins,importlib,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install():
 name='openassembler.pyfront.static_values';module=types.ModuleType(name);module.__file__=str(HERE/'static_values.py');sys.modules[name]=module
 exec(builtins.compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)
 for name,file in(('modules','candidate-modules.py'),('helpers','candidate-helpers.py'),('compile','candidate-pyfront.py')):
  module=importlib.import_module('openassembler.pyfront.'+name);path=HERE/file;exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
