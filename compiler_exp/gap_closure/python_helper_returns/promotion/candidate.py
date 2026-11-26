"""Install only the isolated return-path candidate modules."""
import builtins,importlib,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install():
 for part in('helpers','modules'):
  module=importlib.import_module('openassembler.pyfront.'+part);path=HERE/('candidate-'+part+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 name='openassembler.pyfront.return_flow';module=types.ModuleType(name);module.__file__=str(HERE/'return_flow.py');sys.modules[name]=module
 exec(builtins.compile(Path(module.__file__).read_bytes(),module.__file__,'exec'),module.__dict__)
 for name in('pyfront',):
  module=importlib.import_module('openassembler.pyfront.'+('compile'if name=='pyfront'else name));path=HERE/('candidate-'+name+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
