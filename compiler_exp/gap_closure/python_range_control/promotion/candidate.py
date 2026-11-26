"""Install isolated original loop-control modules only in this process."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def install():
 for name,suffix in(('compile','pyfront'),('return_flow','return_flow')):
  module=importlib.import_module('openassembler.pyfront.'+name);path=HERE/('candidate-'+suffix+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
