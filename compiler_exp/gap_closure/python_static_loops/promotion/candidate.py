"""Load staged tuple semantics in an isolated process, leaving production intact."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install():
 for name in('static_values','modules','helpers','pyfront'):
  module=importlib.import_module('openassembler.pyfront.'+('compile'if name=='pyfront'else name));path=HERE/('candidate-'+name+'.py')
  exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
