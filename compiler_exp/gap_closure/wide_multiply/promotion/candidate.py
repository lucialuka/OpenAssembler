"""Install exact staged backend and typed Python modules in an isolated process."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install(baseline=False):
 for name,suffix in(('openassembler.pyfront.compile','pyfront'),('openassembler.backend.ra','ra'),('openassembler.backend.contracts','contracts')):
  module=importlib.import_module(name);path=HERE/(('baseline-'if baseline else'candidate-')+suffix+'.py')
  exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
  if name=='openassembler.pyfront.compile':
   package=importlib.import_module('openassembler.pyfront')
   for attr in('NotSupported','KernelRef'):setattr(package,attr,getattr(module,attr))
