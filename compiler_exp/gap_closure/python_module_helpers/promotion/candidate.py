"""Load staged graph/namespace integration in an isolated interpreter."""
import builtins,importlib,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install():
 name='openassembler.pyfront.modules';module=types.ModuleType(name);module.__file__=str(HERE/'modules.py');sys.modules[name]=module
 exec(builtins.compile((HERE/'modules.py').read_bytes(),module.__file__,'exec'),module.__dict__)
 for name,file in(('openassembler.pyfront.helpers','candidate-helpers.py'),('openassembler.pyfront.compile','candidate-pyfront.py')):
  module=importlib.import_module(name);path=HERE/file;exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 for name,file in(('openassembler.frontend','candidate-frontend.py'),('openassembler.build','candidate-build.py'),('openassembler.native_ownership','candidate-ownership.py'),('openassembler.remote_build','candidate-remote.py'),('openassembler.cli','candidate-cli.py')):
  module=importlib.import_module(name);path=HERE/file;exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront');module=importlib.import_module('openassembler.pyfront.compile')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
