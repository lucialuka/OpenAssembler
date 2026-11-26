"""Install exact experimental modules without editing production."""
import builtins,importlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def install(prefix='candidate'):
    for suffix,path in json.loads((HERE/'inputs.json').read_text()).items():
        module=importlib.import_module(path[:-3].replace('/','.'))
        source=HERE/(prefix+'-'+suffix+'.py')
        exec(builtins.compile(source.read_bytes(),str(source),'exec'),module.__dict__)
    package=importlib.import_module('openassembler.pyfront')
    for mod,names in [('types',('Array','f32','i32','u32','i64','u64')),('buffers',('Buffer','BufferType')),('compile',('NotSupported','compile_file','compile_kernel','find_kernel'))]:
        module=importlib.import_module('openassembler.pyfront.'+mod)
        for name in names:setattr(package,name,getattr(module,name))
