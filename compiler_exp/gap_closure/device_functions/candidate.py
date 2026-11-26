"""Construct isolated parser/inliner candidates; no production mutation."""
from pathlib import Path
import importlib,builtins,sys
HERE=Path(__file__).resolve().parent

def generate():
    text=(HERE/'baseline-parse.py').read_text()
    anchor='class Module:\n    kernels: dict[str, Kernel]\n    decls: list[VarDecl]'
    assert text.count(anchor)==1;text=text.replace(anchor,anchor+'\n    functions: dict = field(default_factory=dict)\n    inlined_calls: dict = field(default_factory=dict)')
    anchor='def parse_module(src: str) -> Module:\n    src = _strip_comments(src)'
    assert text.count(anchor)==1;text=text.replace(anchor,'''def parse_module(src: str) -> Module:
    functions, calls, origins = {}, {}, {}
    if re.search(r'(?<![\\w$.])\\.func\\b', _mask_strings(_strip_comments(src))):
        from openassembler.ptx.inlining import inline_module
        import sys
        src, functions, calls, origins = inline_module(src, sys.modules[__name__])
    src = _strip_comments(src)''')
    anchor='        kernel.parameter_attributes=attributes'
    assert text.count(anchor)==1
    text=text.replace(anchor,"        if name in origins:\n            assert len(kernel.body)==len(origins[name])\n            for instruction,line in zip(kernel.body,origins[name]):instruction.line=line\n"+anchor)
    assert text.count('return Module(kernels, module_decls)')==1;text=text.replace('return Module(kernels, module_decls)','return Module(kernels, module_decls, functions, calls)')
    return text

def install():
    from types import ModuleType
    name='openassembler.ptx.inlining';module=ModuleType(name);module.__file__=str(HERE/'inlining.py');sys.modules[name]=module
    exec(builtins.compile((HERE/'inlining.py').read_bytes(),module.__file__,'exec'),module.__dict__)
    parser=importlib.import_module('openassembler.ptx.parse');path=HERE/'candidate-parse.py';exec(builtins.compile(path.read_bytes(),str(path),'exec'),parser.__dict__)
    for name,filename in [('openassembler.native_ownership','candidate-native-ownership.py'),('openassembler.build','candidate-build.py')]:
        module=importlib.import_module(name);path=HERE/filename
        exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)

if __name__=='__main__':
    (HERE/'candidate-parse.py').write_text(generate())
    def once(source,old,new):
        assert source.count(old)==1,old
        return source.replace(old,new,1)
    source=(HERE/'baseline-build.py').read_text()
    source=once(source,'    parameter_attributes: dict = field(default_factory=dict)','    parameter_attributes: dict = field(default_factory=dict)\n    inlined_device_calls: dict | None = None')
    source=once(source,"        if self.parameter_attributes:\n            r['parameter_attributes']=self.parameter_attributes","        if self.parameter_attributes:\n            r['parameter_attributes']=self.parameter_attributes\n        if self.inlined_device_calls is not None:\n            r['inlined_device_calls']=self.inlined_device_calls")
    source=once(source,'        k.parameter_attributes = module.kernels[kname].parameter_attributes',"        k.parameter_attributes = module.kernels[kname].parameter_attributes\n        if kname in module.inlined_calls:\n            k.inlined_device_calls={'schema':1,'strategy':'direct-scalar-static-inlining',\n                                    'functions':{name:{**module.functions[name],'call_sites':count}\n                                                 for name,count in module.inlined_calls[kname].items()}}")
    (HERE/'candidate-build.py').write_text(source)
    source=(HERE/'baseline-native-ownership.py').read_text()
    old="        if kernel.get('parameter_attributes'):\n            texts[kernel['name']]['parameter_attributes']=deepcopy(kernel['parameter_attributes'])"
    source=once(source,old,old+"\n        if kernel.get('inlined_device_calls') is not None:\n            texts[kernel['name']]['inlined_device_calls']=deepcopy(kernel['inlined_device_calls'])")
    old="        if kernel.get('parameter_attributes'):\n            texts[kernel['name']]['parameter_attributes']=kernel['parameter_attributes']"
    source=once(source,old,old+"\n        if kernel.get('inlined_device_calls') is not None:\n            texts[kernel['name']]['inlined_device_calls']=kernel['inlined_device_calls']")
    (HERE/'candidate-native-ownership.py').write_text(source)
