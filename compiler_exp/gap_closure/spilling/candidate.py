"""Staged allocation candidate using explicit local resources at the build boundary."""
from pathlib import Path
import builtins
import importlib.util
import sys

HERE=Path(__file__).resolve().parent


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def generate():
    ra=(HERE/'baseline-ra.py').read_text()
    ra=once(ra,'    phys = allocate(ops, s.r1)\n    lowered = lower(ops, phys)',
            '    from openassembler.backend.spill import allocate_kernel\n    ops,phys,resources=allocate_kernel(s,ops)\n    lowered = lower(ops, phys)')
    ra=once(ra,'"smem": s.smem_bytes}', '"smem": s.smem_bytes, **resources}')
    build=(HERE/'baseline-build.py').read_text()
    build=once(build,'def generate(kernel, arch: str, opt_level: str, *, tensor_config=None):',
                'def generate(kernel, arch: str, opt_level: str, *, tensor_config=None, resource_info=None):')
    build=once(build,'        labels = st["labels"]','        labels = st["labels"]\n        if resource_info is not None:resource_info.update({k:st[k]for k in ("local_bytes","spill_slots","register_limit")})')
    build=once(build,'        phys = ra.allocate(ops, s.r1)\n        lowered = ra.lower(ops, phys)',
                '        from openassembler.backend.spill import allocate_kernel\n        ops,phys,resources=allocate_kernel(s,ops)\n        if resource_info is not None:resource_info.update(resources)\n        lowered = ra.lower(ops, phys)')
    build=once(build,'            with so.phase(\'backend\'):\n                code, labels, words, exits, nregs = generate(module.kernels[kname], opt.arch, opt.opt_level,\n                                                          tensor_config=tensor_config)',
                '            resources={}\n            with so.phase(\'backend\'):\n                code, labels, words, exits, nregs = generate(module.kernels[kname], opt.arch, opt.opt_level,\n                                                          tensor_config=tensor_config,resource_info=resources)')
    build=once(build,'            k.local_bytes=local_frame(module.kernels[kname]).bytes',
                '            k.local_bytes=resources.get("local_bytes",local_frame(module.kernels[kname]).bytes)')
    build=once(build,'    local_bytes: int = 0','    local_bytes: int = 0\n    register_allocation: dict | None = None')
    build=once(build,'        if self.local_bytes:', '        if self.register_allocation is not None:r["register_allocation"]=self.register_allocation\n        if self.local_bytes:')
    build=once(build,'            k.local_bytes=resources.get("local_bytes",local_frame(module.kernels[kname]).bytes)',
                '            k.local_bytes=resources.get("local_bytes",local_frame(module.kernels[kname]).bytes)\n            if resources.get("spill_slots",0):k.register_allocation=dict(resources)')
    return dict(ra=ra,build=build)


def install():
    name='openassembler.backend.spill';path=HERE/'spill.py'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    import openassembler.backend.ra as ra
    import openassembler.build as build
    for module in(ra,build):
        source=HERE/('candidate-'+Path(module.__file__).name)
        exec(builtins.compile(source.read_text(),str(source),'exec'),module.__dict__)


if __name__=='__main__':
    for name,source in generate().items():(HERE/f'candidate-{name}.py').write_text(source)
