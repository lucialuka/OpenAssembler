"""Stage wide shared-array storage, alignment and address scaling before promotion."""
from pathlib import Path
import builtins
import importlib
HERE=Path(__file__).resolve().parent

def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)

def generate():
    text=(HERE/'baseline-compile.py').read_text()
    text=once(text,'division/remainder, float conversions and wide shared arrays remain rejected.','division/remainder and float conversions remain rejected.')
    text=once(text,'kernel top level as  s = shared_array(f32, 256)  (i32/u32 too; constant','kernel top level as  s = shared_array(f32, 256)  (i32/u32/i64/u64 too; constant')
    text=once(text,'shared_array element type must be f32, i32 or u32','shared_array element type must be f32, i32, u32, i64 or u64')
    text=once(text,'if not isinstance(count, int) or count <= 0:', 'if type(count) is not int or count <= 0:')
    text=once(text,'elem: str             # "f32" | "s32" | "u32"','elem: str             # "f32" | "s32" | "u32" | "s64" | "u64"')
    text=once(text,'''        if elem.ptx in ("s64", "u64"):
            raise NotSupported("shared_array currently requires a 32-bit element type", node)
        self.s.smem_bytes += 4 * count''','''        element_bytes = 8 if elem.ptx in ("s64", "u64") else 4
        # Match declaration order and natural alignment, including inter-array
        # padding, when enforcing the existing per-block static storage limit.
        self.s.smem_bytes = ((self.s.smem_bytes + element_bytes - 1)
                             // element_bytes * element_bytes + element_bytes * count)''')
    text=once(text,'self.s.emit(f" shl.b32 {off.name}, {idx.name}, 2;")','self.s.emit(f" shl.b32 {off.name}, {idx.name}, {3 if element_bytes == 8 else 2};")')
    text=once(text,'''    for sym, ty, count in s.shared:
        out.append(f" .shared .align 4 .{'f32' if ty == 'f32' else 'b32'} {sym}[{count}];")''','''    for sym, ty, count in s.shared:
        element_bytes = 8 if ty in ("s64", "u64") else 4
        storage_type = "b64" if element_bytes == 8 else "f32" if ty == "f32" else "b32"
        out.append(f" .shared .align {element_bytes} .{storage_type} {sym}[{count}];")''')
    return text

def install():
    module=importlib.import_module('openassembler.pyfront.compile')
    path=HERE/'candidate-compile.py';exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    package=importlib.import_module('openassembler.pyfront')
    for name in ('NotSupported','KernelRef'):setattr(package,name,getattr(module,name))
    backend=importlib.import_module('openassembler.backend.compile')
    path=HERE/'candidate-backend-compile.py';exec(builtins.compile(path.read_bytes(),str(path),'exec'),backend.__dict__)

if __name__=='__main__':
    (HERE/'candidate-compile.py').write_text(generate())
    source=(HERE/'baseline-backend-compile.py').read_text()
    old='if i in join_at or _opname(o.text) in ("BRA", "EXIT", "WARPSYNC", "MEMBAR", "ERRBAR", "CCTL"):'
    new='if i in join_at or _opname(o.text) in ("BRA", "EXIT", "WARPSYNC", "MEMBAR", "ERRBAR", "CCTL", "BAR"):'
    (HERE/'candidate-backend-compile.py').write_text(once(source,old,new))
