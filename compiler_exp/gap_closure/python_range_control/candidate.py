"""Isolated overflow-to-exhaustion guards for constant nonunit i32 range steps."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def install():
 module=importlib.import_module('openassembler.pyfront.compile');path=HERE/'candidate-pyfront.py'
 exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text();old='''        self.s.emit(f"{cont}:")
        if step > 0:
''';new='''        self.s.emit(f"{cont}:")
        if abs(step) > 1:
            # Bounds are i32. A mathematical next value outside i32 has
            # necessarily crossed stop, so exhaust without wrapping.
            overflow = self.s.new_reg("pred")
            relation, limit = ("gt", _I32_MAX - step) if step > 0 else ("lt", _I32_MIN - step)
            self.s.emit(f" setp.{relation}.s32 {overflow.name}, {ctr.name}, {limit};")
            self.s.emit(f" @{overflow.name} bra {end};")
        if step > 0:
'''
 assert text.count(old)==1;text=text.replace(old,new,1);(HERE/'candidate-pyfront.py').write_text(text)
