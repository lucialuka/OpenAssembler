"""Isolated low 64-bit product from recovered 32-bit multiply and multiply-high."""
from pathlib import Path
import builtins,importlib
HERE=Path(__file__).resolve().parent

def generate():
 source=(HERE/'baseline-ra.py').read_text();anchor='        if op == "mul" and "wide" in m and t in ("u32", "s32"):';assert source.count(anchor)==1
 source=source.replace(anchor,'''        if op == "mul" and m == ("lo", t) and t in ("u64", "s64"):
            (xl, xh), (yl, yh) = self.operand64(a[1]), self.operand64(a[2])
            low, carry, cross, high = self.new(), self.new(), self.new(), self.new()
            self.emit("IMAD {d0}, {s0}, {s1}, RZ", [low], [xl, yl])
            self.emit("IMAD.HI.U32 {d0}, {s0}, {s1}, RZ", [carry], [xl, yl])
            self.emit("IMAD {d0}, {s0}, {s1}, {s2}", [cross], [xh, yl, carry])
            self.emit("IMAD {d0}, {s0}, {s1}, {s2}", [high], [xl, yh, cross])
            self.bind(a[0].name, (low, high))
            return
'''+anchor)
 contracts=(HERE/'baseline-contracts.py').read_text();anchor="    for dst,src in product(('s32','u32','s64','u64'), repeat=2): add('cvt',dst+'.'+src,2)";assert contracts.count(anchor)==1
 contracts=contracts.replace(anchor,"    for t in ('u64','s64'): add('mul','lo.'+t,3)\n"+anchor)
 return source,contracts

def install():
 for name in('ra','contracts'):
  module=importlib.import_module('openassembler.backend.'+name);path=HERE/('candidate-'+name+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)

if __name__=='__main__':
 for name,source in zip(('ra','contracts'),generate()):(HERE/('candidate-'+name+'.py')).write_text(source)
