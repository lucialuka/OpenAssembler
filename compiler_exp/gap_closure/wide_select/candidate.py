"""Isolated wide select using existing independently recovered scalar SEL words."""
from pathlib import Path
import importlib,builtins
HERE=Path(__file__).resolve().parent

def generate():
    source=(HERE/'baseline-ra.py').read_text();anchor='        if op == "selp" and t in ("b32", "u32", "s32", "f32"):'
    assert source.count(anchor)==1
    source=source.replace(anchor,'''        if op == "selp" and t in ("b64", "u64", "s64", "f64"):
            predicate = self.pred(a[3])
            left, right = self.operand64(a[1]), self.operand64(a[2])
            low, high = self.new(), self.new()
            self.emit("SEL {d0}, {s0}, {s1}, {p0}", [low], [left[0], right[0], predicate])
            self.emit("SEL {d0}, {s0}, {s1}, {p0}", [high], [left[1], right[1], predicate])
            self.bind(a[0].name, (low, high))
            return
'''+anchor)
    contracts=(HERE/'baseline-contracts.py').read_text();anchor="    for t in ('u32','s32','b32','f32'): add('selp',t,4)";assert contracts.count(anchor)==1
    contracts=contracts.replace(anchor,"    for t in ('u32','s32','b32','f32','b64','u64','s64','f64'): add('selp',t,4)")
    return source,contracts

def install():
    for name in('ra','contracts'):
        module=importlib.import_module('openassembler.backend.'+name);path=HERE/('candidate-'+name+'.py');exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)

if __name__=='__main__':
    for name,source in zip(('ra','contracts'),generate()):(HERE/('candidate-'+name+'.py')).write_text(source)
