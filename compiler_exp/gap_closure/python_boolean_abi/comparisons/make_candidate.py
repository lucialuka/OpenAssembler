"""Stage canonical predicate comparison without changing production."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
if __name__=='__main__':
 baseline=HERE/'baseline-pyfront.py'
 if not baseline.exists():baseline.write_bytes((ROOT/'openassembler/pyfront/compile.py').read_bytes())
 source=baseline.read_text();old='''        if ka == "pred":
            raise NotSupported("comparing comparison results", node)''';new='''        if ka == "pred":
            # Compare the canonical zero/one values while retaining once-only
            # evaluation of each original predicate operand and chain snapshot.
            left, right = self.s.new_reg("u32"), self.s.new_reg("u32")
            self.s.emit(f" selp.u32 {left.name}, 1, 0, {a.name};")
            self.s.emit(f" selp.u32 {right.name}, 1, 0, {b.name};")
            a, b, ka = left, right, "u32"'''
 assert source.count(old)==1;source=source.replace(old,new);(HERE/'candidate-pyfront.py').write_text(source)
 print('Staged canonical predicate comparison using existing u32 primitives; production unchanged')
