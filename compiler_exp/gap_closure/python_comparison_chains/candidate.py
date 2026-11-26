"""Isolated condition-only comparison chains with single evaluation and short circuit."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def install():
 module=importlib.import_module('openassembler.pyfront.compile');path=HERE/'candidate-pyfront.py';exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text();old='''        if isinstance(test, ast.Compare):
            pred, inverted = self._compare(test, branch=True)''';new='''        if isinstance(test, ast.Compare) and len(test.ops) > 1:
            return self._chain_jump(test, label, when)
        if isinstance(test, ast.Compare):
            pred, inverted = self._compare(test, branch=True)''';assert text.count(old)==1;text=text.replace(old,new,1)
 anchor='    def _mov(self, dst: _Reg, src: _Reg, kind: str) -> None:\n';addition='''    def _chain_jump(self, test: ast.Compare, label: str, when: bool) -> None:
        # Every intermediate operand is evaluated once, on the live path only.
        failed = self.s.new_label("chainfalse") if when else label
        left = test.left
        for operator, right in zip(test.ops, test.comparators):
            a, ka, b, kb = self._pair(left, right, test)
            saved = self.s.new_reg(kb)
            self._mov(saved, b, kb)
            comparison = ast.copy_location(ast.Compare(left=(a, ka), ops=[operator],
                                                       comparators=[(saved, kb)]), test)
            pred, inverted = self._compare(comparison, branch=True)
            self.s.emit(f" @{'' if inverted else '!'}{pred.name} bra {failed};")
            left = (saved, kb)
        if when:
            self.s.emit(f" bra {label};")
            self.s.emit(f"{failed}:")

'''+anchor;assert text.count(anchor)==1;text=text.replace(anchor,addition,1);(HERE/'candidate-pyfront.py').write_text(text)
