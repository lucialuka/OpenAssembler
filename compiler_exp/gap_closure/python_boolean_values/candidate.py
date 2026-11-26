"""Isolated Boolean values and operand-returning homogeneous scalar and/or."""
import builtins,importlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(text,old,new):
 assert text.count(old)==1,old;return text.replace(old,new,1)
def install():
 module=importlib.import_module('openassembler.pyfront.compile');path=HERE/'candidate-pyfront.py';exec(builtins.compile(path.read_bytes(),str(path),'exec'),module.__dict__)
 package=importlib.import_module('openassembler.pyfront')
 for name in('NotSupported','KernelRef','compile_file','compile_kernel','find_kernel'):setattr(package,name,getattr(module,name))
if __name__=='__main__':
 text=(HERE/'baseline-pyfront.py').read_text();text=once(text,'_BUILTINS = {"min", "max", "abs", "float", "int", "range"}','_BUILTINS = {"min", "max", "abs", "float", "int", "range", "bool"}')
 text=once(text,'''        if isinstance(node, ast.Compare):
            pred, _ = self._compare(node, branch=False)''','''        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            return self._boolean_expression(node), "pred"
        if isinstance(node, ast.Compare):
            pred, _ = self._compare(node, branch=False)''')
 text=once(text,'''        if isinstance(node, ast.BoolOp):
            raise NotSupported("'and'/'or' are only supported directly in an if/while condition "
                               "(e.g. 'if a < b and c < d:'), not assigned to a variable or nested "
                               "in another expression", node)''','''        if isinstance(node, ast.BoolOp):
            return self._boolean_operands(node)''')
 text=once(text,'''        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            raise NotSupported("bool constants are only supported as a whole if/while condition "
                               "('while True:')", node)''','''        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            value, kind = self._lit_as(int(node.value), "s32", node)
            return self._boolean_value(value, kind), "pred"''')
 text=once(text,'''        if isinstance(node.op, ast.Not):
            raise NotSupported("'not' is only supported in an if/while condition (it is "
                               "compiled into the branch); write the inverse comparison instead", node)''','''        if isinstance(node.op, ast.Not):
            return self._boolean_expression(node), "pred"''')
 text=once(text,'''        value, kind = self._eval(arg)
        if kind not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("typed integer casts require integer values, not floats or predicates", node)''','''        value, kind = self._eval(arg)
        if kind == "pred":
            converted = self.s.new_reg("s32")
            self.s.emit(f" selp.s32 {converted.name}, 1, 0, {value.name};")
            value, kind = converted, "s32"
        if kind not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("typed integer casts require integer or Boolean values", node)''')
 text=once(text,'''        args, short = node.args, fname.split(".")[-1]
''','''        args, short = node.args, fname.split(".")[-1]
        if fname == "bool":
            if len(args) != 1:
                raise NotSupported("bool() takes exactly one scalar argument", node)
            return self._boolean_value(*self._eval(args[0])), "pred"
''')
 anchor='    def _mov(self, dst: _Reg, src: _Reg, kind: str) -> None:\n'
 text=once(text,anchor,'''    def _boolean_value(self, value, kind):
        if kind == "pred":
            return value
        if kind in ("s64", "u64"):
            zero, _ = self._lit_as(0, kind, self.fdef)
            equal = self._wide_compare(value, zero, kind, "eq")
        else:
            equal = self.s.new_reg("pred")
            zero = "0f00000000" if kind == "f32" else "0"
            self.s.emit(f" setp.eq.{kind} {equal.name}, {value.name}, {zero};")
        result = self.s.new_reg("pred")
        self.s.emit(f" not.pred {result.name}, {equal.name};")
        return result

    def _boolean_expression(self, node):
        result, done = self.s.new_reg("s32"), self.s.new_label("boolean")
        self.s.emit(f" mov.s32 {result.name}, 0;")
        self._jump(node, done, when=False)
        self.s.emit(f" mov.s32 {result.name}, 1;")
        self.s.emit(f"{done}:")
        return self._boolean_value(result, "s32")

    def _boolean_operands(self, node):
        value, kind = self._eval(node.values[0])
        result, done = self.s.new_reg(kind), self.s.new_label("booloperands")
        self._mov(result, value, kind)
        for operand in node.values[1:]:
            truth = self._boolean_value(result, kind)
            self.s.emit(f" @{'' if isinstance(node.op, ast.Or) else '!'}{truth.name} bra {done};")
            literal = _const_value(operand)
            value, other = self._lit_as(literal, kind, operand) if literal is not None and kind != "pred" else self._eval(operand)
            if other != kind:
                raise NotSupported("and/or operands require one fixed scalar kind", operand)
            self._mov(result, value, kind)
        self.s.emit(f"{done}:")
        return result, kind

'''+anchor);(HERE/'candidate-pyfront.py').write_text(text)
