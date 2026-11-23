"""Construct a retained typed integer candidate from pre-extension source snapshots."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def replace(text,old,new):
    assert text.count(old)==1,(old[:100],text.count(old))
    return text.replace(old,new)


def build():
    files={p.name:p.read_text() for p in (HERE/'before').glob('*.py')}
    s=files['types.py'].replace('# "s32" | "u32" | "f32"', '# "s32" | "u32" | "s64" | "u64" | "f32"')
    s=s.replace('u32 = Scalar("u32")','u32 = Scalar("u32")\ni64 = Scalar("s64")\nu64 = Scalar("u64")');files['types.py']=s
    s=files['__init__.py'].replace('Array, f32, i32, u32','Array, f32, i32, i64, u32, u64').replace('"f32", "i32", "u32",','"f32", "i32", "u32", "i64", "u64",');files['__init__.py']=s
    s=files['helpers.py'].replace("target[3:] not in c._MATH1", "target[3:] not in c._MATH1 | set(c._INT_TYPES)")
    files['helpers.py']=s
    s=files['compile.py']
    s=replace(s,'"f32", "i32", "u32"}', '"f32", "i32", "u32", "i64", "u64"}')
    s=replace(s,'_BUILTINS = {', '_INT_TYPES = {"i32": "s32", "u32": "u32", "i64": "s64", "u64": "u64"}\n_BUILTINS = {')
    s=replace(s,'r in ("pf.f32", "pf.i32", "pf.u32")','r in ("pf.f32", "pf.i32", "pf.u32", "pf.i64", "pf.u64")')
    s=s.replace('Array element type must be f32, i32 or u32','Array element type must be f32, i32, u32, i64 or u64').replace('use i32, u32, f32 or Array(f32|i32|u32)','use i32, u32, i64, u64, f32 or Array of those types')
    s=replace(s,'        if kind == "ptr":','        if kind in ("ptr", "s64", "u64"):')
    s=replace(s,'        self.s.smem_bytes += 4 * count','        if elem.ptx in ("s64", "u64"):\n            raise NotSupported("shared_array currently requires a 32-bit element type", node)\n        self.s.smem_bytes += 4 * count')
    s=replace(s,'op = {"f32": "mov.f32", "pred": "mov.pred"}.get(kind, "mov.u32")','op = {"f32": "mov.f32", "pred": "mov.pred", "s64": "mov.b64", "u64": "mov.b64"}.get(kind, "mov.u32")')
    s=replace(s,'        if isinstance(v, int):\n            if want == "f32":','''        if want in ("s64", "u64"):
            low, high = (-(1 << 63), (1 << 63) - 1) if want == "s64" else (0, (1 << 64) - 1)
            if type(v) is not int or not low <= v <= high:
                raise NotSupported(f"literal {v!r} does not fit in {want}", node)
            r = self.s.new_reg(want)
            self.s.emit(f" mov.{want} {r.name}, {v};")
            return r, want
        if isinstance(v, int):
            if want == "f32":''')
    s=replace(s,'        if isinstance(op, ast.Div):','''        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            return self._integer_bits(op, left, right, node, into=into)
        if isinstance(op, ast.Div):''')
    s=replace(s,'        if ka == "pred":\n            raise NotSupported("arithmetic on a comparison result", node)','''        if ka == "pred":
            raise NotSupported("arithmetic on a comparison result", node)
        if ka in ("s64", "u64") and mnem == "mul":
            raise NotSupported("64-bit multiplication is not supported; no narrowing is performed", node)''')
    s=replace(s,'f" {mnem}{\'.lo\' if mnem == \'mul\' else \'\'}.s32 {dst.name}, {a.name}, {b.name};"','f" {mnem}{\'.lo\' if mnem == \'mul\' else \'\'}.{ka if ka in (\'s64\',\'u64\') else \'s32\'} {dst.name}, {a.name}, {b.name};"')
    anchor='    def _to_f32(self, x, node) -> _Reg:'
    methods='''    def _integer_bits(self, op, left, right, node, into=None):
        if isinstance(op, (ast.LShift, ast.RShift)):
            value, kind = self._ev(left)
            bits = 64 if kind in ("s64", "u64") else 32
            count = None if isinstance(right, tuple) else _const_value(right)
            if kind not in ("s32", "u32", "s64", "u64") or type(count) is not int or not 0 <= count < bits:
                raise NotSupported(f"integer shifts require a constant count in [0, {bits-1}]", node)
            result = into if into is not None and into.kind == kind else self.s.new_reg(kind)
            mnemonic = "shl.b" + str(bits) if isinstance(op, ast.LShift) else "shr." + kind
            self.s.emit(f" {mnemonic} {result.name}, {value.name}, {count};")
            return result, kind
        a, ka, b, kb = self._pair(left, right, node)
        if ka != kb or ka not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("bitwise operands require matching integer types", node)
        result = into if into is not None and into.kind == ka else self.s.new_reg(ka)
        mnemonic = {ast.BitAnd:"and", ast.BitOr:"or", ast.BitXor:"xor"}[type(op)]
        self.s.emit(f" {mnemonic}.b{64 if ka.endswith('64') else 32} {result.name}, {a.name}, {b.name};")
        return result, ka

    def _integer_cast(self, want, arg, node):
        literal = _const_value(arg)
        if literal is not None:
            if type(literal) is not int:
                raise NotSupported("typed integer casts require integer values, not floats", node)
            bits = 64 if want.endswith("64") else 32
            literal &= (1 << bits) - 1
            if want.startswith("s") and literal >= (1 << (bits - 1)):
                literal -= 1 << bits
            return self._lit_as(literal, want, arg)
        value, kind = self._eval(arg)
        if kind not in ("s32", "u32", "s64", "u64"):
            raise NotSupported("typed integer casts require integer values, not floats or predicates", node)
        result = self.s.new_reg(want)
        self.s.emit(f" cvt.{want}.{kind} {result.name}, {value.name};")
        return result, want

    def _wide_compare(self, a, b, kind, comparison):
        # Exact lexicographic comparison: signed/unsigned high words, then unsigned low words.
        al, ah, bl, bh = [self.s.new_reg("u32") for _ in range(4)]
        self.s.emit(f" mov.b64 {{{al.name},{ah.name}}}, {a.name};")
        self.s.emit(f" mov.b64 {{{bl.name},{bh.name}}}, {b.name};")
        first, equal, low, masked, result = [self.s.new_reg("pred") for _ in range(5)]
        if comparison in ("eq", "ne"):
            self.s.emit(f" setp.{comparison}.u32 {first.name}, {ah.name}, {bh.name};")
            self.s.emit(f" setp.{comparison}.u32 {low.name}, {al.name}, {bl.name};")
            self.s.emit(f" {'and' if comparison == 'eq' else 'or'}.pred {result.name}, {first.name}, {low.name};")
        else:
            high_cmp = "lt" if comparison in ("lt", "le") else "gt"
            high_kind = "s32" if kind == "s64" else "u32"
            self.s.emit(f" setp.{high_cmp}.{high_kind} {first.name}, {ah.name}, {bh.name};")
            self.s.emit(f" setp.eq.u32 {equal.name}, {ah.name}, {bh.name};")
            self.s.emit(f" setp.{comparison}.u32 {low.name}, {al.name}, {bl.name};")
            self.s.emit(f" and.pred {masked.name}, {equal.name}, {low.name};")
            self.s.emit(f" or.pred {result.name}, {first.name}, {masked.name};")
        return result

'''
    s=replace(s,anchor,methods+anchor)
    s=replace(s,'        if ka == "f32" and cmp == "ne":','''        if ka in ("s64", "u64"):
            return self._wide_compare(a, b, ka, cmp), False
        if ka == "f32" and cmp == "ne":''')
    s=replace(s,'        if isinstance(node.op, ast.Not):','''        if isinstance(node.op, ast.Invert):
            value, kind = self._eval(node.operand)
            if kind not in ("s32", "u32", "s64", "u64"):
                raise NotSupported("bitwise inversion requires an integer", node)
            result = self.s.new_reg(kind)
            self.s.emit(f" xor.b{64 if kind.endswith('64') else 32} {result.name}, {value.name}, {hex((1 << (64 if kind.endswith('64') else 32)) - 1)};")
            return result, kind
        if isinstance(node.op, ast.Not):''')
    s=replace(s,'        if k not in ("f32", "s32"):', '''        if k == "s64":
            dst = self.s.new_reg(k)
            self.s.emit(f" sub.s64 {dst.name}, 0, {v.name};")
            return dst, k
        if k not in ("f32", "s32"):''')
    s=replace(s,'        if fname in ("min", "max"):', '''        if fname.startswith("pf.") and short in _INT_TYPES:
            if len(args) != 1:
                raise NotSupported(f"{short}() takes exactly one integer argument", node)
            return self._integer_cast(_INT_TYPES[short], args[0], node)
        if fname in ("min", "max"):''')
    s=replace(s,'        ci = _const_value(sub.slice)','        element_bytes = 8 if arr.elem in ("s64", "u64") else 4\n        ci = _const_value(sub.slice)')
    s=s.replace('if 4 * ci <','if element_bytes * ci <').replace('{4 * ci}', '{element_bytes * ci}').replace('{idx.name}, 4;', '{idx.name}, {element_bytes};')
    s=s.replace("{'f32' if arr.elem == 'f32' else 'u32'}", "{arr.elem if arr.elem in ('f32','s64','u64') else 'u32'}")
    s=replace(s,'        if ptx_t == "u64":', '        if pname in s.arrays:')
    s=replace(s,"{'f32' if ptx_t == 'f32' else 'u32'}", "{ptx_t if ptx_t in ('f32','s64','u64') else 'u32'}")
    # Truthiness of a wide integer uses exact half comparisons too.
    s=replace(s,'                zero = "0f00000000" if k == "f32" else "0"\n                self.s.emit(f" setp.eq.{k} {pred.name}, {v.name}, {zero};")', '''                if k in ("s64", "u64"):
                    zero, _ = self._lit_as(0, k, test)
                    pred = self._wide_compare(v, zero, k, "eq")
                else:
                    zero = "0f00000000" if k == "f32" else "0"
                    self.s.emit(f" setp.eq.{k} {pred.name}, {v.name}, {zero};")''')
    start = s.index('    def _wide_compare(')
    end = s.index('    def _to_f32(', start)
    s = s[:start] + """    def _wide_compare(self, a, b, kind, comparison):
        result = self.s.new_reg("pred")
        self.s.emit(f" setp.{comparison}.{kind} {result.name}, {a.name}, {b.name};")
        return result

""" + s[end:]
    s = s.replace('import inspect\n', 'import inspect\nimport json\n')
    s = s.replace('    device_call_sites: int = 0', '    device_call_sites: int = 0\n    wide_integer: bool = False')
    s = s.replace('        if kind in ("ptr", "s64", "u64"):', '        if kind in ("s64", "u64"):\n            self.wide_integer = True\n        if kind in ("ptr", "s64", "u64"):')
    s = s.replace('    out = [f".visible .entry {name}({param_list})\\n{{"]', '    out = [f".visible .entry {name}({param_list})\\n{{"]')
    anchor = '    for sym, ty, count in s.shared:'
    metadata = """    if s.wide_integer:
        contract = {"kernel": name, "parameters": [
            {"name": pname, "kind": "array" if pname in s.arrays else "scalar",
             "dtype": s.arrays[pname].elem if pname in s.arrays else ptx_t}
            for ptx_t, pname in s.params]}
        out.insert(0, "// openassembler-pyfront-types-v1 " + json.dumps(contract, sort_keys=True, separators=(",", ":")))
"""
    assert s.count(anchor) == 1
    s = s.replace(anchor, metadata + anchor)
    files['compile.py']=s
    for name,text in files.items():(HERE/'candidate'/name).write_text(text)

if __name__=='__main__':build()
