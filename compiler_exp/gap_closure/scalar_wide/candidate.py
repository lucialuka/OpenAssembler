"""Isolated scalar-wide candidate using recovered instruction forms only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


def generate():
    text = (HERE / 'baseline-ra.py').read_text()
    text = replace_once(text, '''                self.emit("IADD3.X {d0}, {s0}, ~{s1}, RZ, {q0}, !PT", [hi], [xh, yh, (carry, False)])''', '''                inverted = self.new()
                self.emit("LOP3.LUT {d0}, {s0}, RZ, RZ, 0x0f, !PT", [inverted], [yh])
                self.emit("IADD3.X {d0}, {s0}, {s1}, RZ, {q0}, !PT", [hi], [xh, inverted, (carry, False)])''')
    first = text.index('            elif isinstance(a[2], ImmI) and 0 <= a[2].value < 32:')
    last = text.index('            self.bind(a[0].name, (lo, hi))', first)
    text = text[:first] + '''            else:
                return self.shift64(ins, t)
''' + text[last:]
    old = '''        elif op in ("shl", "shr") and isinstance(srcs[1], ImmI) and srcs[1].value < 32:
            sh = srcs[1].value
            if op == "shl":
                e(f"SHF.L.U32 {{d0}}, {{s0}}, {sh:#x}, RZ", [x])
            else:
                e(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {sh:#x}, {{s0}}", [x])'''
    new = old.replace('elif op in ("shl", "shr") and isinstance(srcs[1], ImmI) and srcs[1].value < 32:', 'elif op in ("shl", "shr"):')
    new = new.replace('''            sh = srcs[1].value
            if op == "shl":''', '''            if isinstance(srcs[1], ImmI) and 0 <= srcs[1].value < 32:
                sh = srcs[1].value
                if op == "shl":
                    e(f"SHF.L.U32 {{d0}}, {{s0}}, {sh:#x}, RZ", [x])
                else:
                    e(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {sh:#x}, {{s0}}", [x])
            elif op == "shl":''')
    new = new.replace('e(f"SHF.L.U32 {{d0}}, {{s0}}, {sh:#x}, RZ", [x])\n            else:', 'e("SHF.L.U32 {d0}, {s0}, {s1}, RZ", [x, self.reg32(srcs[1])])\n            else:')
    new = new.replace('e(f"SHF.R.{\'S32\' if signed else \'U32\'}.HI {{d0}}, RZ, {sh:#x}, {{s0}}", [x])\n', 'e(f"SHF.R.{\'S32\' if signed else \'U32\'}.HI {{d0}}, RZ, {sh:#x}, {{s0}}", [x])\n', 1)
    new = new.rsplit('e(f"SHF.R.', 1)[0] + '''e(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {{s1}}, {{s0}}", [x, self.reg32(srcs[1])])'''
    text = replace_once(text, old, new)
    text = replace_once(text, '    def saturating_integer(self, ins):', '''    def shift64(self, ins, dtype):
        """PTX u32 counts clamp at 64, including signed shifts across word 32."""
        low, high = self.operand64(ins.args[1])
        count = self.reg32(ins.args[2])
        result_low, result_high = self.new(), self.new()
        if ins.op == "shl":
            self.emit("SHF.L.U32 {d0}, {s0}, {s1}, RZ", [result_low], [low, count])
            self.emit("SHF.L.U64.HI {d0}, {s0}, {s1}, {s2}", [result_high], [low, count, high])
        else:
            self.emit("SHF.R.U64 {d0}, {s0}, {s1}, {s2}", [result_low], [low, count, high])
            signed = dtype == "s64"
            self.emit(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {{s1}}, {{s0}}",
                      [result_high], [high, count])
            if signed:
                delta, arithmetic_low, selected = self.new(), self.new(), self.new()
                upper = self.newp()
                self.emit("IADD3 {d0}, {s0}, -0x20, RZ", [delta], [count])
                self.emit("SHF.R.S32.HI {d0}, RZ, {s1}, {s0}", [arithmetic_low], [high, delta])
                self.emit("ISETP.GE.U32.AND {p0}, PT, {s0}, 0x20, PT", [upper], [count])
                self.emit("SEL {d0}, {s0}, {s1}, {p0}", [selected],
                          [arithmetic_low, result_low, (upper, False)])
                result_low = selected
        self.bind(ins.args[0].name, (result_low, result_high))

    def saturating_integer(self, ins):''')
    text = replace_once(text, '''        p = self.newp()
        x, y = self.reg32(a[1]), self.val32(a[2])
        if t in ("u32", "s32", "b32"):''', '''        if t in ("s64", "u64"):
            return self.setp64(ins, t, cmp)
        p = self.newp()
        x, y = self.reg32(a[1]), self.val32(a[2])
        if t in ("u32", "s32", "b32"):''')
    text = replace_once(text, '    def flush_compare_input(self, value):', '''    def setp64(self, ins, dtype, comparison):
        """Lexicographic high-word comparison with unsigned low-word tie break."""
        low_x, high_x = self.operand64(ins.args[1])
        low_y, high_y = self.operand64(ins.args[2])
        result, high_pred = self.newp(), self.newp()
        if comparison in ("EQ", "NE"):
            self.emit(f"ISETP.{comparison}.U32.AND {{p0}}, PT, {{s0}}, {{s1}}, PT",
                      [high_pred], [high_x, high_y])
            if comparison == "EQ":
                self.emit("ISETP.EQ.U32.AND {p0}, PT, {s0}, {s1}, {q0}",
                          [result], [low_x, low_y, (high_pred, False)])
            else:
                low_pred = self.newp()
                self.emit("ISETP.NE.U32.AND {p0}, PT, {s0}, {s1}, PT", [low_pred], [low_x, low_y])
                self.emit("PLOP3.LUT {p0}, PT, {q0}, {q1}, PT, 0xfc, 0x0",
                          [result], [(high_pred, False), (low_pred, False)])
        else:
            high_equal, low_pred = self.newp(), self.newp()
            strict = "LT" if comparison in ("LT", "LE") else "GT"
            suffix = "" if dtype == "s64" else ".U32"
            self.emit(f"ISETP.{strict}{suffix}.AND {{p0}}, PT, {{s0}}, {{s1}}, PT",
                      [high_pred], [high_x, high_y])
            self.emit("ISETP.EQ.U32.AND {p0}, PT, {s0}, {s1}, PT", [high_equal], [high_x, high_y])
            self.emit(f"ISETP.{comparison}.U32.AND {{p0}}, PT, {{s0}}, {{s1}}, {{q0}}",
                      [low_pred], [low_x, low_y, (high_equal, False)])
            self.emit("PLOP3.LUT {p0}, PT, {q0}, {q1}, PT, 0xfc, 0x0",
                      [result], [(high_pred, False), (low_pred, False)])
        self.bind_pred(ins.args[0].name, result)

    def flush_compare_input(self, value):''')
    text = replace_once(text, 'self.bind(a[0].name, (lo, hi))\n            elif dst.endswith("32")', 'self.bind(a[0].name, self.snapshot_parts((lo, hi)))\n            elif dst.endswith("32")')
    text = replace_once(text, 'self.bind(a[0].name, self.operand64(a[1])[0])', 'self.bind(a[0].name, self.snapshot_parts((self.operand64(a[1])[0],))[0])')
    text = replace_once(text, 'self.bind(a[0].name, self.val32(a[1]))\n            else:', 'self.bind(a[0].name, self.snapshot_parts((self.val32(a[1]),))[0])\n            else:')
    text = replace_once(text, 'self.bind(a[0].name, self.operand64(a[1]))\n            return', 'self.bind(a[0].name, self.snapshot_parts(self.operand64(a[1])))\n            return')
    contract = (HERE / 'baseline-contracts.py').read_text()
    contract = replace_once(contract, "    for cmp in ('eq','ne','lt','le','gt','ge','equ'", "    for t in ('s64','u64'):\n        for cmp in ('eq','ne','lt','le','gt','ge'): add('setp',cmp+'.'+t,3)\n    for cmp in ('lo','ls','hi','hs'): add('setp',cmp+'.u64',3)\n    for cmp in ('eq','ne','lt','le','gt','ge','equ'")
    return {'openassembler/backend/ra.py': text.encode(), 'openassembler/backend/contracts.py': contract.encode()}


def install():
    import openassembler.backend.ra as ra
    import openassembler.backend.contracts as contracts
    for module in (contracts, ra):
        path = HERE / ('candidate-' + Path(module.__file__).name)
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)


if __name__ == '__main__':
    files = generate()
    for name, data in files.items():
        (HERE / ('candidate-' + Path(name).name)).write_bytes(data)
    (HERE / 'candidate-identities.json').write_text(json.dumps({
        'baseline': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        'candidate': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}, indent=2) + '\n')
