"""Construct an isolated candidate from exact baseline anchors; never production imported."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('candidate baseline anchor changed: ' + old[:80])
    return text.replace(old, new, 1)


def sources():
    path = ROOT / 'openassembler/backend/ra.py'
    text = path.read_text()
    text = replace_once(text, '    def newp(self) -> PV:', '''    def bind_pred(self, name: str, value: PV) -> None:
        """Preserve partial/loop-carried predicate definitions in one fixed home."""
        if name in self.homes:
            home = self.homes[name]
            if home is None:
                home = self.homes[name] = self.newp()
            if not isinstance(home, PV):
                raise NotSupported(f"mixed predicate/register assignments to {name}")
            last = self.ops[-1] if self.ops else None
            if (last is not None and last.defs == [value] and not last.label
                    and last.guard == getattr(self, "cur_guard", None)
                    and value not in self.env.values()):
                # The new predicate has no other reader: compute straight into
                # the home, preserving the original instruction and its guard.
                last.defs = [home]
            elif value != home:
                self.emit("PLOP3.LUT {p0}, PT, {q0}, PT, PT, 0xf0, 0x0",
                          [home], [(value, False)])
            value = home
        self.env[name] = value

    def newp(self) -> PV:''')
    text = replace_once(text, '''            if ins.args and isinstance(ins.args[0], Var) and ins.op not in ("st", "bra", "red"):
                counts[ins.args[0].name] = counts.get(ins.args[0].name, 0) + 1''', '''            if ins.args and ins.op not in ("st", "bra", "red"):
                destination = ins.args[0]
                destinations = destination.elems if isinstance(destination, Vec) else (destination,)
                for value in destinations:
                    if isinstance(value, Var):
                        counts[value.name] = counts.get(value.name, 0) + 1''')
    # The predicate route must precede generic MOV, which accepts only GPR values.
    old = '''        if "pred" in m and op in ("and", "or", "xor", "not", "mov"):
            p = self.newp()
            srcs = [self.pred(a[1])]
            if op in ("and", "or", "xor"):
                srcs.append(self.pred(a[2]))
                lut = {"and": 0xc0, "or": 0xfc, "xor": 0x3c}[op]
                fmt = f"PLOP3.LUT {{p0}}, PT, {{q0}}, {{q1}}, PT, {lut:#x}, 0x0"
            else:
                fmt = f"PLOP3.LUT {{p0}}, PT, {{q0}}, PT, PT, {0x0f if op == 'not' else 0xf0:#x}, 0x0"
            self.emit(fmt, [p], srcs)
            self.env[a[0].name] = p
            return
'''
    new = old.replace('self.env[a[0].name] = p', 'self.bind_pred(a[0].name, p)')
    text = replace_once(text, old, '')
    text = replace_once(text, '        if op == "mov" and isinstance(a[1], Var) and a[1].name in SREG_S2R:', new + '        if op == "mov" and isinstance(a[1], Var) and a[1].name in SREG_S2R:')
    text = replace_once(text, '        self.env[a[0].name] = p\n\n    def flush_compare_input', '        self.bind_pred(a[0].name, p)\n\n    def flush_compare_input')
    text = replace_once(text, '''        if "v2" in m:
            raise NotSupported("ld.global.v2")''', '''        if "v2" in m:
            g = (self.new(), self.new())
            self.emit(f"LDG.E.64 {{d0}}, [{{s0}}.64{off}]", [g], [base], var_lat=True)
            for element, value in zip(a[0].elems, g):
                if element is not None:
                    self.bind(element.name, value)
            return''')
    text = replace_once(text, '''        if t not in ("b32", "u32", "s32", "f32"):
            raise NotSupported(f"st.global.{t}")
        self.emit(f"STG.E [{{s0}}.64{off}], {{s1}}", [], [base, self.reg32(src)], side=True)''', '''        if t in ("b64", "u64", "s64", "f64"):
            value = self.wide(self.operand64(src))
            self.emit(f"STG.E.64 [{{s0}}.64{off}], {{s1}}", [], [base, value], side=True)
            return
        suffix = {"b8": ".U8", "u8": ".U8", "s8": ".U8", "b16": ".U16", "u16": ".U16", "s16": ".U16",
                  "b32": "", "u32": "", "s32": "", "f32": ""}.get(t)
        if suffix is None:
            raise NotSupported(f"st.global.{t}")
        self.emit(f"STG.E{suffix} [{{s0}}.64{off}], {{s1}}", [], [base, self.reg32(src)], side=True)''')
    contracts = (ROOT / 'openassembler/backend/contracts.py').read_text()
    contracts = replace_once(contracts, "for width in ('','v2.'): add('st','global.'+cache+width+t,2)", "for width in ('','v2.','v4.'): add('st','global.'+cache+width+t,2)")
    contracts = replace_once(contracts, "    for t in ('u8','s8','u16','s16','b32'", "    for t in ('b8','u8','s8','b16','u16','s16','b64','u64','s64','f64'):\n        for cache in ('','wb.','cg.','cs.','wt.'): add('st','global.'+cache+t,2)\n    for t in ('u8','s8','u16','s16','b32'")
    return {'openassembler/backend/ra.py': text.encode(), 'openassembler/backend/contracts.py': contracts.encode()}


def install():
    import openassembler.backend.ra as ra
    import openassembler.backend.contracts as contracts
    for module in (contracts, ra):
        code = (HERE / ('candidate-' + Path(module.__file__).name)).read_text()
        exec(compile(code, str(HERE / ('candidate-' + Path(module.__file__).name)), 'exec'), module.__dict__)


if __name__ == '__main__':
    result = sources()
    for name, data in result.items():
        (HERE / ('candidate-' + Path(name).name)).write_bytes(data)
    (HERE / 'candidate-identities.json').write_text(json.dumps({
        'baseline': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in result},
        'candidate': {n: hashlib.sha256(data).hexdigest() for n, data in result.items()}}, indent=2) + '\n')
