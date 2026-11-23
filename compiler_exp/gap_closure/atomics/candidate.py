"""Isolated GPU-relaxed integer atomic candidate; no encoder table changes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def generate():
    text = (HERE / 'baseline-ra.py').read_text()
    text = once(text, '        if op == "ld" and "global" in m:\n            return self.load_global(ins, t)',
                '        if op in ("atom", "red"):\n            return self.atomic_global(ins)\n        if op == "ld" and "global" in m:\n            return self.load_global(ins, t)')
    text = once(text, '    def store_global(self, ins, t):', '''    def atomic_global(self, ins):
        """Only admitted GPU-scope relaxed 32-bit integer RMW/reduction forms."""
        returning = ins.op == "atom"
        address = ins.args[1] if returning else ins.args[0]
        base, offset = self._addr(address)
        operation = ins.mods[-2]
        sources = ins.args[2:] if returning else ins.args[1:]
        values = [self.reg32(value) for value in sources]
        if returning:
            result = self.new()
            if operation == "cas":
                self.emit(f"ATOMG.E.CAS.STRONG.GPU PT, {{d0}}, [{{s0}}{offset}], {{s1}}, {{s2}}",
                          [result], [base, *values], var_lat=True, side=True)
            else:
                self.emit(f"ATOMG.E.{operation.upper()}.STRONG.GPU PT, {{d0}}, [{{s0}}.64{offset}], {{s1}}",
                          [result], [base, *values], var_lat=True, side=True)
            self.bind(ins.args[0].name, result)
        else:
            self.emit(f"RED.E.ADD.STRONG.GPU [{{s0}}.64{offset}], {{s1}}", [], [base, *values], side=True)

    def store_global(self, ins, t):''')
    contract = (HERE / 'baseline-contracts.py').read_text()
    contract = once(contract, "    for op in ('ret','bra'):", """    for prefix in ('global.','relaxed.gpu.global.'):
        for dtype in ('u32','s32'):
            add('atom',prefix+'add.'+dtype,3)
            add('red',prefix+'add.'+dtype,2)
        add('atom',prefix+'exch.b32',3)
        add('atom',prefix+'cas.b32',4)
    for op in ('ret','bra'):""")
    contract = once(contract, "    if ins.op in ('ld','st'):", """    if ins.op in ('atom','red'):
        returning = ins.op == 'atom'
        address = a[1] if returning else a[0]
        sources = a[2:] if returning else a[1:]
        if (returning and not isinstance(a[0],Var) or not isinstance(address,Addr)
                or address.off % 4 or any(not isinstance(value,(Var,ImmI)) for value in sources)):
            raise ContractError(f'{name}: requires a register result, naturally aligned address and scalar integer sources')
    elif ins.op in ('ld','st'):""")
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
