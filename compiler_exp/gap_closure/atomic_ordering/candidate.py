"""Isolated ordering candidate: retain exact scopes, fences and scoreboard order."""
from pathlib import Path
import hashlib
import json
import builtins

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def generate():
    ra=(HERE/'baseline-ra.py').read_text()
    ra=once(ra,'''        returning = ins.op == "atom"
        address = ins.args[1] if returning else ins.args[0]
        base, offset = self._addr(address)''','''        returning = ins.op == "atom"
        address = ins.args[1] if returning else ins.args[0]
        shared = "shared" in ins.mods
        if shared:
            if address.base in self.sym_shared:
                base = self.reg32(ImmI(self.sym_shared[address.base]))
            else:
                base = self.reg32(Var(address.base))
            offset = (("+" if address.off>0 else "-")+hex(abs(address.off))) if address.off else ""
        else:
            base, offset = self._addr(address)
        semantics = ins.mods[0]
        if semantics in ("release","acq_rel"):
            self.emit("MEMBAR.ALL.GPU", [], [], side=True)
            self.emit("ERRBAR", [], [], side=True)''')
    ra=once(ra,'''            if operation == "cas":''','''            if shared:
                self.emit(f"ATOMS.{operation.upper()} {{d0}}, [{{s0}}{offset}], {{s1}}",
                          [result], [base,*values], var_lat=True, side=True)
            elif operation == "cas":''')
    ra=once(ra,'''            self.bind(ins.args[0].name, result)
        else:
            self.emit(f"RED.E.ADD.STRONG.GPU''','''            self.bind(ins.args[0].name, result)
            if semantics in ("acquire","acq_rel"):
                self.emit("CCTL.IVALL", [], [], side=True)
        else:
            self.emit(f"RED.E.ADD.STRONG.GPU''')
    contract=(HERE/'baseline-contracts.py').read_text()
    contract=once(contract,"    for op in ('ret','bra'):","""    for semantics in ('acquire','release','acq_rel'):
        prefix=semantics+'.gpu.global.'
        for dtype in ('u32','s32'):add('atom',prefix+'add.'+dtype,3)
        add('atom',prefix+'exch.b32',3)
        add('atom',prefix+'cas.b32',4)
    for prefix in ('shared.','relaxed.cta.shared.'):
        for dtype in ('u32','s32'):add('atom',prefix+'add.'+dtype,3)
        add('atom',prefix+'exch.b32',3)
    for op in ('ret','bra'):""")
    compile=(HERE/'baseline-compile.py').read_text()
    compile=once(compile,'("BRA", "EXIT", "WARPSYNC"):', '("BRA", "EXIT", "WARPSYNC", "MEMBAR", "ERRBAR", "CCTL"):')
    sched=(HERE/'baseline-sched.py').read_text()
    sched=once(sched,'"MEMBAR", "DEPBAR", "S2R", "CS2R", "WARPSYNC", "SHFL")',
               '"MEMBAR", "ERRBAR", "CCTL", "DEPBAR", "S2R", "CS2R", "WARPSYNC", "SHFL")')
    return {'openassembler/backend/ra.py':ra.encode(),'openassembler/backend/contracts.py':contract.encode(),
            'openassembler/backend/compile.py':compile.encode(),'openassembler/backend/sched.py':sched.encode()}


def install():
    import openassembler.backend.ra as ra
    import openassembler.backend.contracts as contracts
    import openassembler.backend.compile as compile
    import openassembler.backend.sched as sched
    for module in (contracts,compile,sched,ra):
        path=HERE/('candidate-'+Path(module.__file__).name)
        exec(builtins.compile(path.read_text(),str(path),'exec'),module.__dict__)


if __name__=='__main__':
    files=generate()
    for name,data in files.items():(HERE/('candidate-'+Path(name).name)).write_bytes(data)
    (HERE/'candidate-identities.json').write_text(json.dumps({
        'baseline':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        'candidate':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}},indent=2)+'\n')
