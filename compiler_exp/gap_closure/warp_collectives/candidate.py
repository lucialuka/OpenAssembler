"""Full-mask scalar shuffle candidate using only existing recovered templates."""
from pathlib import Path
import builtins
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def generate():
    ra=(HERE/'baseline-ra.py').read_text()
    start=ra.index('        if op == "shfl":')
    end=ra.index('        if op == "mov" and isinstance(a[1], Var) and a[1].name in self.sym_shared:',start)
    original_shuffle=ra[start:end]
    ra=ra[:start]+'''        if op == "shfl" and m[1] != "bfly":
            mode=m[1]
            clamp=0 if mode=="up" else 31
            if (len(a)!=5 or not isinstance(a[0],Var) or ins.guard is not None
                    or not isinstance(a[3],ImmI) or a[3].value!=clamp
                    or not isinstance(a[4],ImmI) or a[4].value!=0xffffffff):
                raise NotSupported("shuffle requires an unpredicated full warp, full mask, and clamp 0 for up / 31 otherwise")
            source,delta=self.reg32(a[1]),self.reg32(a[2])
            mask=self.reg32(ImmI(0xffffffff))
            self.emit("WARPSYNC {s0}",[],[mask],side=True)
            if mode=="up":
                # PTX b uses only its low five bits. For an invalid source,
                # select our own lane before the full-warp indexed shuffle.
                lane,bits,index,selected=self.new(),self.new(),self.new(),self.new()
                valid=self.newp()
                self.emit("S2R {d0}, SR_LANEID",[lane],[],var_lat=True)
                self.emit("LOP3.LUT {d0}, {s0}, 0x1f, RZ, 0xc0, !PT",[bits],[delta])
                self.emit("IADD3 {d0}, {s0}, -{s1}, RZ",[index],[lane,bits])
                self.emit("ISETP.GE.U32.AND {p0}, PT, {s0}, {s1}, PT",[valid],[lane,bits])
                self.emit("SEL {d0}, {s0}, {s1}, {q0}",[selected],[index,lane,(valid,False)])
                delta=selected
            result=self.new()
            opcode="IDX" if mode=="up" else mode.upper()
            self.emit(f"SHFL.{opcode} PT, {{d0}}, {{s0}}, {{s1}}, 0x1f",[result],[source,delta],var_lat=True)
            self.bind(a[0].name,result)
            return
''' +original_shuffle+ra[end:]
    contracts=(HERE/'baseline-contracts.py').read_text()
    old="    add('shfl','sync.bfly.b32',5)"
    assert contracts.count(old)==1
    contracts=contracts.replace(old,"    for mode in ('bfly','idx','up','down'):add('shfl','sync.'+mode+'.b32',5)")
    return {'ra':ra,'contracts':contracts}


def install():
    import openassembler.backend.ra as ra
    import openassembler.backend.contracts as contracts
    for module in (contracts,ra):
        source=HERE/('candidate-'+Path(module.__file__).name)
        exec(builtins.compile(source.read_text(),str(source),'exec'),module.__dict__)


if __name__=='__main__':
    generated=generate()
    for name,source in generated.items():(HERE/f'candidate-{name}.py').write_text(source)
    (HERE/'identities.json').write_text(json.dumps({
        'baseline':{name:hashlib.sha256((HERE/f'baseline-{name}.py').read_bytes()).hexdigest()for name in generated},
        'candidate':{name:hashlib.sha256(source.encode()).hexdigest()for name,source in generated.items()}},indent=2)+'\n')
