"""Generate standalone selector, admission and exact-oracle memory data loading."""
import builtins
from pathlib import Path

HERE=Path(__file__).resolve().parent


def once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new,1)


def generate():
    ra=(HERE/'baseline-ra.py').read_text()
    old='''        if t not in ("b32", "u32", "s32", "f32"):
            raise NotSupported(f"{op}.shared.{t}")
        if op == "ld":'''
    new='''        if t in ("b8","u8","s8","b16","u16","s16","b64","u64","s64","f64"):
            if nvec!=1:raise NotSupported("new shared widths require scalar memory operands")
            width=8 if t.endswith("64")else 2 if t.endswith("16")else 1
            if off%width:raise NotSupported("shared literal offset requires natural alignment")
            if ad.base in self.sym_shared:
                declaration=next(d for d in self.k.decls if d.space=="shared"and d.name==ad.base)
                if ad.off<0 or ad.off+width>declaration.size or declaration.align<width:
                    raise NotSupported("shared symbol access exceeds its declared size/alignment")
            suffix={"b8":".U8","u8":".U8","s8":".S8","b16":".U16","u16":".U16","s16":".S16"}.get(t,".64")
            if op=="ld":
                destination=tuple(self.new()for _ in range(2))if width==8 else self.new()
                self.emit(f"LDS{suffix} {{d0}}, {addr}",[destination],uses0,var_lat=True)
                self.bind(a[0].name,destination)
            else:
                value=self.wide(self.operand64(a[1]))if width==8 else self.reg32(a[1])
                self.emit(f"STS{suffix} {addr}, {{s{si}}}",[],uses0+[value],side=True)
            return
        if t not in ("b32", "u32", "s32", "f32"):
            raise NotSupported(f"{op}.shared.{t}")
        if op == "ld":'''
    ra=once(ra,old,new)
    old='''        addr = f"[{{s0}}{'+' + hex(off) if off else ''}]" if base is not None else f"[RZ+{off:#x}]"'''
    new='''        offset=("+" if off>0 else "-")+hex(abs(off)) if off else ""
        addr = f"[{{s0}}{offset}]" if base is not None else f"[RZ+{off:#x}]"'''
    ra=once(ra,old,new)
    contracts=(HERE/'baseline-contracts.py').read_text()
    anchor="    for t in ('b8','u8','s8','b16','u16','s16','b64','u64','s64','f64'):\n"
    contracts=once(contracts,anchor,"    for t in ('b8','u8','s8','b16','u16','s16','b64','u64','s64','f64'):\n        for op in ('ld','st'):add(op,'shared.'+t,2)\n"+anchor)
    encode=(HERE/'baseline-encode.py').read_text()
    anchor='        for k, members in groups.items():\n'
    encode=once(encode,anchor,'''        memory = ROOT / "specs" / arch / "memory_templates.json"
        if memory.exists():
            for r in json.loads(memory.read_text()):
                w = int(r["word"],16)
                ins = parse_insn(0,r["text"],w)
                groups[key(ins)][w] = ins
''' +anchor)
    compile_source=(HERE/'baseline-compile.py').read_text()
    old='''        if _opname(txt) in ("BRA", "EXIT"):
            st = max(st, 5)'''
    new=old+'''
        if _opname(txt)=="BAR" and any(_opname(o.text) in ("LDS","STS") and ".64 " in o.text for o in code):
            # Shared 64-bit exchanges require the observed barrier issue gap.
            # Retain the safe scheduler/ptxas gap rather than the scalar model's 1.
            st = max(st, 6)'''
    compile_source=once(compile_source,old,new)
    return dict(ra=ra,contracts=contracts,encode=encode,compile=compile_source)


def install():
    import openassembler.backend.ra as ra
    import openassembler.backend.contracts as contracts
    import openassembler.isa.encode as encode
    import openassembler.backend.compile as compiler
    import openassembler.build as build
    for name,module in(('encode',encode),('compile',compiler),('contracts',contracts),('ra',ra)):
        path=HERE/f'candidate-{name}.py';exec(builtins.compile(path.read_text(),str(path),'exec'),module.__dict__)
    # The isolated candidate reads its own exact data; normal production later
    # reads the same generated rows from specs/sm_80/memory_templates.json.
    from openassembler.emu.program import parse_insn
    import json
    encoder=encode.Encoder('sm_80')
    for row in json.loads((HERE/'candidate-templates.json').read_text()):
        ins=parse_insn(0,row['text'],int(row['word'],16));template=encoder._learn([ins],canonical=False)
        assert template and all(field is not None for _,field in template.slots)
        encoder.templates[encode.key(ins)]=template
    build._ENCODERS.clear();build._ENCODERS['sm_80']=encoder


if __name__=='__main__':
    for name,source in generate().items():(HERE/f'candidate-{name}.py').write_text(source)
