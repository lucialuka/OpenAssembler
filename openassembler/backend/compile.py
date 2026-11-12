"""Shared naive PTX selection and production SASS scheduling (sm_80).

Compiler assigns one physical register per PTX virtual register. The naive
A/B backend re-exports this selector with its original conservative scheduler.
Production register allocation lives in backend.ra; it shares Out and the
safe/measured-latency scheduler here with the scalar and tensor pipelines.

Unsupported PTX raises NotSupported. Immediate integer division/remainder uses
the proved strength-reduction sequences from openassembler.opt.divconst.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openassembler.isa.word import Control
from openassembler.opt import divconst
from openassembler.ptx.parse import ImmI, Instr, Kernel, Var, Vec

PARAM_BASE = 0x160

# PTX special register -> (S2R name | c[0x0] offset). Offsets from docs/research/02.
SREG = {"%tid.x": ("s2r", "SR_TID.X"), "%tid.y": ("s2r", "SR_TID.Y"), "%tid.z": ("s2r", "SR_TID.Z"),
        "%ctaid.x": ("s2r", "SR_CTAID.X"), "%ctaid.y": ("s2r", "SR_CTAID.Y"), "%ctaid.z": ("s2r", "SR_CTAID.Z"),
        "%laneid": ("s2r", "SR_LANEID"), "%lanemask_eq": ("s2r", "SR_EQMASK"),
        "%lanemask_lt": ("s2r", "SR_LTMASK"), "%lanemask_ge": ("s2r", "SR_GEMASK"),
        "%ntid.x": ("c0", 0x0), "%ntid.y": ("c0", 0x4), "%ntid.z": ("c0", 0x8),
        "%nctaid.x": ("c0", 0xC), "%nctaid.y": ("c0", 0x10), "%nctaid.z": ("c0", 0x14)}


class NotSupported(Exception):
    pass


@dataclass
class Out:
    text: str
    var_lat: bool = False   # variable latency: needs a write scoreboard
    reads: list = field(default_factory=list)   # physical regs read
    writes: list = field(default_factory=list)  # physical regs written


class Compiler:
    def __init__(self, k: Kernel):
        self.k = k
        self.reg: dict[str, int] = {}
        self.next = 2           # R0/R1 reserved (R1 = stack pointer by convention)
        self.code: list[Out] = []
        self.param_off = {}
        off = 0
        for ty, name in k.params:
            sz = 8 if ty.endswith("64") else 4
            off = (off + sz - 1) // sz * sz
            self.param_off[name] = PARAM_BASE + off
            off += sz
        self.ptr_regs: set[str] = set()

    # ---- registers ---------------------------------------------------------------

    def r(self, name: str, wide: bool = False) -> int:
        if name not in self.reg:
            if wide and self.next % 2:
                self.next += 1
            self.reg[name] = self.next
            self.next += 2 if wide else 1
            if self.next > 250:
                raise NotSupported("out of registers")
        return self.reg[name]

    def opnd(self, op) -> str:
        if isinstance(op, ImmI):
            return f"{op.value & 0xFFFFFFFF:#x}"
        if isinstance(op, Var):
            if op.name.startswith("%") and op.name not in self.reg and op.name.split(".")[0] in (
                    "%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%smid", "%clock", "%clock64"):
                raise NotSupported(f"special register operand {op.name}")
            if op.name not in self.reg and not self._defined_later(op.name):
                raise NotSupported(f"read of undefined register {op.name}")
            return f"R{self.r(op.name)}"
        raise NotSupported(f"operand {op}")

    def _defined_later(self, name: str) -> bool:
        return False

    def emit(self, text, var_lat=False, reads=(), writes=()):
        self.code.append(Out(text, var_lat, list(reads), list(writes)))

    # ---- instruction selection --------------------------------------------------------

    def compile(self) -> list[Out]:
        self.emit("MOV R1, c[0x0][0x28]", writes=[1])
        self.emit("ULDC.64 UR4, c[0x0][0x118]")  # global memory descriptor
        for ins in self.k.body:
            if ins.guard is not None:
                raise NotSupported("predicated PTX")
            self.select(ins)
        return self.code

    def select(self, ins: Instr) -> None:
        op, m, a = ins.op, ins.mods, ins.args
        t = next((x for x in m if x[:1] in ("s", "u", "b", "f") and x[1:].isdigit()), None)
        if op == "ld" and "param" in m:
            d = a[0].name
            off = self.param_off[a[1].base] + a[1].off
            if t in ("u64", "b64", "s64"):
                rd = self.r(d, wide=True)
                self.emit(f"MOV R{rd}, c[0x0][{off:#x}]", writes=[rd])
                self.emit(f"MOV R{rd + 1}, c[0x0][{off + 4:#x}]", writes=[rd + 1])
            else:
                rd = self.r(d)
                self.emit(f"MOV R{rd}, c[0x0][{off:#x}]", writes=[rd])
            return
        if op == "cvta":
            d, s = a[0].name, a[1].name
            self.reg[d] = self.r(s, wide=True)  # global generic == global: alias
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name.startswith("%") and a[1].name in SREG:
            rd = self.r(a[0].name)
            how, what = SREG[a[1].name]
            if how == "s2r":
                self.emit(f"S2R R{rd}, {what}", var_lat=True, writes=[rd])
            else:  # launch constant in c[0x0]
                self.emit(f"MOV R{rd}, c[0x0][{what:#x}]", writes=[rd])
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name.startswith("%") and a[1].name not in self.reg:
            raise NotSupported(f"special register {a[1].name}")
        if op == "mov" and isinstance(a[1], Vec):  # pack {lo, hi} into 64-bit
            rd = self.r(a[0].name, wide=True)
            lo, hi = (self.r(e.name) for e in a[1].elems)
            self.emit(f"MOV R{rd}, R{lo}", reads=[lo], writes=[rd])
            self.emit(f"MOV R{rd + 1}, R{hi}", reads=[hi], writes=[rd + 1])
            return
        if op == "mov" and t in ("b64", "u64", "f64") and isinstance(a[0], Var) and isinstance(a[1], Var):  # alias
            # Only alias a register that already holds a value. A symbol (`mov.u64 %rd, smem`)
            # or never-written name would alias garbage: found on A100 (cvta.shared probe).
            if a[1].name not in self.reg:
                raise NotSupported(f"64-bit mov from symbol/undefined {a[1].name}")
            self.reg[a[0].name] = self.reg[a[1].name]
            return
        if op == "mov" and isinstance(a[0], Vec):  # unpack 64-bit into {lo, hi}
            rs = self.r(a[1].name, wide=True)
            lo, hi = (self.r(e.name) for e in a[0].elems)
            self.emit(f"MOV R{lo}, R{rs}", reads=[rs], writes=[lo])
            self.emit(f"MOV R{hi}, R{rs + 1}", reads=[rs + 1], writes=[hi])
            return
        if op == "mov" and t in ("u32", "b32", "s32", "f32"):
            rd = self.r(a[0].name)
            src = self.opnd(a[1])
            self.emit(f"MOV R{rd}, {src}", reads=self._regs(a[1]), writes=[rd])
            return
        if op == "mul" and "wide" in m:
            rd = self.r(a[0].name, wide=True)
            sfx = ".U32" if t == "u32" else ""
            self.emit(f"IMAD.WIDE{sfx} R{rd}, {self.opnd(a[1])}, {self.opnd(a[2])}, RZ",
                      reads=self._regs(a[1], a[2]), writes=[rd, rd + 1])
            return
        if op == "add" and t in ("s64", "u64"):
            rd = self.r(a[0].name, wide=True)
            x, y = self.r(a[1].name, wide=True), self.r(a[2].name, wide=True)
            self.emit(f"IADD3 R{rd}, P0, R{x}, R{y}, RZ", reads=[x, y], writes=[rd])
            self.emit(f"IADD3.X R{rd + 1}, R{x + 1}, R{y + 1}, RZ, P0, !PT", reads=[x + 1, y + 1], writes=[rd + 1])
            return
        if op == "ld" and "global" in m:
            ad = a[1]
            base = self.r(ad.base, wide=True)
            off = f"+{ad.off:#x}" if ad.off else ""
            if "v4" in m:
                regs = [self.r(e.name) for e in a[0].elems]
                if regs != list(range(regs[0], regs[0] + 4)) or regs[0] % 4:
                    regs = self._fresh_quad(a[0].elems)
                self.emit(f"LDG.E.128 R{regs[0]}, [R{base}.64{off}]", var_lat=True,
                          reads=[base, base + 1], writes=regs)
                return
            if "v2" in m or "v8" in m:
                raise NotSupported("ld.global.v2")
            sfx = {"u8": ".U8", "s8": ".S8", "u16": ".U16", "s16": ".S16", "b32": "", "u32": "", "s32": "",
                   "f32": "", "b64": ".64", "u64": ".64", "s64": ".64", "f64": ".64"}.get(t)
            if sfx is None:
                raise NotSupported(f"ld.global.{t}")
            caching = [x for x in m if x in ("nc", "cg", "cs", "lu", "ca", "cv", "volatile")]
            if caching:
                raise NotSupported(f"cache operator .{caching[0]}")  # needs its own encoding (CONSTANT/EF/LU...)
            rd = self.r(a[0].name, wide=sfx == ".64")
            self.emit(f"LDG.E{sfx} R{rd}, [R{base}.64{off}]", var_lat=True, reads=[base, base + 1],
                      writes=[rd, rd + 1] if sfx == ".64" else [rd])
            return
        if op == "st" and "global" in m:
            ad, src = a[0], a[1]
            base = self.r(ad.base, wide=True)
            off = f"+{ad.off:#x}" if ad.off else ""
            if "v2" in m:
                lo, hi = (self.r(e.name) for e in src.elems)
                pair = self.next + (self.next % 2)
                self.next = pair + 2
                self.emit(f"MOV R{pair}, R{lo}", reads=[lo], writes=[pair])
                self.emit(f"MOV R{pair + 1}, R{hi}", reads=[hi], writes=[pair + 1])
                self.emit(f"STG.E.64 [R{base}.64{off}], R{pair}", reads=[base, base + 1, pair, pair + 1])
                return
            if t not in ("b32", "u32", "s32", "f32"):
                raise NotSupported(f"st.global.{t}")
            self.emit(f"STG.E [R{base}.64{off}], {self.opnd(src)}", reads=[base, base + 1] + self._regs(src))
            return
        if op == "ret":
            self.emit("EXIT")
            return
        if t in ("u32", "s32", "b32"):
            return self.alu32(ins, t)
        raise NotSupported(f"{op}.{'.'.join(m)}")

    def _fresh_quad(self, elems):
        base = (self.next + 3) // 4 * 4
        self.next = base + 4
        for k, e in enumerate(elems):
            self.reg[e.name] = base + k
        return [base + k for k in range(4)]

    def _regs(self, *ops):
        return [self.r(o.name) for o in ops if isinstance(o, Var)]

    def alu32(self, ins, t):
        op, a = ins.op, ins.args
        d = self.r(a[0].name)
        srcs = a[1:]
        x = self.opnd(srcs[0]) if srcs else None
        y = self.opnd(srcs[1]) if len(srcs) > 1 else None
        rr = self._regs(*srcs)
        signed = t == "s32"
        if op in ("div", "rem") and isinstance(srcs[1], ImmI):
            return self.divconst(ins, d, srcs[0], srcs[1].value, signed)
        imm_neg = lambda o: f"{(-o.value) & 0xFFFFFFFF:#x}"  # noqa: E731
        table = {
            "add": lambda: f"IADD3 R{d}, {x}, {y}, RZ",
            "sub": lambda: (f"IADD3 R{d}, {x}, {imm_neg(srcs[1])}, RZ" if isinstance(srcs[1], ImmI)
                            else f"IADD3 R{d}, {x}, -{y}, RZ"),
            "and": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0xc0, !PT",
            "or": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0xfc, !PT",
            "xor": lambda: f"LOP3.LUT R{d}, {x}, {y}, RZ, 0x3c, !PT",
            "not": lambda: f"LOP3.LUT R{d}, RZ, {x}, RZ, 0x33, !PT",
            "neg": lambda: f"IADD3 R{d}, RZ, -{x}, RZ",
            "min": lambda: f"IMNMX{'' if signed else '.U32'} R{d}, {x}, {y}, PT",
            "max": lambda: f"IMNMX{'' if signed else '.U32'} R{d}, {x}, {y}, !PT",
        }
        if op == "mul" and "lo" in ins.mods:
            self.emit(f"IMAD R{d}, {x}, {y}, RZ", reads=rr, writes=[d])
            return
        if op == "mul" and "hi" in ins.mods:
            self.emit(f"IMAD.HI{'' if signed else '.U32'} R{d}, {x}, {y}, RZ", reads=rr, writes=[d])
            return
        if op == "mad" and "lo" in ins.mods:
            self.emit(f"IMAD R{d}, {x}, {y}, {self.opnd(srcs[2])}", reads=rr, writes=[d])
            return
        if op in ("shl", "shr") and isinstance(srcs[1], ImmI) and srcs[1].value >= 32:
            raise NotSupported("shift >= 32")
        if op == "shl" and isinstance(srcs[1], ImmI) and srcs[1].value < 32:
            self.emit(f"SHF.L.U32 R{d}, {x}, {srcs[1].value:#x}, RZ", reads=rr, writes=[d])
            return
        if op == "shr" and isinstance(srcs[1], ImmI) and srcs[1].value < 32:
            kind = "S32" if signed else "U32"
            self.emit(f"SHF.R.{kind}.HI R{d}, RZ, {srcs[1].value:#x}, {x}", reads=rr, writes=[d])
            return
        if op in table:
            self.emit(table[op](), reads=rr, writes=[d])
            return
        raise NotSupported(f"{op}.{t}")

    def divconst(self, ins, d, x_op, divisor, signed):
        if divisor == 0 or (signed and not 2 <= divisor < (1 << 31)) or divisor < 2:
            raise NotSupported("div by 0/1/negative constant")
        seq = divconst.lower(ins.op, signed, divisor)
        xr = self.r(x_op.name)
        names = {"x": f"R{xr}"}
        for o in seq.ops:
            k, dst = o[0], o[1]
            if dst not in names:
                names[dst] = f"R{self.next}"
                self.next += 1
        v = lambda o: f"{o & 0xFFFFFFFF:#x}" if isinstance(o, int) else names[o]  # noqa: E731
        tmp = f"R{self.next}"
        self.next += 1
        for o in seq.ops:
            k, dst = o[0], names[o[1]]
            s = o[2:]
            if k == "lea_hi_sign":
                # q = t + (t >>> 31) as SHF + IADD3 (same result as LEA.HI, proven by the
                # same exhaustive check since openassembler.opt.divconst.run defines the op).
                self.emit(f"SHF.R.U32.HI {tmp}, RZ, 0x1f, {v(s[1])}", reads=[int(v(s[1])[1:])], writes=[int(tmp[1:])])
                self.emit(f"IADD3 {dst}, {v(s[0])}, {tmp}, RZ", reads=[int(v(s[0])[1:]), int(tmp[1:])],
                          writes=[int(dst[1:])])
                continue
            text = {
                "mov": lambda: f"MOV {dst}, {v(s[0])}",
                "mulhi_u": lambda: f"IMAD.HI.U32 {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "mulhi_s": lambda: f"IMAD.HI {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "shr_u": lambda: f"SHF.R.U32.HI {dst}, RZ, {s[1]:#x}, {v(s[0])}",
                "shr_s": lambda: f"SHF.R.S32.HI {dst}, RZ, {s[1]:#x}, {v(s[0])}",
                "add": lambda: f"IADD3 {dst}, {v(s[0])}, {v(s[1])}, RZ",
                "sub": lambda: f"IADD3 {dst}, {v(s[0])}, -{v(s[1])}, RZ",
                "lea_hi_sign": lambda: f"LEA.HI {dst}, {v(s[1])}, {v(s[0])}, RZ, 0x1",
                "mad": lambda: f"IMAD {dst}, {v(s[0])}, {v(s[1])}, {v(s[2])}",
            }[k]()
            regs_read = [int(v(x)[1:]) for x in s if isinstance(x, str)]
            self.emit(text, reads=regs_read, writes=[int(dst[1:])])
        self.emit(f"MOV R{d}, {names[seq.result]}", reads=[int(names[seq.result][1:])], writes=[d])


# ---- scheduling: conservative control bits ---------------------------------------------

MEM_ASYNC = ("LDG", "STG", "LDS", "STS", "LDL", "STL", "ATOMG", "ATOMS", "RED", "LDGSTS", "LDSM",
             "MUFU", "SHFL", "F2I")
# F2I collects its source asynchronously, independently of result completion.
# Runtime integer div/rem fragments can overwrite that source immediately after
# conversion. A write scoreboard alone protects the result, not this WAR edge.
# The A100 F2I-only ablation in exp/native_indexing fixes the original large-grid
# indexing failure without changing the arithmetic, operands or instruction order.


def _opname(text: str) -> str:
    t = text.split()
    return (t[1] if t[0].startswith("@") else t[0]).split(".")[0]


def _pdefs(text: str) -> set:
    import re
    body = re.sub(r"^@!?P\d ", "", text)
    op, _, rest = body.partition(" ")
    args = [x.strip() for x in rest.split(",")]
    count = 2 if re.search(r"SETP|PLOP3|R2P|IADD3|LEA|LOP3", op) else 1 if op.startswith("FCHK") else 0
    return {x for x in args[:count] if re.fullmatch(r"P[0-6]", x)}


def _puses(text: str) -> set:
    import re
    g = re.match(r"@!?(P[0-6]) ", text)
    body = re.sub(r"^@!?P\d ", "", text)
    op, _, rest = body.partition(" ")
    args = [x.strip().lstrip("!") for x in rest.split(",")]
    count = 2 if re.search(r"SETP|PLOP3|R2P|IADD3|LEA|LOP3", op) else 1 if op.startswith("FCHK") else 0
    return {x for x in args[count:] if re.fullmatch(r"P[0-6]", x)} | ({g[1]} if g else set())


def schedule(code: list[Out], labels: dict | None = None, model: str = "safe") -> list[tuple[str, Control]]:
    """In-order, no reordering.
    * Variable-latency writes (loads, S2R): write scoreboard; readers and over-writers wait.
    * Memory and operand-collection ops read their source registers asynchronously:
      read scoreboard; later
      writers of those registers wait (WAR) - required once registers are reused.
    * Control flow: scoreboard state is not tracked across joins, so the instruction at
      every label and every branch waits on all outstanding scoreboards.
    * Fixed-latency results:
        model="safe"   stall 6 after every op (above every latency mined from ptxas)
        model="mined"  stall = exact distance to the next dependent consumer, using the
                       per-class minimum latencies mined from ptxas (tools/mine_latency.py
                       -> corpus/sm_80/machine_model.json) plus the mined issue gaps.
    """
    out = []
    pending_w: dict[int, int] = {}
    pending_r: dict[int, set[int]] = {}
    next_sb = 0
    join_at = set((labels or {}).values())

    def take():
        nonlocal next_sb
        b = next_sb
        next_sb = (next_sb + 1) % 6
        return b

    for i, o in enumerate(code):
        wait = 0
        live = set(pending_w.values()) | {b for bars in pending_r.values() for b in bars}
        if i in join_at or _opname(o.text) in ("BRA", "EXIT", "WARPSYNC"):
            for b in live:
                wait |= 1 << b
        pred_reads = [("p", p) for p in _puses(o.text)]
        pred_writes = [("p", p) for p in _pdefs(o.text)]
        for r in o.reads + o.writes + pred_reads + pred_writes:
            if r in pending_w:
                wait |= 1 << pending_w[r]
        for r in o.writes:
            for b in pending_r.get(r, ()):
                wait |= 1 << b
        # Select and retire both old slot generations before publishing this
        # instruction's mappings. A read-barrier allocation must not erase a
        # just-published write mapping when its old generation is also waited.
        wbar = take() if o.var_lat else 7
        rbar = take() if _opname(o.text) in MEM_ASYNC and o.reads else 7
        for b in (wbar, rbar):
            if b in live:
                wait |= 1 << b
        pending_w = {r: b for r, b in pending_w.items() if not (wait & (1 << b))}
        retired = {b for b in range(6) if wait & (1 << b)}
        pending_r = {r: bars - retired for r, bars in pending_r.items() if bars - retired}
        if wbar != 7:
            for r in o.writes + pred_writes:
                pending_w[r] = wbar
        if rbar != 7:
            for r in o.reads:
                pending_r.setdefault(r, set()).add(rbar)
        stall = 2 if o.var_lat else 6
        if _opname(o.text) in ("STG", "EXIT", "BRA", "WARPSYNC"):
            stall = 5
        out.append((o.text, Control(stall=stall, yield_=False, wbar=wbar, rbar=rbar, wait=wait, reuse=0)))
    if model == "mined":
        out = _mined_stalls(code, out, join_at)
    return _branch_pred_latency(out)


PRED_TO_GUARD = 13   # SETP/PLOP3 -> any @P-guarded op (A100: tools/lat_sweep.py *.g; = ptxas's minimum)


def _branch_pred_latency(out):
    """An instruction guarded by @Px reads Px at issue: on the A100 a guard 6..12 cycles
    after its ISETP/FSETP used the OLD predicate (loops ran one extra iteration; guarded
    IADD3/IMAD/FADD kept stale results), while the emulator, which has no timing, was
    right. Measured minimum: 13 cycles for every guarded opcode tried. A predicate read as
    a source operand (SEL ..., P0) needs only 4. Enforce 13 by raising stalls."""
    import re

    out = list(out)
    for i, (t, c) in enumerate(out):
        m = re.match(r"@!?(P\d) ", t)
        if not m:
            continue
        for j in range(i - 1, -1, -1):
            body = re.sub(r"^@!?P\d ", "", out[j][0])
            ops = body.split(None, 1)
            if len(ops) < 2:
                continue
            dsts = [x.strip() for x in ops[1].split(",")[:2]]
            if m[1] in _pdefs(out[j][0]):
                gap = sum(out[q][1].stall for q in range(j, i))
                q = i - 1
                while gap < PRED_TO_GUARD and q >= j:
                    t2, c2 = out[q]
                    add = min(15 - c2.stall, PRED_TO_GUARD - gap)
                    # stall 12..15 is only legal with yield set (word.control_valid); ptxas
                    # writes "S13 Y" in exactly this spot
                    out[q] = (t2, _dc_replace(c2, stall=c2.stall + add,
                                              yield_=c2.yield_ or c2.stall + add > 11))
                    gap += add
                    q -= 1
                break
    return out


# Fixed-latency pipes on sm_80, from the A100 sweep (corpus/sm_80/hw_latency.json):
# producer -> consumer is 4 cycles inside a pipe and 5 across pipes.
_FMA_PIPE = ("IMAD", "FFMA", "FADD", "FMUL", "HFMA2", "HADD2", "HMUL2")
_ALU_PIPE = ("IADD3", "LOP3", "SHF", "IMNMX", "MOV", "SEL", "ISETP", "FSETP", "FMNMX", "FSEL",
             "LEA", "PRMT", "PLOP3", "IABS", "SGXT", "BMSK", "FLO", "POPC", "BREV")


def _pipe(op: str):
    base = op.split(".")[0]
    if base in _FMA_PIPE:
        return "fma"
    if base in _ALU_PIPE:
        return "alu"
    return None


def hw_latency(prod: str, cons: str) -> int:
    """RAW latency (cycles) from fixed-latency `prod` to `cons` (opcode names). Falls back
    to 6 for producers outside the measured pipes. Exceptions measured on the A100:
    FMNMX (alu) -> FADD (fma) is 5 like any cross-pipe pair."""
    p, c = _pipe(prod), _pipe(cons)
    if p is None:
        return 6
    if prod.startswith("IMAD") and cons.split(".")[0] in ("FFMA", "FADD", "FMUL"):
        return 5      # int <-> fp inside the fma pipe: not measured; assume cross-pipe
    if prod.split(".")[0] in ("FFMA", "FADD", "FMUL") and cons.startswith("IMAD"):
        return 5
    return 4 if p == c else 5


def _dc_replace(c, **kw):
    import dataclasses
    return dataclasses.replace(c, **kw)


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        m = json.loads((root / "corpus" / "sm_80" / "machine_model.json").read_text())
        lat = {c: v["fixed_min"] for c, v in m["classes"].items() if v["fixed_min"] is not None}
        _MODEL = (lat, m["pairs"])
    return _MODEL


def _mined_stalls(code: list[Out], sched, join_at) -> list:
    """Replace the fixed stall-6 with distance-to-next-dependent-consumer, using the
    mined minimum latency of each producer class (and per-pair when known).
    Conservative where the model has no data: falls back to 6."""
    import sys
    from pathlib import Path

    from openassembler.emu.program import parse_insn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
    from mine_latency import op_class

    lat, pairs = _load_model()
    cls = [op_class(parse_insn(0, o.text, 0)) for o in code]
    issue = [0]
    for i in range(1, len(code)):
        t = issue[i - 1] + 1
        # dependence on any earlier fixed-latency producer of our reads (scoreboarded ones excluded)
        for j in range(i - 1, max(-1, i - 16), -1):
            if sched[j][1].wbar != 7:
                continue
            if set(code[j].writes) & set(code[i].reads + code[i].writes) or \
                    _pdefs(code[j].text) & _puses(code[i].text):
                L = hw_latency(_opname(code[j].text), _opname(code[i].text))
                t = max(t, issue[j] + L)
        if i in join_at or _opname(code[i - 1].text) in ("BRA", "EXIT"):
            # unknown predecessor across a join: every fixed-latency result issued before
            # the block boundary must be complete (max measured latency 5; 6 otherwise)
            t = max(t, issue[i - 1] + 1)
            for j in range(i - 1, max(-1, i - 16), -1):
                if sched[j][1].wbar == 7 and code[j].writes:
                    t = max(t, issue[j] + max(hw_latency(_opname(code[j].text), "IMAD"),
                                              hw_latency(_opname(code[j].text), "IADD3")))
        issue.append(t)
    out = []
    for i, (txt, c) in enumerate(sched):
        st = min(15, issue[i + 1] - issue[i]) if i + 1 < len(sched) else c.stall
        st = max(st, 2) if (c.wbar != 7 or c.rbar != 7) else max(st, 1)
        if _opname(txt) in ("BRA", "EXIT"):
            st = max(st, 5)
        out.append((txt, Control(stall=st, yield_=c.yield_, wbar=c.wbar, rbar=c.rbar, wait=c.wait, reuse=0)))
    return out
