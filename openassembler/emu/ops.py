"""Instruction semantics.

Integer semantics follow Mesa NAK's constant folder (nak/ir.rs `fold()`), which
upstream tests against hardware. Where NAK has no folder or the corpus uses a
form NAK doesn't model, the rule is marked HYPOTHESIS and must be confirmed by
the differential harness (tools/diff_emu.py) before it's relied on.

Handler kinds:
  @alu    per-thread, runs only where the guard is true; pc advances by 16
  @warp   gets every guard-true thread of the converged group at once
  @ctrl   per-thread, sets the thread's pc itself
"""

from __future__ import annotations

import dataclasses as _dc

from typing import Callable

from openassembler.emu.machine import M32, EmuError, Machine, Thread, Unsupported, Warp, special_reg
from openassembler.emu.operands import Imm, Label, Pred, Reg
from openassembler.emu.program import Insn

ALU: dict[str, Callable] = {}
WARP: dict[str, Callable] = {}
CTRL: dict[str, Callable] = {}
NOPS = {"NOP", "BSSY", "BSYNC", "BREAK", "WARPSYNC", "DEPBAR", "LDGDEPBAR", "ERRBAR", "MEMBAR",
        "CCTL", "NANOSLEEP", "PMTRIG", "BMOV", "YIELD", "FENCE"}


def alu(*names):
    def reg(fn):
        for n in names:
            ALU[n] = fn
        return fn
    return reg


def warp(*names):
    def reg(fn):
        for n in names:
            WARP[n] = fn
        return fn
    return reg


def ctrl(*names):
    def reg(fn):
        for n in names:
            CTRL[n] = fn
        return fn
    return reg


def execute(m: Machine, wp: Warp, group: list[Thread], insn: Insn) -> None:
    op = insn.op
    if op in NOPS:
        for th in group:
            th.pc += 16
        return
    if op in CTRL:
        for th in group:
            if m.guard(th, wp, insn):
                CTRL[op](m, th, wp, insn)
            else:
                th.pc += 16
        return
    if op in WARP:
        active = [th for th in group if m.guard(th, wp, insn)]
        if active:
            WARP[op](m, wp, active, insn)
        for th in group:
            th.pc += 16
        return
    if op in ALU:
        # Uniform datapath (UR/UP destination): one execution per warp. Running it per
        # thread made read-modify-write ops like "UIADD3 UR4, UR4, -0x3" apply 32 times
        # (found when ptxas's unrolled bench loops never terminated).
        d = insn.ops[0] if insn.ops else None
        uniform = getattr(d, "file", None) in ("UR", "UP")
        done = False
        for th in group:
            if (not uniform or not done) and m.guard(th, wp, insn):
                ALU[op](m, th, wp, insn)
                done = True
            th.pc += 16
        return
    raise Unsupported(f"{insn.text!r} at {insn.addr:#x}")


# ---- helpers -------------------------------------------------------------------

def s32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v & 0x80000000 else v


def u(v: int, bits: int = 32) -> int:
    return v & ((1 << bits) - 1)


def lut3(lut: int, a: int, b: int, c: int, mask: int = M32) -> int:
    """LOP3 truth table: bit i of `lut` is the output for (a,b,c) = bits of i (a = MSB).
    Equivalent to NAK's LogicOp3::eval; lut 0xF0 = a, 0xCC = b, 0xAA = c."""
    r = 0
    for i in range(8):
        if lut >> i & 1:
            r |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
    return r & mask


def bool_op(insn: Insn, x: bool, acc: bool) -> bool:
    if insn.has("AND"):
        return x and acc
    if insn.has("OR"):
        return x or acc
    if insn.has("XOR"):
        return x != acc
    raise Unsupported(f"boolean op in {insn.text}")


def int_cmp(insn: Insn, a: int, b: int, signed: bool) -> bool:
    if signed:
        a, b = s32(a), s32(b)
    for name, fn in (("EQ", a == b), ("NE", a != b), ("LT", a < b), ("LE", a <= b),
                     ("GT", a > b), ("GE", a >= b)):
        if name in insn.mods:
            return fn
    if "F" in insn.mods:
        return False
    if "T" in insn.mods:
        return True
    raise Unsupported(f"compare op in {insn.text}")


def split_pred_dsts(ops: list, n_max: int) -> tuple[list[Pred], list]:
    """Pop leading predicate *destinations* that follow the register dst."""
    preds = []
    while ops and isinstance(ops[0], Pred) and len(preds) < n_max:
        preds.append(ops.pop(0))
    return preds, ops


def no_pred_src(p) -> None:
    # HYPOTHESIS: a trailing predicate on LOP3/IMNMX-style ops other than !PT has
    # meaning we haven't measured yet. Fail loudly instead of guessing.
    if not (isinstance(p, Pred) and p.idx == 7 and p.neg):
        raise Unsupported(f"non-!PT predicate source {p}")


# ---- moves, special registers, constants ---------------------------------------

@alu("MOV", "UMOV")
def _mov(m, th, wp, insn):
    d, a = insn.ops[0], insn.ops[1]
    if len(insn.ops) > 2 and not (isinstance(insn.ops[2], Imm) and insn.ops[2].int32() == 0xF):
        raise Unsupported(f"MOV with lane mask {insn.text}")
    m.wr(th, wp, d, m.src(th, wp, a))


@alu("S2R", "S2UR")
def _s2r(m, th, wp, insn):
    m.wr(th, wp, insn.ops[0], special_reg(m, th, wp, insn.ops[1].text))


@alu("CS2R")
def _cs2r(m, th, wp, insn):
    name = insn.ops[1].text
    width = 32 if insn.has("32") else 64
    if name == "SRZ":
        v = 0
    elif name in ("SR_CLOCKLO", "SR_GLOBALTIMERLO"):
        v = m.steps  # any monotonic value is a valid clock
    else:
        raise Unsupported(name)
    m.wr(th, wp, insn.ops[0], v, width=width)


@alu("ULDC", "LDC")
def _ldc(m, th, wp, insn):
    width = 64 if insn.has("64") else 32
    m.wr(th, wp, insn.ops[0], m.rd(th, wp, insn.ops[1], width=width), width=width)


# ---- integer arithmetic ---------------------------------------------------------

@alu("IADD3", "UIADD3")
def _iadd3(m, th, wp, insn):
    ops = list(insn.ops)
    d = ops.pop(0)
    couts, ops = split_pred_dsts(ops, 2)
    a, b, c = ops[:3]
    if insn.has("X"):
        # NAK OpIAdd3X: '~' allowed on sources, plus two carry-in predicates.
        total = sum(m.src(th, wp, x) for x in (a, b, c))
        total += sum(m.prd(th, wp, p) for p in ops[3:5])
    else:
        # NAK OpIAdd3: negation counts as (~x + 1) in the 34-bit sum, not as -x.
        total = 0
        for x in (a, b, c):
            v = m.rd(th, wp, x)
            total += ((~v & M32) + 1) if getattr(x, "neg", False) else v
    m.wr(th, wp, d, total & M32)
    for k, p in enumerate(couts):
        m.pwr(th, wp, p, total >= (k + 1) << 32)


@alu("IMAD", "UIMAD")
def _imad(m, th, wp, insn):
    ops = list(insn.ops)
    d = ops.pop(0)
    couts, ops = split_pred_dsts(ops, 1)
    a, b, c = ops[:3]
    signed = not insn.has("U32")
    av, bv = m.src(th, wp, a), m.src(th, wp, b)
    if insn.has("WIDE"):
        if signed:
            av, bv = s32(av), s32(bv)
        cv = m.rd(th, wp, c, width=64)
        if getattr(c, "neg", False):
            cv = -cv
        r = (av * bv + cv) & ((1 << 64) - 1)
        m.wr(th, wp, d, r, width=64)
        return
    if insn.has("HI"):
        # C is a 64-bit register pair, like IMAD.WIDE: d = hi32(a*b + C64).
        # Measured: ptxas lowers mad.hi.s32 d, a, b, c to "R8 = 0; R9 = c;
        # IMAD.HI d, a, b, R8" (corpus probe mad.hi.s32.r).
        if signed:
            av, bv = s32(av), s32(bv)
        cv = m.rd(th, wp, c, width=64)
        r = ((av * bv + cv) & ((1 << 64) - 1)) >> 32
        m.wr(th, wp, d, r & M32)
        return
    r = av * bv + m.src(th, wp, c)
    if insn.has("X"):
        r += m.prd(th, wp, ops[3])
    m.wr(th, wp, d, r & M32)
    if couts:
        m.pwr(th, wp, couts[0], (u(av * bv) + m.src(th, wp, c)) >= 1 << 32)


@alu("LEA", "ULEA")
def _lea(m, th, wp, insn):
    # NAK OpLea / OpLeaX. Text forms:
    #   LEA Rd, [Pc,] a, b, shift                  d = (a << s) + b
    #   LEA.HI Rd, [Pc,] a, b, a_hi, shift         d = ((a_hi:a) >> (32 - s)) + b
    #   .X adds a trailing carry-in predicate and uses '~' instead of '-'
    ops = list(insn.ops)
    d = ops.pop(0)
    couts, ops = split_pred_dsts(ops, 1)
    x = insn.has("X")
    cin = m.prd(th, wp, ops.pop()) if x else 0
    a, b = ops[0], ops[1]
    shift = ops[-1].int32()
    av = m.rd(th, wp, a)
    if insn.has("HI"):
        hi = m.rd(th, wp, ops[2])
        sr = (((hi << 32) | av) >> (32 - shift)) & M32
    else:
        sr = (av << shift) & M32
    bv = m.rd(th, wp, b)
    if x:
        sr = ~sr & M32 if getattr(a, "bnot", False) else sr
        bv = ~bv & M32 if getattr(b, "bnot", False) else bv
        total = sr + bv + cin
        ovf = total >= 1 << 32
    else:
        ovf = False
        if getattr(a, "neg", False):
            sr, o = (~sr & M32) + 1, False
            ovf |= sr >= 1 << 32
            sr &= M32
        if getattr(b, "neg", False):
            bv = (~bv & M32) + 1
            ovf |= bv >= 1 << 32
            bv &= M32
        total = sr + bv
        ovf |= total >= 1 << 32
    m.wr(th, wp, d, total & M32)
    if couts:
        m.pwr(th, wp, couts[0], ovf)


@alu("IABS")
def _iabs(m, th, wp, insn):
    m.wr(th, wp, insn.ops[0], abs(s32(m.src(th, wp, insn.ops[1]))) & M32)


@alu("IMNMX", "UIMNMX")
def _imnmx(m, th, wp, insn):
    d, a, b, p = insn.ops
    av, bv = m.src(th, wp, a), m.src(th, wp, b)
    if not insn.has("U32"):
        av, bv = s32(av), s32(bv)
    m.wr(th, wp, d, (min if m.prd(th, wp, p) else max)(av, bv) & M32)


@alu("SEL", "USEL")
def _sel(m, th, wp, insn):
    d, a, b, p = insn.ops
    m.wr(th, wp, d, m.src(th, wp, a) if m.prd(th, wp, p) else m.src(th, wp, b))


@alu("FSEL")
def _fsel(m, th, wp, insn):
    # Like SEL, but immediates are float literals (e.g. "0.25") and |x| / -x apply.
    d, a, b, p = insn.ops
    x = a if m.prd(th, wp, p) else b
    if isinstance(x, Imm):
        m.wr(th, wp, d, fp.encode(fsrc(m, th, wp, x, insn)))
        return
    # Register/cbank sources are selected as raw bits: no NaN canonicalization
    # (A100: ptxas uses FSEL on f64 high words in min/max.f64).
    v = m.rd(th, wp, x)
    if getattr(x, "abs", False):
        v &= 0x7FFFFFFF
    if getattr(x, "neg", False):
        v ^= 0x80000000
    m.wr(th, wp, d, v)


@alu("ISETP", "UISETP")
def _isetp(m, th, wp, insn):
    # ISETP.cmp[.U32].bop[.EX] Pu, Pv, a, b, acc[, low]
    pu, pv, a, b, acc = insn.ops[:5]
    av, bv = m.src(th, wp, a), m.src(th, wp, b)
    cmp = int_cmp(insn, av, bv, signed=not insn.has("U32"))
    if insn.has("EX") and av == bv:
        cmp = m.prd(th, wp, insn.ops[5])      # NAK: equal high words defer to the low compare
    accv = m.prd(th, wp, acc)
    m.pwr(th, wp, pu, bool_op(insn, cmp, accv))
    # HYPOTHESIS: the second destination gets the inverted compare, combined the same way.
    m.pwr(th, wp, pv, bool_op(insn, not cmp, accv))


@alu("LOP3", "ULOP3")
def _lop3(m, th, wp, insn):
    ops = list(insn.ops)
    pd = ops.pop(0) if isinstance(ops[0], Pred) else None
    d, a, b, c, lut, pin = ops
    r = lut3(lut.int32(), m.src(th, wp, a), m.src(th, wp, b), m.src(th, wp, c))
    no_pred_src(pin)
    m.wr(th, wp, d, r)
    if pd is not None:
        m.pwr(th, wp, pd, r != 0)  # HYPOTHESIS: predicate output = (result != 0)


@alu("PLOP3", "UPLOP3")
def _plop3(m, th, wp, insn):
    pd, pd2, a, b, c, lut, lut2 = insn.ops
    av, bv, cv = (int(m.prd(th, wp, p)) for p in (a, b, c))
    m.pwr(th, wp, pd, bool(lut3(lut.int32(), av, bv, cv, 1)))
    m.pwr(th, wp, pd2, bool(lut3(lut2.int32(), av, bv, cv, 1)))


@alu("SHF", "USHF")
def _shf(m, th, wp, insn):
    # SHF.{L,R}[.W].{U32,S32,U64,S64}[.HI] Rd, lo, shift, hi   (NAK OpShf, sm >= 70)
    d, lo, sh, hi = insn.ops
    bits = 64 if (insn.has("U64") or insn.has("S64")) else 32
    s = m.src(th, wp, sh)
    s = s & (bits - 1) if insn.has("W") else min(s, bits)
    x = m.rd(th, wp, lo) | (m.rd(th, wp, hi) << 32)
    signed = insn.has("S32") or insn.has("S64")
    if insn.has("R"):
        if signed:
            x -= (1 << 64) if x >> 63 else 0
        r = x >> s
    else:
        r = x << s
    r &= (1 << 64) - 1
    m.wr(th, wp, d, (r >> 32) if insn.has("HI") else r & M32)


@alu("PRMT", "UPRMT")
def _prmt(m, th, wp, insn):
    d, a, sel, b = insn.ops
    if set(insn.mods) - {""}:
        raise Unsupported(f"PRMT mode {insn.text}")
    src = m.rd(th, wp, a) | (m.rd(th, wp, b) << 32)
    sv = m.rd(th, wp, sel)
    r = 0
    for k in range(4):
        nib = (sv >> (4 * k)) & 0xF
        byte = (src >> (8 * (nib & 7))) & 0xFF
        if nib & 8:
            byte = 0xFF if byte & 0x80 else 0  # sign-replicate the selected byte
        r |= byte << (8 * k)
    m.wr(th, wp, d, r)


@alu("FLO", "UFLO")
def _flo(m, th, wp, insn):
    v = m.src(th, wp, insn.ops[-1])
    if not insn.has("U32") and v & 0x80000000:
        v = ~v & M32
    lead = 32 - v.bit_length()
    m.wr(th, wp, insn.ops[0], lead if insn.has("SH") else (31 - lead) & M32)


@alu("POPC", "UPOPC")
def _popc(m, th, wp, insn):
    m.wr(th, wp, insn.ops[0], bin(m.src(th, wp, insn.ops[1])).count("1"))


@alu("BREV", "UBREV")
def _brev(m, th, wp, insn):
    m.wr(th, wp, insn.ops[0], int(f"{m.src(th, wp, insn.ops[1]):032b}"[::-1], 2))


@alu("SGXT", "USGXT")
def _sgxt(m, th, wp, insn):
    d, a, n = insn.ops
    av, nb = m.src(th, wp, a), m.src(th, wp, n)
    if insn.has("W"):
        nb &= 31
    if nb >= 32:
        r = av
    elif nb == 0:
        r = 0
    else:
        r = av & ((1 << nb) - 1)
        if not insn.has("U32") and r >> (nb - 1) & 1:
            r |= M32 ^ ((1 << nb) - 1)
    m.wr(th, wp, d, r & M32)


@alu("BMSK", "UBMSK")
def _bmsk(m, th, wp, insn):
    # HYPOTHESIS: mask of `width` ones starting at `pos`; width clamps at 32 (wraps with .W).
    d, pos, width = insn.ops
    p, w = m.src(th, wp, pos) & 0xFF, m.src(th, wp, width) & 0xFF
    w = w & 31 if insn.has("W") else min(w, 32)
    m.wr(th, wp, d, (((1 << w) - 1) << p) & M32)


def h2_operands(m, th, wp, ops, fmt, insn):
    """Read packed-half sources. A register (optionally .H0_H0 / .H1_H1) gives
    (lo, hi); a pair of immediates "hi, lo" gives one source. Returns [(lo, hi), ...]."""
    out, k = [], 0
    while k < len(ops):
        op = ops[k]
        if isinstance(op, Imm):
            hi, lo = (_himm(ops[k].text, fmt), _himm(ops[k + 1].text, fmt))
            out.append((lo, hi))
            k += 2
            continue
        v = m.rd(th, wp, op)
        lo, hi = v & 0xFFFF, v >> 16
        if op.sel.startswith("H0_H0"):
            hi = lo
        elif op.sel.startswith("H1_H1"):
            lo = hi
        pair = []
        for h in (lo, hi):
            x = fp.decode(h, fmt, ftz=insn.has("FTZ"))
            if op.abs:
                x = _dc.replace(x, neg=False)
            if op.neg:
                x = _dc.replace(x, neg=not x.neg)
            pair.append(x)
        out.append(tuple(pair))
        k += 1
    return out


def _himm(t: str, fmt) -> "fp.Val":
    from fractions import Fraction
    t = t.strip()
    low = t.lower().lstrip("+-")
    if low == "inf":
        return fp.Val(t.startswith("-"), inf=True)
    if low in ("qnan", "nan"):
        return fp.Val(False, nan=True)
    return fp.from_frac(Fraction(t), zero_neg=t.startswith("-"))


def _hfmt(insn):
    return fp.BF16 if insn.has("BF16_V2") or insn.has("BF16") else fp.F16


def _h2_write(m, th, wp, d, lanes, insn, fmt):
    mode = fp.mode_of(insn.mods)
    if insn.has("F32"):  # .F32: single f32 result from lane 0
        m.wr(th, wp, d, fp.encode(lanes[0], fp.F32, mode, ftz=insn.has("FTZ")))
        return
    hs = []
    for v in lanes:
        b = fp.encode(v, fmt, mode, ftz=insn.has("FTZ"))
        if insn.has("SAT"):
            r = fp.decode(b, fmt)
            if r.nan or r.neg:
                b = 0
            elif r.inf or r.q > 1:
                b = fp.encode(fp.Val(False, q=fp.Fraction(1)), fmt)
        hs.append(b)
    m.wr(th, wp, d, hs[0] | (hs[1] << 16))


@alu("HADD2")
def _hadd2(m, th, wp, insn):
    fmt = _hfmt(insn)
    a, b = h2_operands(m, th, wp, insn.ops[1:], fmt, insn)
    _h2_write(m, th, wp, insn.ops[0], [fp.add(a[i], b[i], fp.mode_of(insn.mods)) for i in (0, 1)], insn, fmt)


@alu("HMUL2")
def _hmul2(m, th, wp, insn):
    fmt = _hfmt(insn)
    a, b = h2_operands(m, th, wp, insn.ops[1:], fmt, insn)
    _h2_write(m, th, wp, insn.ops[0], [fp.mul(a[i], b[i]) for i in (0, 1)], insn, fmt)


@alu("HFMA2")
def _hfma2(m, th, wp, insn):
    # Also covers ptxas's constant idiom "HFMA2.MMA Rd, -RZ, RZ, hi, lo" (= -0*0 + c = c).
    fmt = _hfmt(insn)
    a, b, c = h2_operands(m, th, wp, insn.ops[1:], fmt, insn)
    lanes = [fp.fma(a[i], b[i], c[i], fp.mode_of(insn.mods)) for i in (0, 1)]
    _h2_write(m, th, wp, insn.ops[0], lanes, insn, fmt)


@alu("HMNMX2")
def _hmnmx2(m, th, wp, insn):
    fmt = _hfmt(insn)
    d, *srcs, p = insn.ops
    a, b = h2_operands(m, th, wp, srcs, fmt, insn)
    want_min = m.prd(th, wp, p)
    res = []
    for x, y in zip(a, b):
        if x.nan or y.nan:
            res.append(fp.Val(False, nan=True) if (x.nan and y.nan) or insn.has("NAN") else (y if x.nan else x))
            continue
        c = fp.cmp(x, y)
        if c == 0 and x.q == 0 and not x.inf:
            res.append((x if x.neg else y) if want_min else (y if x.neg else x))
        else:
            res.append(x if (c <= 0) == want_min else y)
    _h2_write(m, th, wp, d, res, insn, fmt)


@alu("F2FP")
def _f2fp(m, th, wp, insn):
    # F2FP[.BF16].PACK_AB Rd, A, B: hi = cvt(A), lo = cvt(B), round-to-nearest.
    if not insn.has("PACK_AB"):
        raise Unsupported(insn.text)
    fmt = fp.BF16 if insn.has("BF16") else fp.F16
    d, a, b = insn.ops
    mode = fp.mode_of(insn.mods)
    hi = fp.encode(fsrc(m, th, wp, a, insn), fmt, mode)
    lo = fp.encode(fsrc(m, th, wp, b, insn), fmt, mode)
    m.wr(th, wp, d, (hi << 16) | lo)


# ---- memory -------------------------------------------------------------------

_SIZES = {"U8": (1, False), "S8": (1, True), "U16": (2, False), "S16": (2, True),
          "64": (8, False), "128": (16, False)}


def mem_size(insn: Insn) -> tuple[int, bool]:
    for k, v in _SIZES.items():
        if k in insn.mods:
            return v
    return 4, False


def _space(m: Machine, th: Thread, insn: Insn):
    return {"LDG": m.gmem, "STG": m.gmem, "LDS": m.smem, "STS": m.smem,
            "LDL": th.local, "STL": th.local, "LD": m.gmem, "ST": m.gmem}[insn.op]


@alu("LDG", "LDS", "LDL", "LD")
def _ld(m, th, wp, insn):
    d, a = insn.ops[-2], insn.ops[-1]
    n, signed = mem_size(insn)
    v = _space(m, th, insn).read(m.addr(th, wp, a), n)
    if signed and v >> (8 * n - 1):
        v |= M32 ^ ((1 << 8 * n) - 1)
    m.wr(th, wp, d, v, width=max(32, 8 * n))


@alu("STG", "STS", "STL", "ST")
def _st(m, th, wp, insn):
    a, s = insn.ops
    n, _ = mem_size(insn)
    _space(m, th, insn).write(m.addr(th, wp, a), n, m.rd(th, wp, s, width=max(32, 8 * n)))


def _atom_apply(insn: Insn, old: int, v: int, v2: int | None, bits: int) -> int:
    mask = (1 << bits) - 1
    sx = lambda x: x - (1 << bits) if x >> (bits - 1) else x  # noqa: E731
    signed = insn.has("S32")
    if insn.has("ADD"):
        if insn.has("F32") or insn.has("F64"):
            return float_atomic(insn, old, v, bits)
        return (old + v) & mask
    if insn.has("MIN"):
        return (sx(old) if signed else old) if (sx(old) if signed else old) <= (sx(v) if signed else v) else v
    if insn.has("MAX"):
        return old if (sx(old) if signed else old) >= (sx(v) if signed else v) else v
    if insn.has("AND"):
        return old & v
    if insn.has("OR"):
        return old | v
    if insn.has("XOR"):
        return old ^ v
    if insn.has("EXCH"):
        return v
    if insn.has("INC"):
        return 0 if old >= v else old + 1
    if insn.has("DEC"):
        return v if old == 0 or old > v else old - 1
    if insn.has("CAS") or insn.has("CAST"):
        return v2 if old == v else old
    raise Unsupported(f"atomic op {insn.text}")


@alu("ATOMG", "ATOMS", "ATOM", "RED")
def _atom(m, th, wp, insn):
    ops = [o for o in insn.ops if not isinstance(o, Pred)]
    bits = 64 if (insn.has("64") or insn.has("F64")) else 32
    mem = m.smem if insn.op == "ATOMS" else m.gmem
    if insn.op == "RED":
        a, v = ops
        d = None
    else:
        d, a, v, *rest = ops
    cas = insn.has("CAS") or insn.has("CAST")
    addr = m.addr(th, wp, a, wide=True if insn.has("E") else None)
    old = mem.read(addr, bits // 8)
    vv = m.rd(th, wp, v, width=bits)
    v2 = m.rd(th, wp, rest[0], width=bits) if cas else None
    mem.write(addr, bits // 8, _atom_apply(insn, old, vv, v2, bits))
    if d is None:
        return
    if insn.has("CAST") and insn.has("SPIN"):
        # ATOMS.CAST.SPIN Rd, [a], cmp, new: Rd = 1 if the swap happened. Inferred from
        # ptxas's retry loop for shared-memory float atomics ("ISETP.EQ.U32 P0, Rd, 0x1").
        m.wr(th, wp, d, int(old == vv))
        return
    m.wr(th, wp, d, old, width=bits)


# ---- control flow ------------------------------------------------------------------

def _branch_ok(m, th, wp, insn) -> bool:
    # "@P0 BRA !P1, target": the second predicate further qualifies the branch.
    preds = [o for o in insn.ops if isinstance(o, Pred)]
    return all(m.prd(th, wp, p) for p in preds)


@ctrl("BRA", "JMP")
def _bra(m, th, wp, insn):
    if insn.mods and set(insn.mods) - {"U"}:
        raise Unsupported(f"branch flavor {insn.text}")
    lbl = next(o for o in insn.ops if isinstance(o, Label))
    th.pc = m.prog.target(lbl) if _branch_ok(m, th, wp, insn) else th.pc + 16


@ctrl("EXIT")
def _exit(m, th, wp, insn):
    th.done = True


@ctrl("CALL")
def _call(m, th, wp, insn):
    th.stack.append(th.pc + 16)
    th.pc = m.prog.target(next(o for o in insn.ops if isinstance(o, Label)))


@ctrl("RET")
def _ret(m, th, wp, insn):
    # ptxas passes the return address in a register (RET.REL.NODEC R4); we keep our
    # own stack and check that the register agrees with it.
    ret = th.stack.pop()
    reg = next((o for o in insn.ops if isinstance(o, Reg)), None)
    if reg is not None and m.rd(th, wp, reg) != ret:
        raise EmuError(f"RET register {m.rd(th, wp, reg):#x} != call site {ret:#x}")
    th.pc = ret


@ctrl("BRX", "JMX")
def _brx(m, th, wp, insn):
    # "BRX R6 -0xe0": target = (pc + 16) + R6 + imm. ptxas loads R6 from its jump
    # table in c[0x2] (absolute .text offsets) and sets imm = -(pc + 16), so the
    # target is the table entry. Verified on A100 (brx probe).
    import re as _re
    mm = _re.match(r"\S+\s+R(\d+)\s+(-?0x[0-9a-f]+)", insn.text)
    if not mm:
        raise Unsupported(insn.text)
    reg = int(mm[1])
    base = 0 if reg == 255 else th.r[reg]
    th.pc = (th.pc + 16 + base + int(mm[2], 16)) & 0xFFFFFFFF


@ctrl("BPT")
def _bpt(m, th, wp, insn):
    raise EmuError(f"trap at {insn.addr:#x}")


# ---- warp collectives ----------------------------------------------------------------

@warp("BAR")
def _bar(m, wp, active, insn):
    # Warps run to completion one after another, so a real CTA-wide barrier would
    # need cooperative scheduling. Single-warp CTAs (all probes) are exact.
    if len(m.warps) > 1:
        # plain BAR.SYNC is handled cooperatively by Machine.run; reductions are not
        raise Unsupported("BAR.RED / BAR.ARV with more than one warp")
    if insn.has("RED"):
        p = insn.ops[-1]
        vals = [m.prd(th, wp, p) for th in active]
        if insn.has("POPC"):
            r = sum(vals)
        elif insn.has("AND"):
            r = all(vals)
        elif insn.has("OR"):
            r = any(vals)
        else:
            raise Unsupported(insn.text)
        for th in active:
            th.r[254] = r  # read back via B2R.RESULT
    return


@alu("B2R")
def _b2r(m, th, wp, insn):
    d = insn.ops[0]
    v = th.r[254]
    if isinstance(insn.ops[-1], Pred):
        m.pwr(th, wp, insn.ops[-1], bool(v))
    if isinstance(d, Reg):
        m.wr(th, wp, d, int(v))


def _lanemask(active) -> int:
    return sum(1 << th.lane for th in active)


@warp("VOTE", "VOTEU")
def _vote(m, wp, active, insn):
    d, p = insn.ops[0], insn.ops[-1]
    vals = {th.lane: m.prd(th, wp, p) for th in active}
    ballot = sum(1 << ln for ln, v in vals.items() if v)
    res = {"ALL": all(vals.values()), "ANY": any(vals.values()),
           "EQ": len(set(vals.values())) == 1}
    for th in active:
        if isinstance(d, Reg):
            m.wr(th, wp, d, ballot)   # VOTE.ANY Rd, PT, Pp: Rd = ballot
        mode = next(k for k in res if k in insn.mods)
        pdst = insn.ops[1] if isinstance(d, Reg) else d
        if isinstance(pdst, Pred) and not (isinstance(d, Reg) and pdst.idx == 7):
            m.pwr(th, wp, pdst, res[mode])


@warp("SHFL")
def _shfl(m, wp, active, insn):
    # SHFL.mode Pout, Rd, Ra, b, c  (b = lane/delta, c = packed clamp|segmask)
    pout, d, a, b, c = insn.ops
    by_lane = {th.lane: th for th in active}
    vals = {th.lane: m.rd(th, wp, a) for th in active}
    out = {}
    for th in active:
        bv = m.rd(th, wp, b) & 0x1F
        cv = m.rd(th, wp, c)
        clamp, segmask = cv & 0x1F, (cv >> 8) & 0x1F
        lane = th.lane
        minl = lane & segmask
        maxl = minl | (clamp & ~segmask & 0x1F)
        if insn.has("IDX"):
            j = minl | (bv & ~segmask & 0x1F); ok = j <= maxl
        elif insn.has("UP"):
            j = lane - bv; ok = j >= maxl
        elif insn.has("DOWN"):
            j = lane + bv; ok = j <= maxl
        elif insn.has("BFLY"):
            j = lane ^ bv; ok = j <= maxl
        else:
            raise Unsupported(insn.text)
        if not ok:
            j = lane
        if j not in by_lane:
            raise Unsupported(f"SHFL from inactive lane {j}")
        out[lane] = (vals[j], ok)
    for th in active:
        v, ok = out[th.lane]
        m.wr(th, wp, d, v)
        m.pwr(th, wp, pout, ok)


@warp("MATCH")
def _match(m, wp, active, insn):
    ops = list(insn.ops)
    pd = ops.pop(0) if isinstance(ops[0], Pred) else None
    d, a = ops
    wide = insn.has("U64")
    vals = {th.lane: m.rd(th, wp, a, width=64 if wide else 32) for th in active}
    for th in active:
        same = sum(1 << ln for ln, v in vals.items() if v == vals[th.lane])
        if insn.has("ALL"):
            allsame = same == _lanemask(active)
            m.wr(th, wp, d, same if allsame else 0)
            if pd is not None:
                m.pwr(th, wp, pd, allsame)
        else:
            m.wr(th, wp, d, same)


@warp("REDUX")
def _redux(m, wp, active, insn):
    d, a = insn.ops
    vals = [m.rd(th, wp, a) for th in active]
    sv = [s32(v) for v in vals] if insn.has("S32") else vals
    if insn.has("SUM"):
        r = sum(vals)
    elif insn.has("MIN"):
        r = min(sv)
    elif insn.has("MAX"):
        r = max(sv)
    elif insn.has("OR"):
        r = 0
        for v in vals:
            r |= v
    elif insn.has("XOR"):
        r = 0
        for v in vals:
            r ^= v
    else:  # bare REDUX = AND
        r = M32
        for v in vals:
            r &= v
    m.wr(active[0], wp, d, r & M32)  # destination is a uniform register


# ---- floating point (exact, via openassembler.emu.fp) ---------------------------------------

from openassembler.emu import fp  # noqa: E402


def fsrc(m, th, wp, op, insn, fmt=fp.F32) -> fp.Val:
    """Float source: immediates are decimal text, others are raw bits; then |x| and -x."""
    if isinstance(op, Imm):
        t = op.text.strip()
        if t.lower().lstrip("+-") in ("inf", "+inf"):
            v = fp.Val(t.startswith("-"), inf=True)
        elif t.lower().lstrip("+-") in ("qnan", "nan"):
            v = fp.Val(False, nan=True)
        elif t.lower().startswith(("0x", "-0x")):
            v = fp.decode(int(t, 16) & ((1 << fmt.width) - 1), fmt)
        else:
            from fractions import Fraction
            v = fp.from_frac(Fraction(t), zero_neg=t.startswith("-"))
    else:
        width = 64 if fmt is fp.F64 else 32
        v = fp.decode(m.rd(th, wp, op, width=width), fmt, ftz=insn.has("FTZ"))
    if getattr(op, "abs", False):
        v = _dc.replace(v, neg=False)
    if getattr(op, "neg", False):
        v = _dc.replace(v, neg=not v.neg)
    return v


def fout(m, th, wp, d, v: fp.Val, insn, fmt=fp.F32) -> None:
    mode = fp.mode_of(insn.mods)
    bits = fp.encode(v, fmt, mode, ftz=insn.has("FTZ"))
    if insn.has("SAT") and fmt is fp.F32:
        # .SAT clamps to [0, 1]; NaN -> +0
        r = fp.decode(bits)
        if r.nan or r.neg:
            bits = 0
        elif r.inf or r.q > 1:
            bits = 0x3F800000
    m.wr(th, wp, d, bits, width=fmt.width)


@alu("FADD")
def _fadd(m, th, wp, insn):
    d, a, b = insn.ops
    fout(m, th, wp, d, fp.add(fsrc(m, th, wp, a, insn), fsrc(m, th, wp, b, insn), fp.mode_of(insn.mods)), insn)


@alu("FMUL")
def _fmul(m, th, wp, insn):
    d, a, b = insn.ops
    fout(m, th, wp, d, fp.mul(fsrc(m, th, wp, a, insn), fsrc(m, th, wp, b, insn)), insn)


@alu("FFMA")
def _ffma(m, th, wp, insn):
    d, a, b, c = insn.ops
    v = fp.fma(*(fsrc(m, th, wp, x, insn) for x in (a, b, c)), fp.mode_of(insn.mods))
    fout(m, th, wp, d, v, insn)


@alu("DADD")
def _dadd(m, th, wp, insn):
    d, a, b = insn.ops
    v = fp.add(fsrc(m, th, wp, a, insn, fp.F64), fsrc(m, th, wp, b, insn, fp.F64), fp.mode_of(insn.mods))
    fout(m, th, wp, d, v, insn, fp.F64)


@alu("DMUL")
def _dmul(m, th, wp, insn):
    d, a, b = insn.ops
    fout(m, th, wp, d, fp.mul(fsrc(m, th, wp, a, insn, fp.F64), fsrc(m, th, wp, b, insn, fp.F64)), insn, fp.F64)


@alu("DFMA")
def _dfma(m, th, wp, insn):
    d, a, b, c = insn.ops
    v = fp.fma(*(fsrc(m, th, wp, x, insn, fp.F64) for x in (a, b, c)), fp.mode_of(insn.mods))
    fout(m, th, wp, d, v, insn, fp.F64)


def _fcmp(insn: Insn, a: fp.Val, b: fp.Val) -> bool:
    c = fp.cmp(a, b)
    unordered = c is None
    table = {"EQ": c == 0, "NE": c not in (0, None), "LT": c == -1, "LE": c in (-1, 0),
             "GT": c == 1, "GE": c in (0, 1), "NUM": not unordered, "NAN": unordered,
             "EQU": unordered or c == 0, "NEU": unordered or c != 0, "LTU": unordered or c == -1,
             "LEU": unordered or c in (-1, 0), "GTU": unordered or c == 1, "GEU": unordered or c in (0, 1),
             "T": True, "F": False}
    for k in insn.mods:
        if k in table:
            return table[k]
    raise Unsupported(f"float compare in {insn.text}")


@alu("FSETP", "DSETP")
def _fsetp(m, th, wp, insn):
    fmt = fp.F64 if insn.op == "DSETP" else fp.F32
    pu, pv, a, b, acc = insn.ops[:5]
    if insn.has("MIN") or insn.has("MAX"):
        # DSETP.{MIN,MAX}.AND Pu, Pv, a, b, acc  (ptxas's min/max.f64 idiom):
        #   Pu = "a is the result" (then SEL/FSEL pick a, else b), combined with acc
        #   Pv = both NaN -> caller ORs the quiet bit into the result's high word
        # NaN rule: a NaN operand loses unless both are NaN; -0 < +0.
        av, bv = fsrc(m, th, wp, a, insn, fmt), fsrc(m, th, wp, b, insn, fmt)
        want_min = insn.has("MIN")
        if av.nan or bv.nan:
            pick_a = not av.nan
        else:
            c = fp.cmp(av, bv)
            if c == 0 and av.q == 0 and not av.inf:
                pick_a = av.neg == want_min
            else:
                pick_a = (c <= 0) == want_min
        m.pwr(th, wp, pu, bool_op(insn, pick_a, m.prd(th, wp, acc)))
        m.pwr(th, wp, pv, av.nan and bv.nan)
        return
    c = _fcmp(insn, fsrc(m, th, wp, a, insn, fmt), fsrc(m, th, wp, b, insn, fmt))
    accv = m.prd(th, wp, acc)
    m.pwr(th, wp, pu, bool_op(insn, c, accv))
    m.pwr(th, wp, pv, bool_op(insn, not c, accv))  # HYPOTHESIS, as for ISETP


@alu("FMNMX")
def _fmnmx(m, th, wp, insn):
    # PTX min/max.f32: NaN inputs are ignored unless both are NaN (-> canonical NaN);
    # HYPOTHESIS: -0 < +0. .NAN propagates NaN instead.
    d, a, b, p = insn.ops
    av, bv = fsrc(m, th, wp, a, insn), fsrc(m, th, wp, b, insn)
    want_min = m.prd(th, wp, p)
    if av.nan or bv.nan:
        if insn.has("NAN") or (av.nan and bv.nan):
            m.wr(th, wp, d, fp.CANON_NAN32)
            return
        m.wr(th, wp, d, fp.encode(bv if av.nan else av))
        return
    c = fp.cmp(av, bv)
    if c == 0 and av.q == 0:
        r = (av if av.neg else bv) if want_min else (bv if av.neg else av)
    else:
        r = av if (c <= 0) == want_min else bv
    m.wr(th, wp, d, fp.encode(r))


@alu("I2F")
def _i2f(m, th, wp, insn):
    # I2F[.F64][.S64|.U32|.U64|.S16...][.RZ|.RP|.RM] Rd, src
    d, a = insn.ops
    # Measured mnemonics: I2F.F64.S64 = s64 -> f64; I2F.S64 = s64 -> f32; I2F.F64 = s32 -> f64.
    dfmt = fp.F64 if insn.has("F64") else fp.F32
    srcbits = 64 if (insn.has("S64") or insn.has("U64")) else 32
    signed = not any(x in insn.mods for x in ("U32", "U64", "U16", "U8"))
    v = m.rd(th, wp, a, width=srcbits)
    if signed and v >> (srcbits - 1):
        v -= 1 << srcbits
    fout(m, th, wp, d, fp.from_frac(fp.Fraction(v)), insn, dfmt)


@alu("F2I")
def _f2i(m, th, wp, insn):
    # F2I[.FTZ][.U32|.S64|.U64][.F64][.TRUNC|.FLOOR|.CEIL][.NTZ] Rd, src
    d, a = insn.ops
    # F2I.S64.F64 = f64 -> s64; F2I.S64 = f32 -> s64; F2I.F64 = f64 -> s32.
    sfmt = fp.F64 if insn.has("F64") else fp.F32
    bits = 64 if (insn.has("S64") or insn.has("U64")) else 32
    signed = not (insn.has("U32") or insn.has("U64"))
    mode = "RZ" if insn.has("TRUNC") else "RM" if insn.has("FLOOR") else "RP" if insn.has("CEIL") else "RN"
    m.wr(th, wp, d, fp.to_int(fsrc(m, th, wp, a, insn, sfmt), bits, signed, mode, src_f64=sfmt is fp.F64), width=max(32, bits))


@alu("F2F")
def _f2f(m, th, wp, insn):
    # F2F.dst.src Rd, src  (e.g. F2F.F64.F32, F2F.F32.F64)
    fmts = {"F64": fp.F64, "F32": fp.F32, "F16": fp.F16}
    dst, src = (fmts[x] for x in insn.mods if x in fmts)
    d, a = insn.ops
    v = fsrc(m, th, wp, a, insn, src)
    if v.nan:  # A100: sign kept, payload shifted to the new width (not canonicalized)
        m.wr(th, wp, d, fp.encode(v, dst, keep_nan=True), width=dst.width)
        return
    fout(m, th, wp, d, v, insn, dst)


@alu("FRND")
def _frnd(m, th, wp, insn):
    d, a = insn.ops
    v = fsrc(m, th, wp, a, insn)
    if v.nan or v.inf:
        m.wr(th, wp, d, fp.encode(v))
        return
    mode = "RZ" if insn.has("TRUNC") else "RM" if insn.has("FLOOR") else "RP" if insn.has("CEIL") else "RN"
    n = fp._round_int(v.q, mode, v.neg)
    m.wr(th, wp, d, fp.encode(fp.Val(v.neg, q=fp.Fraction(n))))


@alu("MUFU")
def _mufu(m, th, wp, insn):
    # HYPOTHESIS: correctly rounded result. Real MUFU is ~1 ulp approximate.
    import math
    d, a = insn.ops
    v = fsrc(m, th, wp, a, insn)
    fn = next(x for x in insn.mods)
    if fn in ("RCP64H", "RSQ64H"):
        # Input: high word of an f64 (low word taken as 0). Output: high word of the
        # f64 reciprocal / reciprocal sqrt. HYPOTHESIS: truncated exact result; ptxas
        # follows it with Newton-Raphson DFMAs, so the final result only needs the
        # seed to be accurate, which the A100 replay checks.
        hi = m.rd(th, wp, a)
        x = fp.decode(hi << 32, fp.F64)
        if x.nan:
            r = 0x7FF80000
        elif x.q == 0:
            r = 0xFFF00000 if x.neg else 0x7FF00000
        elif x.inf:
            r = 0x80000000 if (x.neg and fn == "RCP64H") else 0
        elif fn == "RSQ64H" and x.neg:
            r = 0xFFF80000
        else:
            q = 1 / x.q if fn == "RCP64H" else _inv_sqrt(x.q)
            r = fp.encode(fp.Val(x.neg and fn == "RCP64H", q=q), fp.F64, "RZ") >> 32
        m.wr(th, wp, d, r)
        return
    if fn == "RCP":
        if v.nan:
            r = v
        elif v.inf:
            r = fp.Val(v.neg)
        elif v.q == 0:
            r = fp.Val(v.neg, inf=True)
        else:
            r = fp.Val(v.neg, q=1 / v.q)
        m.wr(th, wp, d, fp.encode(r))
        return
    x = float(fp.signed_frac(v)) if not (v.nan or v.inf) else (math.nan if v.nan else (-math.inf if v.neg else math.inf))
    try:
        # RSQ(-0) = -inf, measured on A100 (IEEE rSqrt rule)
        if fn == "RSQ" and not v.nan and not v.inf and v.q == 0:
            m.wr(th, wp, d, 0xFF800000 if v.neg else 0x7F800000)
            return
        y = {"RSQ": lambda x: 1 / math.sqrt(x) if x > 0 else (math.inf if x == 0 else math.nan),
             "SQRT": lambda x: math.sqrt(x) if x >= 0 else math.nan,
             "EX2": lambda x: 2.0 ** x, "LG2": lambda x: math.log2(x) if x > 0 else (-math.inf if x == 0 else math.nan),
             # HYPOTHESIS (from ptxas's FMUL.RZ by 1/2pi before MUFU.SIN): input in revolutions
             "SIN": lambda x: math.sin(2 * math.pi * x), "COS": lambda x: math.cos(2 * math.pi * x),
             "TANH": math.tanh}[fn](x)
    except (KeyError,):
        raise Unsupported(insn.text)
    except (OverflowError, ValueError):
        y = math.inf if fn == "EX2" else math.nan
    m.wr(th, wp, d, f32_from_float(y))


def f32_from_float(y: float) -> int:
    import math
    if math.isnan(y):
        return fp.CANON_NAN32
    if math.isinf(y):
        return 0xFF800000 if y < 0 else 0x7F800000
    return fp.encode(fp.from_frac(fp.Fraction(y), zero_neg=math.copysign(1, y) < 0))


@alu("FCHK")
def _fchk(m, th, wp, insn):
    # FCHK P, a, b: true when a/b must take the slow path. The exact hardware rule is
    # unmeasured; this is a CONSERVATIVE superset: any special operand, or a quotient /
    # dividend whose exponent leaves [-125, 126] (where the RCP + Newton sequence can
    # overflow, underflow or lose precision). Taking the slow path more often than the
    # GPU is always correct, because the slow path handles every input.
    p, a, b = insn.ops
    av, bv = fsrc(m, th, wp, a, insn), fsrc(m, th, wp, b, insn)
    def exp(q):
        return q.numerator.bit_length() - q.denominator.bit_length()
    special = av.nan or bv.nan or av.inf or bv.inf or bv.q == 0
    if special:
        m.pwr(th, wp, p, True)
        return
    if av.q == 0:
        m.pwr(th, wp, p, True)  # zero dividend: fast path's "FFMA ..., RZ" would drop -0's sign
        return
    ea, eb = exp(av.q), exp(bv.q)
    m.pwr(th, wp, p, not (-125 <= ea <= 126 and -125 <= eb <= 126 and -125 <= ea - eb <= 126))



# ---- tensor cores, ldmatrix, cp.async (openassembler.emu.tensor) ------------------------------

from openassembler.emu import tensor as _tensor  # noqa: E402

WARP["HMMA"] = WARP["IMMA"] = WARP["DMMA"] = _tensor.mma
WARP["LDSM"] = _tensor.ldsm
ALU["LDGSTS"] = _tensor.ldgsts


# ---- byte dot product, abs-diff, float atomics -----------------------------------------

@alu("IDP")
def _idp(m, th, wp, insn):
    # IDP.4A.{U8,S8}.{U8,S8} Rd, Ra, Rb, Rc: d = c + sum(a.byte[i] * b.byte[i])
    d, a, b, c = insn.ops
    sa, sb = insn.mods[1] == "S8", insn.mods[2] == "S8"
    av, bv = m.src(th, wp, a), m.src(th, wp, b)
    acc = m.src(th, wp, c)
    for k in range(4):
        x, y = (av >> (8 * k)) & 0xFF, (bv >> (8 * k)) & 0xFF
        acc += (x - 256 if sa and x >= 128 else x) * (y - 256 if sb and y >= 128 else y)
    m.wr(th, wp, d, acc & M32)


@alu("VABSDIFF")
def _vabsdiff(m, th, wp, insn):
    # VABSDIFF.{U32,S32} Rd, Ra, Rb, Rc: d = |a - b| + c   (PTX sad)
    d, a, b, c = insn.ops
    av, bv = m.src(th, wp, a), m.src(th, wp, b)
    if not insn.has("U32"):
        av, bv = s32(av), s32(bv)
    m.wr(th, wp, d, (abs(av - bv) + m.src(th, wp, c)) & M32)


def float_atomic(insn, old: int, v: int, bits: int) -> int:
    """ATOM*.ADD.F32.FTZ.RN / ADD.F64.RN: IEEE add, round-to-nearest (f32 flushes subnormals)."""
    f = fp.F64 if bits == 64 else fp.F32
    ftz = insn.has("FTZ")
    r = fp.add(fp.decode(old, f, ftz=ftz), fp.decode(v, f, ftz=ftz), "RN")
    return fp.encode(r, f, "RN", ftz=ftz)



def _inv_sqrt(q):
    """floor(1/sqrt(q)) with >= 120 significant bits, for any magnitude (exact
    integer square root). The old fixed-point 2^-200 version returned 0 for
    q > 2^400, which the A100 replay of sqrt.rn.f64 exposed."""
    from fractions import Fraction
    import math
    n, d = q.numerator, q.denominator
    e = n.bit_length() - d.bit_length()          # q ~ 2^e, result ~ 2^(-e/2)
    k = max(120, 120 + (e + 1) // 2)
    return Fraction(math.isqrt(d * 4 ** k // n), 2 ** k)



@alu("QSPC")
def _qspc(m, th, wp, insn):
    # QSPC.E.{S,G,L} Pd, Rd, [Ra]: is the generic address in that state space?
    # The emulator gives kernels only global pointers, so .S/.L -> false, .G -> true.
    # HYPOTHESIS for real shared/local windows (A100: isspacep.shared(global ptr) = 0).
    p = insn.ops[0]
    m.pwr(th, wp, p, insn.has("G"))
    if isinstance(insn.ops[1], Reg) and insn.ops[1].idx != 255:
        m.wr(th, wp, insn.ops[1], 0)
