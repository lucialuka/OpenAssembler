"""PTX instruction semantics (from the PTX ISA spec)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from openassembler.emu import fp
from openassembler.ptx.interp import M, Interp, Lane, NotModeled, Undefined, sext, types_of, _TBITS
from openassembler.ptx.parse import Instr, PairDst, Var, Vec

LANE_OPS: dict[str, Callable] = {}   # per-lane: fn(it, ln, ins) -> None
WARP_OPS: dict[str, Callable] = {}   # collective: fn(it, lanes, ins) -> None


def lane(*names):
    def r(fn):
        for n in names:
            LANE_OPS[n] = fn
        return fn
    return r


def warp(*names):
    def r(fn):
        for n in names:
            WARP_OPS[n] = fn
        return fn
    return r


def execute(it: Interp, ins: Instr, lanes: list[Lane], pc: int) -> dict[int, list[Lane]]:
    """Run one instruction for `lanes`; return {next_pc: lanes}."""
    on = [ln for ln in lanes if it.guard(ln, ins)]
    off = [ln for ln in lanes if not it.guard(ln, ins)]
    nxt: dict[int, list[Lane]] = {pc + 1: list(off)} if off else {}
    if ins.op == "bra":
        if on:
            nxt.setdefault(it.k.labels[ins.args[0].name], []).extend(on)
        return nxt
    if ins.op == "brx":
        for ln in on:
            idx = it.get(ln, ins.args[0], 32)
            tgt = it.k.branchtargets[ins.args[1].name][idx]
            nxt.setdefault(it.k.labels[tgt], []).append(ln)
        return nxt
    if ins.op in ("ret", "exit"):
        for ln in on:
            ln.done = True
        return nxt
    if ins.op == "trap":
        if on:
            raise Undefined("trap")
        return nxt
    if on:
        if ins.op in WARP_OPS:
            WARP_OPS[ins.op](it, on, ins)
        elif ins.op in LANE_OPS:
            for ln in on:
                try:
                    LANE_OPS[ins.op](it, ln, ins)
                except Undefined:
                    ln.undefined = True
        elif ins.op in NOP_OPS:
            pass
        else:
            raise NotModeled(f"{ins.op}.{'.'.join(ins.mods)}")
    nxt.setdefault(pc + 1, []).extend(on)
    return nxt


NOP_OPS = {"membar", "fence", "prefetch", "nanosleep", "pmevent", "brkpt"}


def ty(ins: Instr, k: int = 0) -> str:
    ts = types_of(ins)
    if not ts:
        raise NotModeled(f"no type on {ins.op}.{'.'.join(ins.mods)}")
    return ts[k]


def bits(t: str) -> int:
    return _TBITS[t]


def is_float(t: str) -> bool:
    return t.startswith(("f", "bf"))


def signed(t: str) -> bool:
    return t.startswith("s")


def rd(it, ln, op, t: str) -> int:
    """Read operand as type t; signed types come back as Python negative ints."""
    v = it.get(ln, op, bits(t))
    return sext(v, bits(t)) if signed(t) else v


# ---- data movement -------------------------------------------------------------------

@lane("mov")
def _mov(it, ln, ins):
    d, a = ins.args
    t = ty(ins)
    if isinstance(a, Vec):        # mov.b64 %rd, {%r1, %r2}: pack, element 0 lowest
        n = bits(t) // len(a.elems)
        v = 0
        for k, e in enumerate(a.elems):
            v |= it.get(ln, e, n) << (n * k)
        it.put(ln, d, v, bits(t))
    elif isinstance(d, Vec):      # mov.b64 {%r1, %r2}, %rd: unpack
        n = bits(t) // len(d.elems)
        v = it.get(ln, a, bits(t))
        for k, e in enumerate(d.elems):
            if e is not None:
                it.put(ln, e, v >> (n * k), n)
    elif isinstance(a, Var) and a.name in it.sym and it.sym[a.name][0] in ("shared", "local"):
        it.put(ln, d, it.sym[a.name][1], bits(t))
    else:
        it.put(ln, d, it.get(ln, a, bits(t)), bits(t))


@lane("cvta")
def _cvta(it, ln, ins):
    # Global generic == global address in our model; shared/local window addresses are
    # opaque, so only round-tripping through cvta is meaningful.
    d, a = ins.args
    it.put(ln, d, it.get(ln, a, 64), 64)


@lane("ld")
def _ld(it, ln, ins):
    d, a = ins.args
    t = ty(ins)
    if "param" in ins.mods:
        it.put(ln, d, it.L.params[a.base], bits(t))
        return
    mem = it.space_mem(ins, ln)
    base = it.addr(ln, a)
    elems = d.elems if isinstance(d, Vec) else (d,)
    n = bits(t) // 8
    for k, e in enumerate(elems):
        v = mem.read(base + k * n, n)
        if e is not None:
            reg_bits = 32 if n < 4 else bits(t)  # sub-word loads widen into 32-bit regs
            it.put(ln, e, sext(v, bits(t)) if signed(t) else v, reg_bits)


@lane("st")
def _st(it, ln, ins):
    a, s = ins.args
    t = ty(ins)
    mem = it.space_mem(ins, ln)
    base = it.addr(ln, a)
    elems = s.elems if isinstance(s, Vec) else (s,)
    n = bits(t) // 8
    for k, e in enumerate(elems):
        mem.write(base + k * n, n, it.get(ln, e, max(bits(t), 16) if n < 4 else bits(t)))


# ---- integer arithmetic ----------------------------------------------------------------

def _int_result(it, ln, d, v: int, t: str) -> None:
    it.put(ln, d, v, bits(t))


@lane("add", "sub")
def _add(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _fadd(it, ln, ins)
    d, a, b = ins.args
    av, bv = rd(it, ln, a, t), rd(it, ln, b, t)
    if ins.has("cc"):
        # add.cc / sub.cc: carry/borrow out into the (single, per-lane) CC.CF flag
        n = bits(t)
        ua, ub = av & M(n), bv & M(n)
        if ins.op == "add":
            r = ua + ub
            ln.regs["%CC.CF"] = r >> n
        else:
            r = ua - ub
            ln.regs["%CC.CF"] = int(ua < ub)
        _int_result(it, ln, d, r, t)
        return
    if ins.has("sat"):
        r = av + bv if ins.op == "add" else av - bv
        _int_result(it, ln, d, max(-(1 << 31), min((1 << 31) - 1, r)), t)
        return
    _int_result(it, ln, d, av + bv if ins.op == "add" else av - bv, t)


@lane("addc", "subc")
def _addc(it, ln, ins):
    t = ty(ins)
    d, a, b = ins.args
    n = bits(t)
    ua, ub = it.get(ln, a, n), it.get(ln, b, n)
    cf = ln.regs.get("%CC.CF", 0)
    r = ua + ub + cf if ins.op == "addc" else ua - ub - cf
    if ins.has("cc"):
        ln.regs["%CC.CF"] = (r >> n) & 1 if ins.op == "addc" else int(r < 0)
    _int_result(it, ln, d, r, t)


@lane("mul")
def _mul(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _fmul(it, ln, ins)
    d, a, b = ins.args
    n = bits(t)
    p = rd(it, ln, a, t) * rd(it, ln, b, t)
    if ins.has("wide"):
        it.put(ln, d, p, 2 * n)
    elif ins.has("hi"):
        it.put(ln, d, p >> n, n)
    else:
        it.put(ln, d, p, n)


@lane("mad", "fma")
def _mad(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _ffma(it, ln, ins)
    d, a, b, c = ins.args
    n = bits(t)
    p = rd(it, ln, a, t) * rd(it, ln, b, t)
    if ins.has("wide"):
        it.put(ln, d, p + rd(it, ln, c, ("s" if signed(t) else "u") + str(2 * n)), 2 * n)
    elif ins.has("hi"):
        it.put(ln, d, (p >> n) + rd(it, ln, c, t), n)
    else:
        it.put(ln, d, p + rd(it, ln, c, t), n)


@lane("div", "rem")
def _div(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _fdiv(it, ln, ins)
    d, a, b = ins.args
    av, bv = rd(it, ln, a, t), rd(it, ln, b, t)
    if bv == 0:
        raise Undefined("integer division by zero")
    q = abs(av) // abs(bv) * (1 if (av < 0) == (bv < 0) else -1)  # truncate toward zero
    it.put(ln, d, q if ins.op == "div" else av - q * bv, bits(t))


@lane("min", "max")
def _minmax(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _fminmax(it, ln, ins)
    d, a, b = ins.args
    av, bv = rd(it, ln, a, t), rd(it, ln, b, t)
    it.put(ln, d, min(av, bv) if ins.op == "min" else max(av, bv), bits(t))


@lane("abs", "neg")
def _absneg(it, ln, ins):
    t = ty(ins)
    if is_float(t):
        return _fabsneg(it, ln, ins)
    d, a = ins.args
    v = rd(it, ln, a, t)
    it.put(ln, d, abs(v) if ins.op == "abs" else -v, bits(t))


@lane("and", "or", "xor")
def _logic(it, ln, ins):
    t = ty(ins)
    d, a, b = ins.args
    n = bits(t)
    if t == "pred":
        av, bv = it.pred(ln, a.name), it.pred(ln, b.name)
        r = {"and": av and bv, "or": av or bv, "xor": av != bv}[ins.op]
        it.put(ln, d, int(r), 1)
        return
    av, bv = it.get(ln, a, n), it.get(ln, b, n)
    it.put(ln, d, {"and": av & bv, "or": av | bv, "xor": av ^ bv}[ins.op], n)


@lane("not")
def _not(it, ln, ins):
    t = ty(ins)
    d, a = ins.args
    if t == "pred":
        it.put(ln, d, int(not it.pred(ln, a.name)), 1)
    else:
        it.put(ln, d, ~it.get(ln, a, bits(t)), bits(t))


@lane("cnot")
def _cnot(it, ln, ins):
    d, a = ins.args
    it.put(ln, d, int(it.get(ln, a, bits(ty(ins))) == 0), bits(ty(ins)))


@lane("shl", "shr")
def _shift(it, ln, ins):
    t = ty(ins)
    d, a, b = ins.args
    n = bits(t)
    s = it.get(ln, b, 32)          # shift amount is always u32; clamped per spec
    v = rd(it, ln, a, t) if ins.op == "shr" else it.get(ln, a, n)
    if ins.op == "shl":
        it.put(ln, d, v << s if s < n else 0, n)
    else:
        it.put(ln, d, v >> min(s, n), n)  # signed: Python >> is arithmetic


@lane("shf")
def _shf(it, ln, ins):
    d, a, b, c = ins.args
    lo, hi, s = it.get(ln, a, 32), it.get(ln, b, 32), it.get(ln, c, 32)
    s = s & 31 if ins.has("wrap") else min(s, 32)
    x = (hi << 32) | lo
    it.put(ln, d, (x << s) >> 32 if ins.has("l") else x >> s, 32)


@lane("popc")
def _popc(it, ln, ins):
    d, a = ins.args
    it.put(ln, d, bin(it.get(ln, a, bits(ty(ins)))).count("1"), 32)


@lane("clz")
def _clz(it, ln, ins):
    d, a = ins.args
    n = bits(ty(ins))
    it.put(ln, d, n - it.get(ln, a, n).bit_length(), 32)


@lane("brev")
def _brev(it, ln, ins):
    d, a = ins.args
    n = bits(ty(ins))
    it.put(ln, d, int(f"{it.get(ln, a, n):0{n}b}"[::-1], 2), n)


@lane("bfe")
def _bfe(it, ln, ins):
    t = ty(ins)
    d, a, b, c = ins.args
    n = bits(t)
    v = it.get(ln, a, n)
    pos, ln_ = it.get(ln, b, 32) & 0xFF, it.get(ln, c, 32) & 0xFF
    sbit = 0
    if signed(t):
        sbit = (v >> (n - 1)) & 1 if pos + ln_ >= n else ((v >> (pos + ln_ - 1)) & 1 if ln_ else 0)
    r = 0
    for i in range(n):
        if i < ln_ and pos + i <= n - 1:
            r |= ((v >> (pos + i)) & 1) << i
        elif i >= ln_ or pos + i > n - 1:
            r |= sbit << i
    it.put(ln, d, r, n)


@lane("bfi")
def _bfi(it, ln, ins):
    t = ty(ins)
    d, f, b, c, e = ins.args
    n = bits(t)
    fv, bv = it.get(ln, f, n), it.get(ln, b, n)
    pos, ln_ = it.get(ln, c, 32) & 0xFF, it.get(ln, e, 32) & 0xFF
    r = bv
    for i in range(ln_):
        if pos + i < n:
            r = (r & ~(1 << (pos + i))) | (((fv >> i) & 1) << (pos + i))
    it.put(ln, d, r, n)


@lane("prmt")
def _prmt(it, ln, ins):
    if len(ins.mods) > 1:
        raise NotModeled("prmt mode")
    d, a, b, c = ins.args
    src = it.get(ln, a, 32) | (it.get(ln, b, 32) << 32)
    sel = it.get(ln, c, 32)
    r = 0
    for k in range(4):
        nib = (sel >> (4 * k)) & 0xF
        byte = (src >> (8 * (nib & 7))) & 0xFF
        if nib & 8:
            byte = 0xFF if byte & 0x80 else 0
        r |= byte << (8 * k)
    it.put(ln, d, r, 32)


@lane("lop3")
def _lop3(it, ln, ins):
    d, a, b, c, lut = ins.args
    x, y, z = (it.get(ln, o, 32) for o in (a, b, c))
    L = lut.value
    r = 0
    for i in range(8):
        if L >> i & 1:
            r |= (x if i & 4 else ~x) & (y if i & 2 else ~y) & (z if i & 1 else ~z)
    it.put(ln, d, r, 32)


@lane("sad")
def _sad(it, ln, ins):
    t = ty(ins)
    d, a, b, c = ins.args
    av, bv = rd(it, ln, a, t), rd(it, ln, b, t)
    it.put(ln, d, abs(av - bv) + rd(it, ln, c, t), bits(t))


@lane("dp4a")
def _dp4a(it, ln, ins):
    ta, tb = types_of(ins)[:2]
    d, a, b, c = ins.args
    av, bv = it.get(ln, a, 32), it.get(ln, b, 32)
    s = rd(it, ln, c, "s32" if "s32" in (ta, tb) else "u32")
    for k in range(4):
        x, y = (av >> (8 * k)) & 0xFF, (bv >> (8 * k)) & 0xFF
        s += (sext(x, 8) if ta == "s32" else x) * (sext(y, 8) if tb == "s32" else y)
    it.put(ln, d, s, 32)


# ---- compare / select ------------------------------------------------------------------

_INT_CMP = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
            "lo": lambda a, b: a < b, "ls": lambda a, b: a <= b, "hi": lambda a, b: a > b,
            "hs": lambda a, b: a >= b}


def compare(it, ln, ins, a, b) -> bool:
    t = types_of(ins)[-1]
    cmp = next(m for m in ins.mods if m in _INT_CMP or m in _F_CMP)
    if is_float(t):
        fmt = fp.F64 if t == "f64" else fp.F32
        va, vb = fsrc(it, ln, a, fmt, ins), fsrc(it, ln, b, fmt, ins)
        return _F_CMP[cmp](fp.cmp(va, vb))
    return _INT_CMP[cmp](rd(it, ln, a, t), rd(it, ln, b, t))


_F_CMP = {"eq": lambda c: c == 0, "ne": lambda c: c in (-1, 1), "lt": lambda c: c == -1,
          "le": lambda c: c in (-1, 0), "gt": lambda c: c == 1, "ge": lambda c: c in (0, 1),
          "equ": lambda c: c in (None, 0), "neu": lambda c: c != 0, "ltu": lambda c: c in (None, -1),
          "leu": lambda c: c in (None, -1, 0), "gtu": lambda c: c in (None, 1),
          "geu": lambda c: c in (None, 0, 1), "num": lambda c: c is not None, "nan": lambda c: c is None}


@lane("setp")
def _setp(it, ln, ins):
    d, a, b, *rest = ins.args
    r = compare(it, ln, ins, a, b)
    comb = next((m for m in ins.mods if m in ("and", "or", "xor")), None)
    if comb:
        c = it.pred(ln, rest[0].name)
        rq = {"and": (r and c, (not r) and c), "or": (r or c, (not r) or c),
              "xor": (r != c, (not r) != c)}[comb]
    else:
        rq = (r, not r)
    if isinstance(d, PairDst):
        it.put(ln, d.a, int(rq[0]), 1)
        it.put(ln, d.b, int(rq[1]), 1)
    else:
        it.put(ln, d, int(rq[0]), 1)


@lane("set")
def _set(it, ln, ins):
    d, a, b = ins.args
    r = compare(it, ln, ins, a, b)
    dt = types_of(ins)[0]
    it.put(ln, d, (M(32) if r else 0) if dt != "f32" else (0x3F800000 if r else 0), 32)


@lane("selp")
def _selp(it, ln, ins):
    d, a, b, p = ins.args
    n = bits(ty(ins))
    it.put(ln, d, it.get(ln, a, n) if it.pred(ln, p.name) else it.get(ln, b, n), n)


@lane("slct")
def _slct(it, ln, ins):
    d, a, b, c = ins.args
    dt, ct = types_of(ins)[0], types_of(ins)[1]
    n = bits(dt)
    if ct == "f32":
        v = fp.decode(it.get(ln, c, 32), ftz=ins.has("ftz"))
        take_a = (not v.neg) or v.q == 0 and not v.nan  # c >= 0 (incl. -0)
    else:
        take_a = rd(it, ln, c, "s32") >= 0
    it.put(ln, d, it.get(ln, a, n) if take_a else it.get(ln, b, n), n)


@lane("cvt")
def _cvt(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.cvt(it, ln, ins)


# ---- float stubs (implemented in semantics_fp) -------------------------------------------

def fsrc(it, ln, op, fmt, ins):
    from openassembler.ptx import semantics_fp as SF
    return SF.fsrc(it, ln, op, fmt, ins)


def _fadd(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.fadd(it, ln, ins)


def _fmul(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.fmul(it, ln, ins)


def _ffma(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.ffma(it, ln, ins)


def _fdiv(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.fdiv(it, ln, ins)


def _fminmax(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.fminmax(it, ln, ins)


def _fabsneg(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.fabsneg(it, ln, ins)


@lane("rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh")
def _unary_f(it, ln, ins):
    from openassembler.ptx import semantics_fp as SF
    SF.unary_approx(it, ln, ins)


@lane("copysign")
def _copysign(it, ln, ins):
    d, a, b = ins.args
    n = bits(ty(ins))
    sign = 1 << (n - 1)
    it.put(ln, d, (it.get(ln, b, n) & ~sign) | (it.get(ln, a, n) & sign), n)


@lane("testp")
def _testp(it, ln, ins):
    d, a = ins.args
    t = types_of(ins)[0]
    v = fp.decode(it.get(ln, a, bits(t)), fp.F64 if t == "f64" else fp.F32)
    sub = (not v.nan and not v.inf and v.q != 0 and
           v.q < Fraction(2) ** (1 - (fp.F64 if t == "f64" else fp.F32).bias))
    r = {"finite": not (v.nan or v.inf), "infinite": v.inf, "number": not v.nan, "notanumber": v.nan,
         "normal": not (v.nan or v.inf or v.q == 0 or sub), "subnormal": sub}[ins.mods[0]]
    it.put(ln, d, int(r), 1)


# ---- atomics ----------------------------------------------------------------------------

@lane("atom", "red")
def _atom(it, ln, ins):
    t = ty(ins)
    n = bits(t)
    if is_float(t):
        raise NotModeled("float atomics")
    args = list(ins.args)
    d = args.pop(0) if ins.op == "atom" else None
    a, b, *rest = args
    mem = it.space_mem(ins, ln)
    addr = it.addr(ln, a)
    old = mem.read(addr, n // 8)
    bv = it.get(ln, b, n)
    so, sb = (sext(old, n), sext(bv, n)) if signed(t) else (old, bv)
    op = next(m for m in ins.mods if m in ("add", "min", "max", "and", "or", "xor", "exch", "inc", "dec", "cas"))
    new = {"add": old + bv, "min": min(so, sb), "max": max(so, sb), "and": old & bv, "or": old | bv,
           "xor": old ^ bv, "exch": bv, "inc": 0 if old >= bv else old + 1,
           "dec": bv if old == 0 or old > bv else old - 1,
           "cas": (it.get(ln, rest[0], n) if old == bv else old) if rest else None}[op]
    mem.write(addr, n // 8, new & M(n))
    if d is not None:
        it.put(ln, d, old, n)


# ---- warp collectives (lanes = active lanes, in lane order) ------------------------------------

@warp("shfl")
def _shfl(it, lanes, ins):
    d, a, b, c, mask = ins.args
    pdst = None
    if isinstance(d, PairDst):
        d, pdst = d.a, d.b
    mode = next(m for m in ins.mods if m in ("idx", "up", "down", "bfly"))
    by = {ln.lane: ln for ln in lanes}
    vals = {ln.lane: it.get(ln, a, 32) for ln in lanes}
    out = {}
    for ln in lanes:
        bv, cv = it.get(ln, b, 32) & 0x1F, it.get(ln, c, 32)
        clamp, segmask = cv & 0x1F, (cv >> 8) & 0x1F
        maxl = (ln.lane & segmask) | (clamp & ~segmask & 0x1F)
        minl = ln.lane & segmask
        j, ok = {"idx": (minl | (bv & ~segmask & 0x1F), None), "up": (ln.lane - bv, None),
                 "down": (ln.lane + bv, None), "bfly": (ln.lane ^ bv, None)}[mode]
        ok = j >= maxl if mode == "up" else j <= maxl
        if not ok:
            j = ln.lane
        if j not in by:
            raise Undefined("shfl from inactive lane")
        out[ln.lane] = (vals[j], ok)
    for ln in lanes:
        v, ok = out[ln.lane]
        it.put(ln, d, v, 32)
        if pdst is not None:
            it.put(ln, pdst, int(ok), 1)


@warp("vote")
def _vote(it, lanes, ins):
    d, p = ins.args[0], ins.args[1]
    vals = [it.pred(ln, p.name) for ln in lanes]
    if ins.has("ballot"):
        r = sum(1 << ln.lane for ln, v in zip(lanes, vals) if v)
        for ln in lanes:
            it.put(ln, d, r, 32)
        return
    r = {"all": all(vals), "any": any(vals), "uni": len(set(vals)) == 1}[
        next(m for m in ins.mods if m in ("all", "any", "uni"))]
    for ln in lanes:
        it.put(ln, d, int(r), 1)


@warp("activemask")
def _activemask(it, lanes, ins):
    r = sum(1 << ln.lane for ln in lanes)
    for ln in lanes:
        it.put(ln, ins.args[0], r, 32)


@warp("match")
def _match(it, lanes, ins):
    d, a, _mask_op = ins.args
    pdst = None
    if isinstance(d, PairDst):
        d, pdst = d.a, d.b
    n = bits(ty(ins))
    vals = {ln.lane: it.get(ln, a, n) for ln in lanes}
    active = sum(1 << ln.lane for ln in lanes)
    for ln in lanes:
        same = sum(1 << j for j, v in vals.items() if v == vals[ln.lane])
        if ins.has("all"):
            ok = same == active
            it.put(ln, d, same if ok else 0, 32)
            if pdst is not None:
                it.put(ln, pdst, int(ok), 1)
        else:
            it.put(ln, d, same, 32)


@warp("redux")
def _redux(it, lanes, ins):
    d, a, _m = ins.args
    t = ty(ins)
    vals = [rd(it, ln, a, t) for ln in lanes]
    op = next(m for m in ins.mods if m in ("add", "min", "max", "and", "or", "xor"))
    r = vals[0]
    for v in vals[1:]:
        r = {"add": r + v, "min": min(r, v), "max": max(r, v), "and": r & v, "or": r | v, "xor": r ^ v}[op]
    for ln in lanes:
        it.put(ln, d, r, 32)


# ---- barriers ---------------------------------------------------------------------------
# CTA barriers (bar/barrier .sync/.arrive/.red) are scheduled by Interp.run/_bar_arrive,
# which needs every warp's state; this module only decodes their operands. Only the
# warp-scope bar.warp.sync stays an ordinary collective.

@dataclass
class BarrierForm:
    bid: int                    # barrier id 0..15
    count: int | None           # expected threads, None = whole CTA
    waits: bool                 # .sync/.red block; .arrive does not
    red: str | None             # "popc" / "and" / "or"
    preds: dict[int, bool]      # bar.red: id(lane) -> predicate value
    aligned: bool


def cta_barrier(ins: Instr) -> bool:
    return ins.op in ("bar", "barrier") and not ins.has("warp")


def _uniform(it, lanes, op, what: str, ins: Instr) -> int:
    vals = {it.get(ln, op, 32) for ln in lanes}
    if len(vals) != 1:
        raise NotModeled(f"{ins.op}.{'.'.join(ins.mods)}: {what} differs between threads of a warp "
                         f"({sorted(vals)}); undefined behavior per the PTX ISA")
    return vals.pop()


def barrier_form(it, ins: Instr, lanes: list[Lane]) -> BarrierForm:
    """Decode the CTA-barrier forms nvcc and hand-written PTX use:
        bar{.cta}.sync a{, b}      barrier{.cta}.sync{.aligned} a{, b}
        bar{.cta}.arrive a, b      barrier{.cta}.arrive{.aligned} a, b
        bar{.cta}.red.popc.u32 d, a{, b}, {!}p
        bar{.cta}.red.{and,or}.pred d, a{, b}, {!}p     (barrier.red likewise)
    a = barrier id 0..15, b = thread count (immediate or register, uniform per
    warp). __syncthreads() is `bar.sync 0`; __syncthreads_count/_and/_or are the
    bar.red forms. Anything else (bare `bar`, .cluster, bar.arrive without a
    count, other reduction types) is rejected rather than guessed at."""
    name = f"{ins.op}.{'.'.join(ins.mods)}"
    known = {"cta", "sync", "arrive", "red", "aligned", "popc", "and", "or", "u32", "pred"}
    extra = [m for m in ins.mods if m not in known]
    if extra:
        raise NotModeled(f"{name}: modifier(s) {extra} not modeled (only CTA-scope barriers)")
    kinds = [m for m in ("sync", "arrive", "red") if m in ins.mods]
    if len(kinds) != 1:
        raise NotModeled(f"{name}: expected exactly one of .sync/.arrive/.red")
    if ins.has("aligned") and ins.op == "bar":
        raise NotModeled(f"{name}: .aligned is only valid on barrier.*")
    kind, args = kinds[0], list(ins.args)
    red, preds = None, {}
    if kind == "red":
        ops = [m for m in ("popc", "and", "or") if m in ins.mods]
        want = {"popc": "u32", "and": "pred", "or": "pred"}
        if len(ops) != 1 or not ins.has(want[ops[0]]) or len(args) not in (3, 4):
            raise NotModeled(f"{name}: expected bar.red.popc.u32 / bar.red.and.pred / bar.red.or.pred "
                             "d, a{, b}, {!}p")
        red, p = ops[0], args[-1]
        if not isinstance(p, Var):
            raise NotModeled(f"{name}: predicate operand {p}")
        preds = {id(ln): it.pred(ln, p.name) for ln in lanes}
        args = args[1:-1]
    elif any(m in ins.mods for m in ("popc", "and", "or", "u32", "pred")):
        raise NotModeled(f"{name}: reduction modifiers without .red")
    if not 1 <= len(args) <= 2 or (kind == "arrive" and len(args) != 2):
        raise NotModeled(f"{name}: expected barrier id{' and thread count' if kind == 'arrive' else ''}"
                         f"{'' if kind == 'arrive' else ' and optional thread count'}")
    bid = _uniform(it, lanes, args[0], "barrier id", ins)
    if bid > 15:
        raise NotModeled(f"{name}: barrier id {bid} out of range 0..15")
    count = _uniform(it, lanes, args[1], "thread count", ins) if len(args) == 2 else None
    return BarrierForm(bid, count, kind != "arrive", red, preds,
                       aligned=ins.op == "bar" or ins.has("aligned"))


@warp("bar", "barrier")
def _bar_warp(it, lanes, ins):
    """bar.warp.sync membermask (__syncwarp). The scheduler runs the lowest-pc
    lane group first, so any live lane of the mask that is not executing this
    instruction with us has diverged past it and would have to arrive later at a
    (possibly different) bar.warp.sync -- that rendezvous is not modeled, so it is
    rejected instead of treated as a no-op. Exited lanes in the mask are ignored."""
    if not ins.has("warp", "sync"):
        raise NotModeled(f"{ins.op}.{'.'.join(ins.mods)}")
    mask = _uniform(it, lanes, ins.args[0], "membermask", ins)
    here = {ln.lane for ln in lanes}
    w = it.warps[lanes[0].warp]
    away = [ln.lane for ln in w if not ln.done and mask >> ln.lane & 1 and ln.lane not in here]
    if away:
        raise NotModeled(f"bar.warp.sync: lanes {away[:8]} of the membermask are on a divergent path; "
                         "divergent __syncwarp is not modeled")
