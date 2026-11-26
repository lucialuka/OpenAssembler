"""M3 v1: PTX -> SASS with a virtual-register IR, copy elimination, and linear-scan
register allocation (sm_80, straight-line kernels).

Pipeline
  select()    PTX -> IR over virtual registers. PTX moves are renamings (no code);
              kernel params stay c[0x0][...] operands; (mul.wide + add.s64) becomes
              one IMAD.WIDE; v4/v2 vectors become "groups" that must get consecutive,
              aligned physical registers.
  dce()       drop instructions whose results are never used (and have no side effect).
  allocate()  linear scan over live intervals; groups (64/128-bit values) get
              aligned consecutive registers; a group member defined by a plain copy
              is coalesced into its slot so the copy disappears.
  lower()     IR -> SASS text with physical registers (feeds compile.schedule + encoder).

Every unsupported construct raises NotSupported; nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openassembler.opt import divconst
from openassembler.ptx.parse import ImmF, Addr, ImmI, Instr, Kernel, Var, Vec

PARAM_BASE = 0x160
SREG_S2R = {"%tid.x": "SR_TID.X", "%tid.y": "SR_TID.Y", "%tid.z": "SR_TID.Z",
            "%ctaid.x": "SR_CTAID.X", "%ctaid.y": "SR_CTAID.Y", "%ctaid.z": "SR_CTAID.Z",
            "%laneid": "SR_LANEID", "%lanemask_eq": "SR_EQMASK", "%lanemask_lt": "SR_LTMASK",
            "%lanemask_ge": "SR_GEMASK"}
SREG_C0 = {"%ntid.x": 0x0, "%ntid.y": 0x4, "%ntid.z": 0x8, "%nctaid.x": 0xC, "%nctaid.y": 0x10, "%nctaid.z": 0x14}


class NotSupported(Exception):
    pass


# ---- IR --------------------------------------------------------------------------------

@dataclass(frozen=True)
class V:
    """A 32-bit virtual register."""
    n: int

    def __repr__(self):
        return f"v{self.n}"


@dataclass(frozen=True)
class C:
    """c[0x0][off] operand (kernel params, launch constants)."""
    off: int


@dataclass(frozen=True)
class I:
    """32-bit immediate."""
    v: int


@dataclass
class Op:
    """One SASS instruction over virtual registers.

    fmt:   SASS text with {d0} {d1}.. for defs and {s0} {s1}.. for uses
    defs:  list of tuples of V; a tuple of length 2/4 must be consecutive, aligned regs
    uses:  same, for sources (V, C, I or tuples of V for wide sources)
    side:  has an effect besides its defs (store, exit) -> never removed by DCE
    var_lat: variable latency (needs a write scoreboard)
    """
    fmt: str
    defs: list = field(default_factory=list)
    uses: list = field(default_factory=list)
    side: bool = False
    var_lat: bool = False
    copy: bool = False      # plain 32-bit MOV d, s (candidate for coalescing)
    guard: tuple | None = None   # (P, negated): predicate register P (a PV) guarding this op
    label: str | None = None     # this op is a label pseudo-op (no code)
    target: str | None = None    # branch target label (BRA)


@dataclass(frozen=True)
class PV:
    """A virtual predicate register (allocated to P0..P6)."""
    n: int

    def __repr__(self):
        return f"p{self.n}"


class Selector:
    def __init__(self, k: Kernel):
        self.k = k
        self.ops: list[Op] = []
        self.nv = 0
        self.env: dict[str, object] = {}     # PTX name -> V | tuple[V, V] | C | I | ("cpair", off)
        self.sym_shared = {}
        soff = 0
        for d_ in k.decls:
            if d_.space == "shared":
                soff = (soff + d_.align - 1) // d_.align * d_.align
                self.sym_shared[d_.name] = soff
                soff += d_.size
        self.smem_bytes = soff
        from openassembler.backend.storage import local_frame
        try:self.local_frame=local_frame(k)
        except ValueError as error:raise NotSupported(str(error)) from None
        from openassembler.ptx.parse import parameter_layout
        self.param_off = {name: PARAM_BASE+offset for name, (offset, size) in parameter_layout(k.params).items()}

    def new(self) -> V:
        self.nv += 1
        return V(self.nv)

    def emit(self, fmt, defs=(), uses=(), **kw) -> Op:
        g = getattr(self, "cur_guard", None)
        if g is not None and "guard" not in kw and not kw.get("label"):
            kw["guard"] = g
        op = Op(fmt, list(defs), list(uses), **kw)
        self.ops.append(op)
        return op

    def bind(self, name: str, v) -> None:
        """PTX register `name` := v. For home registers, emit a real copy into the home."""
        if name in getattr(self, "homes", {}):
            if isinstance(v, tuple):
                v = self.pair_value(v)
                h = self.homes[name]
                if h is None:
                    h = self.homes[name] = (self.new(), self.new())
                if not isinstance(h, tuple):
                    raise NotSupported(f"mixed-width assignments to {name}")
                for dst, src in zip(h, v):
                    if dst != src:
                        self.emit("MOV {d0}, {s0}", [dst], [src], copy=True)
                self.env[name] = h
                return
            if not isinstance(v, (V, C, I)):
                raise NotSupported(f"non-32-bit value for multiply-assigned {name}")
            h = self.homes[name]
            if h is None:
                h = self.homes[name] = self.new()
            last = self.ops[-1] if self.ops else None
            if (isinstance(v, V) and last is not None and last.defs == [v] and not last.label
                    and last.guard == getattr(self, "cur_guard", None)
                    and v not in _flat(self.env.values())
                    and v not in _flat(self.homes.values())):
                # The value was just computed by the previous op and nothing has read it:
                # write the home directly instead of computing into a temp + copying.
                last.defs = [h]
            elif v != h:
                self.emit("MOV {d0}, {s0}", [h], [v], copy=isinstance(v, V))
            self.env[name] = h
            return
        self.env[name] = v

    def bind_pred(self, name: str, value: PV) -> None:
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

    def snapshot_parts(self, values):
        """A pack/unpack copies values now, including components of mutable homes."""
        mutable = set(_flat(self.homes.values()))
        if any(value in mutable for value in values):
            snapshots = tuple(self.new() for _ in values)
            for destination, value in zip(snapshots, values):
                self.emit("MOV {d0}, {s0}", [destination], [value], copy=True)
            return snapshots
        return values

    def newp(self) -> PV:
        self.npv = getattr(self, "npv", 0) + 1
        return PV(self.npv)

    def wide(self, t: tuple) -> tuple:
        """A tuple used as a wide (consecutive-register) operand. If no single op
        defines exactly this tuple, copy members into a fresh group; coalesce()
        later deletes every copy that has no real register conflict."""
        if any(t == d for op in self.ops for d in op.defs if isinstance(d, tuple)):
            return t
        fresh = tuple(self.new() for _ in t)
        for f, v in zip(fresh, t):
            self.emit("MOV {d0}, {s0}", [f], [v], copy=True)
        return fresh

    # value lookup -------------------------------------------------------------

    def val32(self, o):
        if isinstance(o, ImmI):
            return I(o.value & 0xFFFFFFFF)
        if isinstance(o, ImmF) and o.width == 32:
            return I(o.bits)          # f32 immediate: raw bits (MOV / FFMA imm take them as is)
        if isinstance(o, Var):
            if o.name not in self.env:
                raise NotSupported(f"read of undefined/special {o.name}")
            v = self.env[o.name]
            if isinstance(v, (V, C, I)):
                return v
        raise NotSupported(f"32-bit operand {o}")

    def reg32(self, o) -> V:
        """Operand that must be a register (materialize c[]/imm with a MOV)."""
        v = self.val32(o)
        if isinstance(v, V):
            return v
        cache = self.__dict__.setdefault("_const_cse", {})
        if getattr(self, "cur_guard", None) is not None:
            # a predicated instruction's operand may not be hoisted into a cache the
            # unpredicated code after it will trust
            d = self.new()
            self.emit("MOV {d0}, {s0}", [d], [v], copy=False)
            return d
        if v not in cache:  # CSE: one register per distinct immediate / c[] value
            d = self.new()
            self.emit("MOV {d0}, {s0}", [d], [v], copy=False)
            cache[v] = d
        return cache[v]

    def val64(self, name: str):
        v = self.env.get(name)
        if v is None:
            raise NotSupported(f"undefined 64-bit {name}")
        return v

    def pair_value(self, v):
        if isinstance(v, tuple) and v[0] == "cpair":
            lo, hi = self.new(), self.new()
            self.emit("MOV {d0}, {s0}", [lo], [C(v[1])])
            self.emit("MOV {d0}, {s0}", [hi], [C(v[1] + 4)])
            return lo, hi
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, V) for x in v):
            return v
        if isinstance(v, tuple) and v[0] == "widemul":
            _, a, b, unsigned = v
            lo, hi = self.new(), self.new()
            self.emit(f"IMAD.WIDE{'.U32' if unsigned else ''} {{d0}}, {{s0}}, {{s1}}, RZ",
                      [(lo, hi)], [a, b])
            return lo, hi
        raise NotSupported(f"64-bit value {v}")

    def pair(self, name: str) -> tuple[V, V]:
        original = self.val64(name)
        v = self.pair_value(original)
        # A lazily materialized parameter/wide multiply may be first used in
        # only one branch. Keep the symbolic value across CFG paths so a later
        # use cannot trust registers whose definition it skipped.
        if not (getattr(self, "has_cfg", False) and isinstance(original, tuple)
                and original[0] in ("cpair", "widemul")):
            self.env[name] = v
        return v

    def operand64(self, operand):
        if isinstance(operand, (ImmI, ImmF)):
            if isinstance(operand, ImmF) and operand.width != 64:
                raise NotSupported(f"64-bit operand requires a 64-bit float literal: {operand}")
            bits = operand.bits if isinstance(operand, ImmF) else operand.value
            return (self.reg32(ImmI(bits & 0xffffffff)),
                    self.reg32(ImmI((bits >> 32) & 0xffffffff)))
        if not isinstance(operand, Var):
            raise NotSupported(f"64-bit operand {operand}")
        return self.pair(operand.name)

    # wide multiply kept symbolic until used, so add.s64 can fuse into IMAD.WIDE ----

    def materialize_widemul(self, name, v):
        _, a, b, unsigned = v
        lo, hi = self.new(), self.new()
        self.emit(f"IMAD.WIDE{'.U32' if unsigned else ''} {{d0}}, {{s0}}, {{s1}}, RZ",
                  [(lo, hi)], [a, b])
        self.env[name] = (lo, hi)
        return lo, hi

    # instruction selection -----------------------------------------------------------

    def run(self) -> list[Op]:
        r1 = self.new()
        self.emit("MOV {d0}, c[0x0][0x28]", [r1], [], side=True)       # stack pointer (R1 by ABI)
        self.r1 = r1
        if self.local_frame.bytes:
            self.emit(f"IADD3 {{d0}}, {{s0}}, -{self.local_frame.bytes:#x}, RZ",[r1],[r1],side=True)
        self.emit("ULDC.64 UR4, c[0x0][0x118]", [], [], side=True)     # global memory descriptor
        # A guarded write is a partial definition (the old value survives when the guard
        # is false), so it needs the same fixed home as a write inside a loop. Missing this
        # made "@p add" write a fresh register (found by tools/lat_sweep.py on the A100).
        self.has_cfg = any(i.op == "bra" or i.guard is not None for i in self.k.body)
        # In kernels with control flow, a PTX register assigned more than once gets one
        # fixed virtual home (so values flow correctly around back edges / joins).
        counts = {}
        for ins in self.k.body:
            if ins.args and ins.op not in ("st", "bra", "red"):
                destination = ins.args[0]
                destinations = destination.elems if isinstance(destination, Vec) else (destination,)
                for value in destinations:
                    if isinstance(value, Var):
                        counts[value.name] = counts.get(value.name, 0) + 1
        self.homes = {n: None for n, c in counts.items() if c > 1} if self.has_cfg else {}
        lab_at = {}
        for name, idx in self.k.labels.items():
            lab_at.setdefault(idx, []).append(name)
        for idx, ins in enumerate(self.k.body):
            for name in lab_at.get(idx, []):
                self.emit("", label=name)
                # A label is a join point: threads can arrive having skipped the code above
                # it. A constant materialized (MOV R, imm) in a branch that only some threads
                # took must not be reused after the join -- "constant CSE" cached it in a
                # register those threads never wrote (found by `if i<3: o[i]=7` followed by
                # `if i>5: o[i]=9`: the second store used the first branch's MOV of 4).
                self.__dict__.pop("_const_cse", None)
            self.cur_guard = None
            if ins.guard is not None:
                p, neg = ins.guard
                if p not in self.env or not isinstance(self.env[p], PV):
                    raise NotSupported(f"guard on undefined predicate {p}")
                self.cur_guard = (self.env[p], neg)
            try:
                self.select(ins)
            except NotSupported as exc:
                if str(exc).startswith('PTX line '):
                    raise
                raise NotSupported(f'PTX line {ins.line} ({ins.op}.{".".join(ins.mods)}): {exc}') from None
        for name in lab_at.get(len(self.k.body), []):
            self.emit("", label=name)
        return self.ops

    def select(self, ins: Instr) -> None:
        from .contracts import ContractError, validate_scalar
        try:
            validate_scalar(ins)
        except ContractError as exc:
            raise NotSupported(f'PTX line {ins.line}: {exc}') from None
        op, m, a = ins.op, ins.mods, ins.args
        t = next((x for x in m if x[:1] in ("s", "u", "b", "f") and x[1:].isdigit()), None)
        if op == "ld" and "param" in m:
            from openassembler.ptx.parse import parameter_load
            try: relative, width = parameter_load(self.k, a[1], t)
            except ValueError as error: raise NotSupported(str(error)) from None
            off = PARAM_BASE+relative
            if width < 4:
                value = self.new()
                self.emit("MOV {d0}, {s0}", [value], [C(off & ~3)])
                shift, bits = (off & 3)*8, width*8
                if t.startswith('s'):
                    left = 32-bits-shift
                    if left:
                        moved = self.new()
                        self.emit(f"SHF.L.U32 {{d0}}, {{s0}}, {left:#x}, RZ", [moved], [value])
                        value = moved
                    result = self.new()
                    self.emit(f"SHF.R.S32.HI {{d0}}, RZ, {32-bits:#x}, {{s0}}", [result], [value])
                else:
                    if shift:
                        moved = self.new()
                        self.emit(f"SHF.R.U32.HI {{d0}}, RZ, {shift:#x}, {{s0}}", [moved], [value])
                        value = moved
                    result = self.new()
                    self.emit(f"LOP3.LUT {{d0}}, {{s0}}, {(1 << bits)-1:#x}, RZ, 0xc0, !PT", [result], [value])
                self.bind(a[0].name, result)
            else:
                self.bind(a[0].name, ("cpair", off) if t and t.endswith("64") else C(off))
            return
        if op in ("ld","st") and "local" in m:
            return self.local(ins,t)
        if op=="mov" and isinstance(a[1],Var) and a[1].name in self.local_frame.offsets:
            if t not in("u32","b32"):
                raise NotSupported("local symbols require explicit 32-bit local addresses")
            destination=self.new()
            self.emit(f"IADD3 {{d0}}, {{s0}}, {self.local_frame.offsets[a[1].name]:#x}, RZ",[destination],[self.r1])
            self.bind(a[0].name,destination)
            return
        if op == "cvta":                                   # global generic == global
            self.bind(a[0].name, self.val64(a[1].name))
            return
        if "pred" in m and op in ("and", "or", "xor", "not", "mov"):
            p = self.newp()
            srcs = [self.pred(a[1])]
            if op in ("and", "or", "xor"):
                srcs.append(self.pred(a[2]))
                lut = {"and": 0xc0, "or": 0xfc, "xor": 0x3c}[op]
                fmt = f"PLOP3.LUT {{p0}}, PT, {{q0}}, {{q1}}, PT, {lut:#x}, 0x0"
            else:
                fmt = f"PLOP3.LUT {{p0}}, PT, {{q0}}, PT, PT, {0x0f if op == 'not' else 0xf0:#x}, 0x0"
            self.emit(fmt, [p], srcs)
            self.bind_pred(a[0].name, p)
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name in SREG_S2R:
            d = self.new()
            self.emit(f"S2R {{d0}}, {SREG_S2R[a[1].name]}", [d], [], var_lat=True)
            self.bind(a[0].name, d)
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name in SREG_C0:
            self.bind(a[0].name, C(SREG_C0[a[1].name]))
            return
        if op == "mov" and isinstance(a[1], Vec):          # pack {lo, hi}: pure renaming
            self.bind(a[0].name, self.snapshot_parts(tuple(self.reg32(e) for e in a[1].elems)))
            return
        if op == "mov" and isinstance(a[0], Vec):          # unpack: pure renaming
            lo, hi = self.snapshot_parts(self.pair(a[1].name))
            for e, v in zip(a[0].elems, (lo, hi)):
                if e is not None:
                    self.bind(e.name, v)
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name in self.sym_shared:
            self.bind(a[0].name, I(self.sym_shared[a[1].name]))   # shared window offset
            return
        if op == "mov":
            if isinstance(a[1], Var) and a[1].name.startswith("%") and a[1].name not in self.env:
                raise NotSupported(f"special register {a[1].name}")
            if t and t.endswith("64"):
                v = self.operand64(a[1]) if isinstance(a[1], (ImmI, ImmF)) else self.val64(a[1].name)
                if isinstance(v, tuple) and v in self.homes.values() and a[0].name not in self.homes:
                    snap = (self.new(), self.new())
                    for dst, src in zip(snap, v):
                        self.emit("MOV {d0}, {s0}", [dst], [src], copy=True)
                    v = snap
                self.bind(a[0].name, v)
            else:
                v = self.val32(a[1])
                if (isinstance(v, V) and a[0].name not in getattr(self, "homes", {})
                        and v in getattr(self, "homes", {}).values()):
                    # `mov %b, %a` where %a is a multiply-assigned (loop-carried) home and %b is
                    # an ordinary single-assignment register: %b is a SNAPSHOT of %a's value
                    # right now. Aliasing it to the home (the pure-renaming fast path) would
                    # make %b see later writes to %a -- e.g. `m = n; for j in range(0, m): m = 100`
                    # ran 100 iterations instead of n. Take a real copy instead.
                    snap = self.new()
                    self.emit("MOV {d0}, {s0}", [snap], [v], copy=True)
                    v = snap
                self.bind(a[0].name, v)
            return
        if op == "mul" and m == ("lo", t) and t in ("u64", "s64"):
            (xl, xh), (yl, yh) = self.operand64(a[1]), self.operand64(a[2])
            low, carry, cross, high = self.new(), self.new(), self.new(), self.new()
            self.emit("IMAD {d0}, {s0}, {s1}, RZ", [low], [xl, yl])
            self.emit("IMAD.HI.U32 {d0}, {s0}, {s1}, RZ", [carry], [xl, yl])
            self.emit("IMAD {d0}, {s0}, {s1}, {s2}", [cross], [xh, yl, carry])
            self.emit("IMAD {d0}, {s0}, {s1}, {s2}", [high], [xl, yh, cross])
            self.bind(a[0].name, (low, high))
            return
        if op == "mul" and "wide" in m and t in ("u32", "s32"):
            left, right = self.snapshot_parts((self.reg32(a[1]), self.val32(a[2])))
            self.bind(a[0].name, ("widemul", left, right, t == "u32"))
            return
        if op in ("add", "sub") and t in ("s64", "u64"):
            if op == "add" and all(isinstance(x, Var) for x in a[1:]):
                x, y = a[1].name, a[2].name
                vx, vy = self.val64(x), self.val64(y)
                for wm, base in ((vy, x), (vx, y)):
                    if isinstance(wm, tuple) and wm[0] == "widemul":
                        _, ma, mb, uns = wm
                        bv = self.val64(base)
                        src = C(bv[1]) if bv[0] == "cpair" else self.wide(self.pair(base))
                        if isinstance(src, C) and not isinstance(mb, V):
                            tmp = self.new()
                            self.emit("MOV {d0}, {s0}", [tmp], [mb])
                            mb = tmp
                        lo, hi = self.new(), self.new()
                        self.emit(f"IMAD.WIDE{'.U32' if uns else ''} {{d0}}, {{s0}}, {{s1}}, {{s2}}",
                                  [(lo, hi)], [ma, mb, src])
                        self.bind(a[0].name, (lo, hi))
                        return
            (xl, xh), (yl, yh) = self.operand64(a[1]), self.operand64(a[2])
            lo, hi, carry = self.new(), self.new(), self.newp()
            if op == "add":
                self.emit("IADD3 {d0}, {p0}, {s0}, {s1}, RZ", [lo, carry], [xl, yl])
                self.emit("IADD3.X {d0}, {s0}, {s1}, RZ, {q0}, !PT", [hi], [xh, yh, (carry, False)])
            else:
                self.emit("IADD3 {d0}, {p0}, {s0}, -{s1}, RZ", [lo, carry], [xl, yl])
                inverted = self.new()
                self.emit("LOP3.LUT {d0}, {s0}, RZ, RZ, 0x0f, !PT", [inverted], [yh])
                self.emit("IADD3.X {d0}, {s0}, {s1}, RZ, {q0}, !PT", [hi], [xh, inverted, (carry, False)])
            self.bind(a[0].name, (lo, hi))
            return
        if op == "cvt" and len(m) == 2 and all(x in ("s32", "u32", "s64", "u64") for x in m):
            dst, src = m
            if dst.endswith("64") and src.endswith("32"):
                lo, hi = self.reg32(a[1]), self.new()
                if src == "s32":
                    self.emit("SHF.R.S32.HI {d0}, RZ, 0x1f, {s0}", [hi], [lo])
                else:
                    self.emit("MOV {d0}, 0x0", [hi])
                self.bind(a[0].name, self.snapshot_parts((lo, hi)))
            elif dst.endswith("32") and src.endswith("64"):
                self.bind(a[0].name, self.snapshot_parts((self.operand64(a[1])[0],))[0])
            elif dst.endswith("32"):
                self.bind(a[0].name, self.snapshot_parts((self.val32(a[1]),))[0])
            else:
                self.bind(a[0].name, self.snapshot_parts(self.operand64(a[1])))
            return
        if t in ("b64", "u64", "s64") and op in ("and", "or", "xor", "shl", "shr"):
            xl, xh = self.operand64(a[1])
            lo, hi = self.new(), self.new()
            if op in ("and", "or", "xor"):
                yl, yh = self.operand64(a[2])
                lut = {"and": 0xc0, "or": 0xfc, "xor": 0x3c}[op]
                for dst, x, y in ((lo, xl, yl), (hi, xh, yh)):
                    self.emit(f"LOP3.LUT {{d0}}, {{s0}}, {{s1}}, RZ, {lut:#x}, !PT", [dst], [x, y])
            else:
                return self.shift64(ins, t)
            self.bind(a[0].name, (lo, hi))
            return
        if op in ("atom", "red"):
            return self.atomic_global(ins)
        if op == "ld" and "global" in m:
            return self.load_global(ins, t)
        if op == "st" and "global" in m:
            return self.store_global(ins, t)
        if op == "ret":
            self.emit("EXIT", side=True)
            return
        if op == "bra":
            self.emit("BRA `({target})", side=True, target=a[0].name)
            return
        if op == "setp":
            return self.setp(ins, t)
        if op == "selp" and t in ("b64", "u64", "s64", "f64"):
            predicate = self.pred(a[3])
            left, right = self.operand64(a[1]), self.operand64(a[2])
            low, high = self.new(), self.new()
            self.emit("SEL {d0}, {s0}, {s1}, {p0}", [low], [left[0], right[0], predicate])
            self.emit("SEL {d0}, {s0}, {s1}, {p0}", [high], [left[1], right[1], predicate])
            self.bind(a[0].name, (low, high))
            return
        if op == "selp" and t in ("b32", "u32", "s32", "f32"):
            p = self.pred(a[3])
            d = self.new()
            self.emit("SEL {d0}, {s0}, {s1}, {p0}", [d], [self.reg32(a[1]), self.val32(a[2]), p])
            self.bind(a[0].name, d)
            return
        if op in ("ld", "st") and "shared" in m:
            return self.shared(ins, t)
        if op == "bar" and ("sync" in m or not m) and len(a) == 1 and isinstance(a[0], ImmI) and a[0].value == 0:
            self.emit("BAR.SYNC.DEFER_BLOCKING 0x0", side=True)
            return
        if op == "shfl" and m[1] != "bfly":
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
        if op == "shfl":
            if (m != ("sync", "bfly", "b32") or len(a) != 5
                    or not isinstance(a[0], Var) or ins.guard is not None
                    or not isinstance(a[3], ImmI) or a[3].value != 31
                    or not isinstance(a[4], ImmI) or a[4].value != 0xffffffff):
                raise NotSupported("only unpredicated full-warp shfl.sync.bfly.b32 with clamp=31")
            # PTX .sync promises convergence of every named lane. The ptxas -O0
            # shuffle helper uses this exact WARPSYNC register-mask sequence;
            # the generated template and field/register sweeps are recorded in
            # exp/weaknesses/reductions_warpsync_evidence.json.
            d = self.new()
            mask = self.reg32(ImmI(0xffffffff))
            self.emit("WARPSYNC {s0}", [], [mask], side=True)
            self.emit("SHFL.BFLY PT, {d0}, {s0}, {s1}, 0x1f", [d],
                      [self.reg32(a[1]), self.reg32(a[2])], var_lat=True)
            self.bind(a[0].name, d)
            return
        if op == "mov" and isinstance(a[1], Var) and a[1].name in self.sym_shared:
            self.bind(a[0].name, I(self.sym_shared[a[1].name]))
            return
        from openassembler.backend.fragments import library
        key = ".".join((op, *m))
        if key in library() and not (op in ("div", "rem") and t in ("s32", "u32") and isinstance(a[2], ImmI) and 2 <= a[2].value < 2**31):
            return self.fragment(ins, key)
        if t == "f32" and op in ("add", "sub", "mul", "fma", "mad", "min", "max", "neg", "abs"):
            return self.f32(ins)
        if t in ("u32", "s32", "b32"):
            return self.alu32(ins, t)
        raise NotSupported(f"{op}.{'.'.join(m)}")

    def fragment(self, ins, key):
        """Expand a validated arithmetic sequence into private virtual registers."""
        from openassembler.backend import fragments as F
        frag = F.library()[key]
        regs, preds = {}, {}
        def reg(n):
            if n not in regs:
                regs[n] = self.new()
            return regs[n]
        def pred(n):
            if n not in preds:
                preds[n] = self.newp()
            return preds[n]
        seqid = self.__dict__.get("_fragment_id", 0) + 1
        self._fragment_id = seqid
        prefix = f"__oa_math_{seqid}_"
        outer = getattr(self, "cur_guard", None)
        self.cur_guard = None
        if outer:
            self.emit("BRA `({target})", side=True, target=prefix + "skip", guard=(outer[0], not outer[1]))
        arity = len(ins.args) - 1
        for n, index in frag["inputs"].items():
            if index >= arity:
                continue
            src = self.val32(ins.args[index + 1])
            self.emit("MOV {d0}, {s0}", [reg(int(n))], [src])
        for line in frag["lines"]:
            if "label" in line:
                self.emit("", label=prefix + line["label"])
                continue
            fmt, defs, uses, guard, target = F.lower_line(line["text"], reg, pred, lambda n: self.wide((reg(n), reg(n + 1))))
            if target and uses:
                # BRA's optional predicate is ANDed with its instruction guard.
                if guard:
                    both = self.newp()
                    self.emit("PLOP3.LUT {p0}, PT, {q0}, {q1}, PT, 0xc0, 0x0", [both], [guard, uses[0]])
                    guard = (both, False)
                else:
                    guard = uses[0]
                fmt, uses = "BRA `({target})", []
            self.emit(fmt, defs, uses, guard=guard, target=prefix + target if target else None,
                      side=bool(target), var_lat=line["var_lat"])
        self.bind(ins.args[0].name, reg(frag["output"]))
        if outer:
            self.emit("", label=prefix + "skip")
        self.cur_guard = outer
        self.__dict__.pop("_const_cse", None)

    # predicates ------------------------------------------------------------------

    def pred(self, o):
        """PTX predicate operand -> (PV, negated)."""
        name = o.name if isinstance(o, Var) else str(o)
        neg = name.startswith("!")
        name = name.lstrip("!")
        v = self.env.get(name)
        if not isinstance(v, PV):
            raise NotSupported(f"undefined predicate {name}")
        return (v, neg)

    def setp(self, ins, t):
        a, m = ins.args, ins.mods
        cmp = next((x for x in m if x in ("eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan")), None)
        if cmp is None or len(a) != 3 or not isinstance(a[0], Var):
            raise NotSupported("setp form")
        cmp = {"lo": "lt", "ls": "le", "hi": "gt", "hs": "ge"}.get(cmp, cmp).upper()
        if t in ("s64", "u64"):
            return self.setp64(ins, t, cmp)
        p = self.newp()
        x, y = self.reg32(a[1]), self.val32(a[2])
        if t in ("u32", "s32", "b32"):
            ty = ".U32" if t in ("u32", "b32") else ""
            self.emit(f"ISETP.{cmp}{ty}.AND {{p0}}, PT, {{s0}}, {{s1}}, PT", [p], [x, y])
        elif t == "f32":
            if 'ftz' in m:
                x, y = self.flush_compare_input(x), self.flush_compare_input(self.reg32(a[2]))
            self.emit(f"FSETP.{cmp}.AND {{p0}}, PT, {{s0}}, {{s1}}, PT", [p], [x, y])
        else:
            raise NotSupported(f"setp.{t}")
        self.bind_pred(a[0].name, p)

    def setp64(self, ins, dtype, comparison):
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

    def flush_compare_input(self, value):
        """Flush subnormals for predicates using already-validated integer ops.

        Both zero signs compare identically. Preserve every normal, infinity
        and NaN bit pattern; do not approximate the floating comparison itself.
        """
        magnitude, result, subnormal = self.new(), self.new(), self.newp()
        self.emit('LOP3.LUT {d0}, {s0}, {s1}, RZ, 0xc0, !PT',
                  [magnitude], [value, I(0x7fffffff)])
        self.emit('ISETP.LT.U32.AND {p0}, PT, {s0}, {s1}, PT',
                  [subnormal], [magnitude, I(0x00800000)])
        self.emit('SEL {d0}, RZ, {s0}, {p0}', [result], [value, (subnormal, False)])
        return result

    # f32 ----------------------------------------------------------------------------

    def f32(self, ins):
        op, a, m = ins.op, ins.args, ins.mods
        if any(x in m for x in ("rz", "rm", "rp", "sat", "ftz", "NaN")):
            raise NotSupported("f32 rounding/sat/ftz modifiers")
        x = self.reg32(a[1])
        d = self.new()
        e = lambda fmt, uses: self.emit(fmt, [d], uses)  # noqa: E731
        if op == "add":
            e("FADD {d0}, {s0}, {s1}", [x, self.val32(a[2])])
        elif op == "sub":
            e("FADD {d0}, {s0}, -{s1}", [x, self.reg32(a[2])])
        elif op == "mul":
            e("FMUL {d0}, {s0}, {s1}", [x, self.val32(a[2])])
        elif op in ("fma", "mad"):
            e("FFMA {d0}, {s0}, {s1}, {s2}", [x, self.reg32(a[2]), self.val32(a[3])])
        elif op in ("min", "max"):
            e(f"FMNMX {{d0}}, {{s0}}, {{s1}}, {'PT' if op == 'min' else '!PT'}", [x, self.val32(a[2])])
        elif op == "neg":
            e("FADD {d0}, -{s0}, -RZ", [x])
        elif op == "abs":
            e("FADD {d0}, |{s0}|, -RZ", [x])
        self.bind(a[0].name, d)

    # shared memory -------------------------------------------------------------------

    def local(self,ins,t):
        address=ins.args[1] if ins.op=="ld" else ins.args[0]
        if not self.local_frame.bytes:
            raise NotSupported("local access requires a declared native frame")
        if t not in("b32","u32","s32","f32") or address.off%4:
            raise NotSupported("local accesses require aligned scalar 32-bit elements")
        if address.base in self.local_frame.offsets:
            if address.off<0 or address.off+4>self.local_frame.sizes[address.base]:
                raise NotSupported("literal local symbol access is outside its declaration")
            base=self.r1;offset=self.local_frame.offsets[address.base]+address.off
        else:
            base=self.reg32(Var(address.base));offset=address.off
        if offset%4:
            raise NotSupported("local symbol layout is not aligned for a 32-bit access")
        if abs(offset)>65535:
            raise NotSupported("local literal address offset exceeds the admitted range")
        suffix=("+" if offset>0 else "-")+hex(abs(offset)) if offset else ""
        if ins.op=="ld":
            result=self.new()
            self.emit(f"LDL {{d0}}, [{{s0}}{suffix}]",[result],[base],var_lat=True)
            self.bind(ins.args[0].name,result)
        else:
            value=self.reg32(ins.args[1])
            self.emit(f"STL [{{s0}}{suffix}], {{s1}}",[],[base,value],side=True)

    def shared(self, ins, t):
        op, a, m = ins.op, ins.args, ins.mods
        ad = a[1] if op == "ld" else a[0]
        if ad.base in self.sym_shared:
            base, off = None, self.sym_shared[ad.base] + ad.off
        else:
            base, off = self.reg32(Var(ad.base)), ad.off
        offset=("+" if off>0 else "-")+hex(abs(off)) if off else ""
        addr = f"[{{s0}}{offset}]" if base is not None else f"[RZ+{off:#x}]"
        uses0 = [base] if base is not None else []
        si = 1 if base is not None else 0
        nvec = 4 if "v4" in m else 2 if "v2" in m else 1
        wide = {1: "", 2: ".64", 4: ".128"}[nvec]
        if t in ("b8","u8","s8","b16","u16","s16","b64","u64","s64","f64"):
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
        if op == "ld":
            if nvec == 1:
                d = self.new()
                self.emit(f"LDS {{d0}}, {addr}", [d], uses0, var_lat=True)
                self.bind(a[0].name, d)
            else:
                g = tuple(self.new() for _ in range(nvec))
                self.emit(f"LDS{wide} {{d0}}, {addr}", [g], uses0, var_lat=True)
                for e_, v in zip(a[0].elems, g):
                    if e_ is not None:
                        self.bind(e_.name, v)
        else:
            src = self.reg32(a[1]) if nvec == 1 else self.wide(tuple(self.reg32(x) for x in a[1].elems))
            self.emit(f"STS{wide} {addr}, {{s{si}}}", [], uses0 + [src], side=True)

    def _addr(self, ad: Addr):
        base = self.wide(self.pair(ad.base))
        return base, ((("+" if ad.off > 0 else "-") + hex(abs(ad.off))) if ad.off else "")

    def load_global(self, ins, t):
        m, a = ins.mods, ins.args
        # Cache hints (.nc .cg .cs .lu .ca) never change the loaded value: dropping them
        # is always correct (only a performance hint is lost). .cv/.volatile change
        # memory semantics, so refuse.
        if any(x in m for x in ("cv", "volatile", "acquire", "relaxed", "mmio", "sys", "gpu", "cta", "cluster")):
            raise NotSupported("memory-ordering load")
        base, off = self._addr(a[1])
        if "v4" in m:
            g = tuple(self.new() for _ in range(4))
            self.emit(f"LDG.E.128 {{d0}}, [{{s0}}.64{off}]", [g], [base], var_lat=True)
            for e, v in zip(a[0].elems, g):
                if e is not None:
                    self.bind(e.name, v)  # bind(): multiply-assigned regs get a copy into their home
            return
        if "v2" in m:
            g = (self.new(), self.new())
            self.emit(f"LDG.E.64 {{d0}}, [{{s0}}.64{off}]", [g], [base], var_lat=True)
            for element, value in zip(a[0].elems, g):
                if element is not None:
                    self.bind(element.name, value)
            return
        sfx = {"u8": ".U8", "s8": ".S8", "u16": ".U16", "s16": ".S16", "b32": "", "u32": "", "s32": "",
               "f32": "", "b64": ".64", "u64": ".64", "s64": ".64", "f64": ".64"}.get(t)
        if sfx is None:
            raise NotSupported(f"ld.global.{t}")
        if sfx == ".64":
            g = (self.new(), self.new())
            self.emit(f"LDG.E.64 {{d0}}, [{{s0}}.64{off}]", [g], [base], var_lat=True)
            self.bind(a[0].name, g)
        else:
            d = self.new()
            self.emit(f"LDG.E{sfx} {{d0}}, [{{s0}}.64{off}]", [d], [base], var_lat=True)
            self.bind(a[0].name, d)

    def atomic_global(self, ins):
        """Only admitted GPU-scope relaxed 32-bit integer RMW/reduction forms."""
        returning = ins.op == "atom"
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
            self.emit("ERRBAR", [], [], side=True)
        operation = ins.mods[-2]
        sources = ins.args[2:] if returning else ins.args[1:]
        values = [self.reg32(value) for value in sources]
        if returning:
            result = self.new()
            if shared:
                self.emit(f"ATOMS.{operation.upper()} {{d0}}, [{{s0}}{offset}], {{s1}}",
                          [result], [base,*values], var_lat=True, side=True)
            elif operation == "cas":
                self.emit(f"ATOMG.E.CAS.STRONG.GPU PT, {{d0}}, [{{s0}}{offset}], {{s1}}, {{s2}}",
                          [result], [base, *values], var_lat=True, side=True)
            else:
                self.emit(f"ATOMG.E.{operation.upper()}.STRONG.GPU PT, {{d0}}, [{{s0}}.64{offset}], {{s1}}",
                          [result], [base, *values], var_lat=True, side=True)
            self.bind(ins.args[0].name, result)
            if semantics in ("acquire","acq_rel"):
                self.emit("CCTL.IVALL", [], [], side=True)
        else:
            self.emit(f"RED.E.ADD.STRONG.GPU [{{s0}}.64{offset}], {{s1}}", [], [base, *values], side=True)

    def store_global(self, ins, t):
        m, a = ins.mods, ins.args
        # Ordering/scope qualifiers need fences (ptxas emits MEMBAR before st.release);
        # dropping them is wrong but invisible to the single-CTA emulator, so refuse.
        # Cache hints (.wb .cg .cs .wt) only affect performance and are dropped.
        if any(x in m for x in ("release", "relaxed", "volatile", "mmio", "sys", "gpu", "cta", "cluster")):
            raise NotSupported("memory-ordered store")
        base, off = self._addr(a[0])
        src = a[1]
        if "v2" in m or "v4" in m:
            vals = self.wide(tuple(self.reg32(e) for e in src.elems))
            w = ".64" if "v2" in m else ".128"
            self.emit(f"STG.E{w} [{{s0}}.64{off}], {{s1}}", [], [base, vals], side=True)
            return
        if t in ("b64", "u64", "s64", "f64"):
            value = self.wide(self.operand64(src))
            self.emit(f"STG.E.64 [{{s0}}.64{off}], {{s1}}", [], [base, value], side=True)
            return
        suffix = {"b8": ".U8", "u8": ".U8", "s8": ".U8", "b16": ".U16", "u16": ".U16", "s16": ".U16",
                  "b32": "", "u32": "", "s32": "", "f32": ""}.get(t)
        if suffix is None:
            raise NotSupported(f"st.global.{t}")
        self.emit(f"STG.E{suffix} [{{s0}}.64{off}], {{s1}}", [], [base, self.reg32(src)], side=True)

    def alu32(self, ins, t):
        op, a = ins.op, ins.args
        if not isinstance(a[0], Var):
            raise NotSupported("non-register destination")
        signed = t == "s32"
        srcs = a[1:]
        if 'sat' in ins.mods:
            return self.saturating_integer(ins)
        if op in ("div", "rem") and isinstance(srcs[1], ImmI):
            return self.divconst(ins, srcs[0], srcs[1].value, signed)
        x = self.reg32(srcs[0]) if srcs else None
        y = self.val32(srcs[1]) if len(srcs) > 1 else None
        d = self.new()
        # ALU operand rule: first source must be a register; second may be reg/imm/c[].
        e = lambda fmt, uses: self.emit(fmt, [d], uses)  # noqa: E731
        if op == "add":
            e("IADD3 {d0}, {s0}, {s1}, RZ", [x, y])
        elif op == "sub":
            if isinstance(y, I):
                e("IADD3 {d0}, {s0}, {s1}, RZ", [x, I((-y.v) & 0xFFFFFFFF)])
            else:
                e("IADD3 {d0}, {s0}, -{s1}, RZ", [x, y])
        elif op in ("and", "or", "xor"):
            lut = {"and": 0xC0, "or": 0xFC, "xor": 0x3C}[op]
            e(f"LOP3.LUT {{d0}}, {{s0}}, {{s1}}, RZ, {lut:#x}, !PT", [x, y])
        elif op == "not":
            e("LOP3.LUT {d0}, RZ, {s0}, RZ, 0x33, !PT", [x])
        elif op == "neg":
            e("IADD3 {d0}, RZ, -{s0}, RZ", [x])
        elif op in ("min", "max"):
            e(f"IMNMX{'' if signed else '.U32'} {{d0}}, {{s0}}, {{s1}}, {'PT' if op == 'min' else '!PT'}", [x, y])
        elif op == "mul" and "lo" in ins.mods:
            e("IMAD {d0}, {s0}, {s1}, RZ", [x, y])
        elif op == "mul" and "hi" in ins.mods:
            e(f"IMAD.HI{'' if signed else '.U32'} {{d0}}, {{s0}}, {{s1}}, RZ", [x, y])
        elif op == "mad" and "lo" in ins.mods:
            e("IMAD {d0}, {s0}, {s1}, {s2}", [x, y, self.reg32(srcs[2])])
        elif op in ("shl", "shr"):
            if isinstance(srcs[1], ImmI) and 0 <= srcs[1].value < 32:
                sh = srcs[1].value
                if op == "shl":
                    e(f"SHF.L.U32 {{d0}}, {{s0}}, {sh:#x}, RZ", [x])
                else:
                    e(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {sh:#x}, {{s0}}", [x])
            elif op == "shl":
                e("SHF.L.U32 {d0}, {s0}, {s1}, RZ", [x, self.reg32(srcs[1])])
            else:
                e(f"SHF.R.{'S32' if signed else 'U32'}.HI {{d0}}, RZ, {{s1}}, {{s0}}", [x, self.reg32(srcs[1])])
        else:
            raise NotSupported(f"{op}.{t}")
        self.bind(a[0].name, d)

    def shift64(self, ins, dtype):
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

    def saturating_integer(self, ins):
        """Signed add/sub saturation, including INT_MIN and aliased operands."""
        x, y = self.reg32(ins.args[1]), self.reg32(ins.args[2])
        value, overflow, sign, limit, result = (self.new() for _ in range(5))
        predicate = self.newp()
        self.emit('IADD3 {d0}, {s0}, ' + ('-' if ins.op == 'sub' else '') + '{s1}, RZ',
                  [value], [x, y])
        # add: (x ^ value) & (y ^ value); sub: (x ^ y) & (x ^ value).
        lut = 0x42 if ins.op == 'add' else 0x18
        self.emit(f'LOP3.LUT {{d0}}, {{s0}}, {{s1}}, {{s2}}, {lut:#x}, !PT',
                  [overflow], [x, y, value])
        self.emit('ISETP.LT.AND {p0}, PT, {s0}, 0x0, PT', [predicate], [overflow])
        self.emit('SHF.R.S32.HI {d0}, RZ, 0x1f, {s0}', [sign], [x])
        self.emit('LOP3.LUT {d0}, {s0}, {s1}, RZ, 0x3c, !PT', [limit], [sign, I(0x7fffffff)])
        self.emit('SEL {d0}, {s0}, {s1}, {p0}', [result], [limit, value, (predicate, False)])
        self.bind(ins.args[0].name, result)

    def divconst(self, ins, x_op, divisor, signed):
        if divisor < 2 or (signed and divisor >= (1 << 31)):
            raise NotSupported("div by 0/1/negative constant")
        seq = divconst.lower(ins.op, signed, divisor)
        names = {"x": self.reg32(x_op)}
        val = lambda o: I(o & 0xFFFFFFFF) if isinstance(o, int) else names[o]  # noqa: E731
        for o in seq.ops:
            k, s = o[0], o[2:]
            d = self.new()
            fmt = {
                "mov": "MOV {d0}, {s0}",
                "mulhi_u": "IMAD.HI.U32 {d0}, {s0}, {s1}, RZ",
                "mulhi_s": "IMAD.HI {d0}, {s0}, {s1}, RZ",
                "shr_u": f"SHF.R.U32.HI {{d0}}, RZ, {s[1]:#x}, {{s0}}" if k == "shr_u" else "",
                "shr_s": f"SHF.R.S32.HI {{d0}}, RZ, {s[1]:#x}, {{s0}}" if k == "shr_s" else "",
                "add": "IADD3 {d0}, {s0}, {s1}, RZ",
                "sub": "IADD3 {d0}, {s0}, -{s1}, RZ",
                "mad": "IMAD {d0}, {s0}, {s1}, {s2}",
            }.get(k)
            if k == "lea_hi_sign":
                t = self.new()
                self.emit("SHF.R.U32.HI {d0}, RZ, 0x1f, {s0}", [t], [val(s[1])])
                self.emit("IADD3 {d0}, {s0}, {s1}, RZ", [d], [val(s[0]), t])
            elif k in ("shr_u", "shr_s"):
                self.emit(fmt, [d], [val(s[0])])
            elif fmt:
                self.emit(fmt, [d], [val(x) for x in s])
            else:
                raise NotSupported(f"divconst op {k}")
            names[o[1]] = d
        self.bind(ins.args[0].name, names[seq.result])


# ---- passes ----------------------------------------------------------------------------

def _is_puse(x) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], PV)


def _flat(items):
    """Registers referenced by defs/uses: expands register groups, skips predicate
    operands ((PV, neg) tuples and bare PV)."""
    for x in items:
        if _is_puse(x) or isinstance(x, PV):
            continue
        if isinstance(x, tuple):
            yield from x
        else:
            yield x


def dce(ops: list[Op]) -> list[Op]:
    """Remove ops whose defs are never used and that have no side effect."""
    live: set = set()
    keep = []
    if any(op.label or op.target for op in ops):
        # With control flow, a backward pass isn't enough; use the CFG liveness instead.
        lv = cfg_liveness(ops)
        keep = [op for i, op in enumerate(ops)
                if op.side or op.label or op.target or (_defs(op) & lv[i])]
        return keep
    for op in reversed(ops):
        defs = _defs(op)
        if op.side or defs & live:
            keep.append(op)
            if op.guard is None:
                live -= defs
            live |= {u for u in _all_uses(op)}
    keep.reverse()
    return keep


def _all_uses(op):
    us = {u for u in _flat(op.uses) if isinstance(u, V)}
    us |= {u[0] for u in op.uses if _is_puse(u)}
    if op.guard is not None:
        us.add(op.guard[0])
    return us


def _defs(op):
    return set(_flat(op.defs)) | {d for d in op.defs if isinstance(d, PV)}


def cfg_liveness(ops):
    """live_out[i] for each op (set of V/PV live right after op i), by iterating
    dataflow to a fixpoint over the op-level CFG (fallthrough + branch targets).
    A guarded op does not kill its defs (it may not execute)."""
    n = len(ops)
    lab = {op.label: i for i, op in enumerate(ops) if op.label}
    succ = []
    for i, op in enumerate(ops):
        sc = []
        if op.target is not None:
            sc.append(lab[op.target])
            if op.guard is not None and i + 1 < n:
                sc.append(i + 1)
        elif op.fmt.startswith("EXIT") and op.guard is None:
            pass
        elif i + 1 < n:
            sc.append(i + 1)
        succ.append(sc)
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            out = set().union(*(live_in[j] for j in succ[i])) if succ[i] else set()
            op = ops[i]
            kill = _defs(op) if op.guard is None else set()
            inn = (out - kill) | _all_uses(op)
            if out != live_out[i] or inn != live_in[i]:
                live_out[i], live_in[i] = out, inn
                changed = True
    return live_out


def coalesce(ops: list[Op]) -> list[Op]:
    """Delete copies `d <- s` that feed a wide operand, by renaming d to s, when s can
    live in d's register slot: s is a scalar (in no other group), not already
    coalesced into another slot, and not a member of the same tuple twice."""
    in_group: dict[V, tuple] = {}
    for op in ops:
        for t in list(op.defs) + [u for u in op.uses if isinstance(u, tuple)]:
            if isinstance(t, tuple) and not _is_puse(t):
                for v in t:
                    in_group.setdefault(v, t)
    ren: dict[V, V] = {}
    taken: set = set()
    ndef: dict = {}
    for op in ops:
        for v in _flat(op.defs):
            ndef[v] = ndef.get(v, 0) + 1
    for op in ops:
        if not op.copy:
            continue
        d, s_ = op.defs[0], op.uses[0]
        if not isinstance(s_, V) or s_ in in_group or s_ in taken:
            continue
        # Only merge single-definition values: a loop-carried home register (defined by
        # several copies) must keep its identity, or values crossing the back edge are lost.
        if ndef.get(d, 0) != 1 or ndef.get(s_, 0) != 1:
            continue
        ren[d] = s_
        taken.add(s_)
    if not ren:
        return ops
    def resolve(v):
        while v in ren:
            v = ren[v]
        return v
    def sub(x):
        return x if _is_puse(x) else (tuple(resolve(v) for v in x) if isinstance(x, tuple) else resolve(x))
    out = []
    for op in ops:
        if op.copy and op.defs[0] in ren:
            continue
        op.defs = [sub(d) for d in op.defs]
        op.uses = [sub(u) for u in op.uses]
        out.append(op)
    return out


@dataclass
class Group:
    regs: tuple          # tuple of V, consecutive + aligned to len
    start: int
    end: int


def _groups(ops):
    grp: dict[V, tuple] = {}
    for op in ops:
        for t in list(op.defs) + [u for u in op.uses if isinstance(u, tuple)]:
            if isinstance(t, tuple) and not _is_puse(t):
                if any(v in grp and grp[v] != t for v in t):
                    raise NotSupported("value used in two different register groups")
                for v in t:
                    grp[v] = t
    for op in ops:
        for v in _flat(op.defs):
            grp.setdefault(v, (v,))
        for v in _flat(op.uses):
            if isinstance(v, V):
                grp.setdefault(v, (v,))
    return grp


def _intervals(ops):
    """[first, last] op index where each V is defined, used, or live (CFG liveness),
    so values carried around a loop's back edge stay allocated for the whole loop."""
    start, end = {}, {}
    lv = cfg_liveness(ops) if any(op.target for op in ops) else None
    for i, op in enumerate(ops):
        here = list(_flat(op.defs)) + [u for u in _flat(op.uses) if isinstance(u, V)]
        if lv is not None:
            here += [v for v in lv[i] if isinstance(v, V)]
        for v in here:
            if isinstance(v, V):
                start.setdefault(v, i)
                start[v] = min(start[v], i)
                end[v] = max(end.get(v, i), i)
    return start, end


def _alloc(ops, r1, hints, max_reg):
    """Linear scan over groups in start order. A register is free for value v if
    the previous value in it ended at or before v's start: a value last read by an
    instruction may share a register with that instruction's result (operands are
    read before results are written; ptxas does this too, e.g. IMAD.WIDE R4, R4, ..)."""
    grp = _groups(ops)
    start, end = _intervals(ops)
    src_of = {op.defs[0]: op.uses[0] for op in ops if op.copy and isinstance(op.uses[0], V)}
    groups = sorted(set(grp.values()), key=lambda g: (min(start[v] for v in g), -len(g)))
    phys: dict[V, int] = {r1: 1}
    last_end: dict[int, int] = {1: 10 ** 9}
    for g in groups:
        if g == (r1,):
            continue
        n = len(g)
        prefs = []
        for k, v in enumerate(g):
            if v in hints:
                prefs.append(hints[v] - k)
            if v in src_of and src_of[v] in phys:
                prefs.append(phys[src_of[v]] - k)
        for b in [p for p in prefs if p >= 0 and p % n == 0] + list(range(0, max_reg - n + 1, n)):
            if all(last_end.get(b + k, -1) <= start[v] for k, v in enumerate(g)):
                for k, v in enumerate(g):
                    phys[v] = b + k
                    last_end[b + k] = end[v]
                break
        else:
            raise NotSupported("out of registers")
    return phys


def allocate(ops: list[Op], r1: V, max_reg: int = 250) -> dict[V, int]:
    """Allocate, then re-allocate with copy-affinity hints (a copy's source prefers its
    destination's register) until no improvement; keep the allocation that leaves the
    fewest real copies."""
    copies = [(op.defs[0], op.uses[0]) for op in ops if op.copy and isinstance(op.uses[0], V)]
    hints: dict[V, int] = {}
    best = None
    for _ in range(4):
        phys = _alloc(ops, r1, hints, max_reg)
        left = sum(phys[f] != phys[v] for f, v in copies)
        if best is None or left < best[0]:
            best = (left, phys)
        new = dict(hints)
        for f, v in copies:
            if phys[f] != phys[v]:
                new[v] = phys[f]
        if new == hints:
            break
        hints = new
    return best[1]


def allocate_preds(ops: list[Op]) -> dict:
    """PV -> P0..P6 by interval coloring (same interval rule as registers)."""
    lv = cfg_liveness(ops) if any(op.target for op in ops) else None
    span = {}
    for i, op in enumerate(ops):
        here = [d for d in op.defs if isinstance(d, PV)] + [p for p in _all_uses(op) if isinstance(p, PV)]
        if lv is not None:
            here += [v for v in lv[i] if isinstance(v, PV)]
        for v in here:
            a, b = span.get(v, (i, i))
            span[v] = (min(a, i), max(b, i))
    out, last = {}, {}
    for v, (a, b) in sorted(span.items(), key=lambda kv: kv[1][0]):
        for r in range(7):
            if last.get(r, -1) <= a:
                out[v], last[r] = r, b
                break
        else:
            raise NotSupported("out of predicate registers")
    return out


def lower(ops: list[Op], phys: dict[V, int]) -> list[tuple[str, bool, list[int], list[int]]]:
    """-> [(text, var_lat, reads, writes)] with physical registers. Labels become
    ("<label>", ...) markers resolved by compile_kernel's branch encoding."""
    pphys = allocate_preds(ops)

    def pname(x):
        pv, neg = x if isinstance(x, tuple) else (x, False)
        return ("!" if neg else "") + f"P{pphys[pv]}"

    def name(x):
        if isinstance(x, PV):
            return f"P{pphys[x]}"
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], PV):
            return pname(x)
        if isinstance(x, V):
            return f"R{phys[x]}"
        if isinstance(x, tuple):
            return f"R{phys[x[0]]}"
        if isinstance(x, C):
            return f"c[0x0][{x.off:#x}]"
        if isinstance(x, I):
            return f"{x.v:#x}"
        raise TypeError(x)
    out = []
    for op in ops:
        if op.label:
            out.append((f"<label {op.label}>", False, [], []))
            continue
        pdefs = [d for d in op.defs if isinstance(d, PV)]
        rdefs = [d for d in op.defs if not isinstance(d, PV)]
        kw = {f"d{i}": name(d) for i, d in enumerate(rdefs)}
        kw.update({f"p{i}": name(d) for i, d in enumerate(pdefs)})
        # predicate *uses* referenced as {p0} after defs (SEL) are tuples (PV, neg)
        puses = [u for u in op.uses if isinstance(u, tuple) and len(u) == 2 and isinstance(u[0], PV)]
        ruses = [u for u in op.uses if u not in puses]
        kw.update({f"s{i}": name(u) for i, u in enumerate(ruses)})
        if puses and not pdefs:
            kw.update({f"p{i}": name(u) for i, u in enumerate(puses)})
        kw.update({f"q{i}": name(u) for i, u in enumerate(puses)})
        kw["target"] = op.target or ""
        text = op.fmt.format(**kw)
        if op.guard is not None:
            text = f"@{pname(op.guard)} {text}"
        if op.copy and len(op.defs) == 1 and isinstance(op.uses[0], V) and \
                phys[op.defs[0]] == phys[op.uses[0]]:
            continue  # coalesced copy
        reads = [phys[v] for v in _flat(op.uses) if isinstance(v, V)]
        writes = [phys[v] for v in _flat(op.defs) if isinstance(v, V)]
        out.append((text, op.var_lat, reads, writes))
    return out


def compile_kernel(k: Kernel):
    """PTX kernel -> (sass texts with control bits, stats). Uses the v0 scheduler."""
    from openassembler.backend.compile import Out, schedule

    s = Selector(k)
    ops = coalesce(dce(s.run()))
    from openassembler.backend.spill import allocate_kernel
    ops,phys,resources=allocate_kernel(s,ops)
    lowered = lower(ops, phys)
    # Labels: record which instruction index each label precedes, then drop markers.
    labels, body = {}, []
    for t, vl, r, w in lowered:
        if t.startswith("<label "):
            labels[t[7:-1]] = len(body)
        else:
            body.append((t, vl, r, w))
    code = schedule([Out(t, vl, r, w) for t, vl, r, w in body], labels=labels)
    return code, {"ir_ops": len(ops), "regs": max(phys.values()) + 1, "labels": labels,
                  "smem": s.smem_bytes, **resources}


def encode_code(enc, code, labels) -> list[int]:
    """Encode scheduled code; branches get their target address from `labels`."""
    import re as _re

    words = []
    for i, (t, ctl) in enumerate(code):
        m = _re.search(r"`\((\S+?)\)", t)
        if m and any(op in t.split() for op in ("BRA", "BSSY")):
            words.append(enc.encode_branch(t, 16 * i, 16 * labels[m[1]], ctl))
        elif m and t.split()[1:2] == ["BRA"]:
            words.append(enc.encode_branch(t, 16 * i, 16 * labels[m[1]], ctl))
        else:
            words.append(enc.encode(t, ctl))
    return words
