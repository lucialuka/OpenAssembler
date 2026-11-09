"""Reference PTX interpreter (one CTA, lock-step warps).

Semantics come from the PTX ISA spec, not from SASS. Values are stored as raw
bits per register; each instruction's type suffix says how to read them.
Undefined-by-spec results (div/rem by zero, shift >= width handled per spec)
follow the spec where it defines them and raise `Undefined` where it doesn't,
so the harness can skip those lanes instead of encoding ptxas's choice as truth.

Execution model: a warp runs in lock-step with an active mask. Divergent
branches are handled by a reconvergence stack (immediate post-dominator =
the lowest pending pc, valid for the reducible, forward-structured code in our
probes); collectives see the active lanes.

CTA barriers (bar.sync / barrier.sync / bar.arrive / bar.red, any warp count)
are cooperative, mirroring the SASS emulator's Machine.run: each warp runs
until all its lanes have exited or are parked at a barrier; a barrier
completes when its expected arrivals are in -- for the full-CTA form, every
non-exited thread of the CTA (sm_70+ per-thread semantics: exited threads
count as arrived, which is what makes `if (i >= n) return; ... __syncthreads()`
work) -- and its waiters then resume. See Interp._bar_arrive for the exact
rules and for what is rejected instead of guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from openassembler.emu import fp
from openassembler.emu.machine import Memory
from openassembler.ptx.parse import Addr, ImmF, ImmI, Instr, Module, PairDst, Var, Vec


class Undefined(Exception):
    """Result is undefined by the PTX spec (e.g. integer division by zero)."""


class NotModeled(Exception):
    pass


M = lambda bits: (1 << bits) - 1  # noqa: E731
_TBITS = {"b8": 8, "u8": 8, "s8": 8, "b16": 16, "u16": 16, "s16": 16, "f16": 16, "bf16": 16,
          "b32": 32, "u32": 32, "s32": 32, "f32": 32, "b64": 64, "u64": 64, "s64": 64, "f64": 64,
          "f16x2": 32, "bf16x2": 32, "pred": 1}
_TYPES = set(_TBITS)


def sext(v: int, bits: int) -> int:
    v &= M(bits)
    return v - (1 << bits) if v >> (bits - 1) else v


def types_of(ins: Instr) -> list[str]:
    return [m for m in ins.mods if m in _TYPES]


@dataclass
class Lane:
    tid: tuple[int, int, int]
    lane: int
    regs: dict[str, int] = field(default_factory=dict)
    local: Memory = field(default_factory=Memory)
    done: bool = False
    undefined: bool = False   # some result on this lane was spec-undefined
    warp: int = 0             # index into Interp.warps
    flat: int = 0             # linear thread index in the CTA (x fastest)


@dataclass
class _Arrival:
    ln: Lane
    pc: int
    ins: Instr
    waits: bool               # bar.sync/bar.red park the thread; bar.arrive does not
    pred: bool = False        # bar.red: this thread's (possibly negated) predicate
    aligned: bool = True      # .aligned form: the whole warp must use this one instruction


@dataclass
class _Barrier:
    """One of the CTA's 16 named barriers, between two completions."""
    count: int | None         # expected threads; None = all non-exited threads of the CTA
    red: str | None           # "popc"/"and"/"or" for bar.red, None for sync/arrive
    arrivals: list[_Arrival] = field(default_factory=list)
    n_arrived: int = 0        # counted form: warp-granular arrival count (see _bar_arrive)


@dataclass
class Launch:
    block: tuple[int, int, int] = (32, 1, 1)
    grid: tuple[int, int, int] = (1, 1, 1)
    ctaid: tuple[int, int, int] = (0, 0, 0)
    params: dict[str, int] = field(default_factory=dict)


class Interp:
    def __init__(self, mod: Module, kernel: str, launch: Launch, gmem: Memory | None = None):
        self.mod, self.k, self.L = mod, mod.kernels[kernel], launch
        self.gmem = gmem or Memory()
        self.smem = Memory()
        self.cmem = Memory()
        # Symbol addresses: shared from 0, const in its own space, globals in a high window.
        self.sym: dict[str, tuple[str, int]] = {}
        offs = {"shared": 0, "const": 0, "global": 0x7E00_0000_0000, "local": 0}
        for d in self.mod.decls + self.k.decls:
            base = (offs[d.space] + d.align - 1) // d.align * d.align
            self.sym[d.name] = (d.space, base)
            offs[d.space] = base + d.size
        # Kernel-scope .local declarations may appear after first use in our PTX parser's view.
        self._local_decls = {d.name: self.sym[d.name][1] for d in self.k.decls if d.space == "local"}
        bx, by, bz = launch.block
        tids = [(x, y, z) for z in range(bz) for y in range(by) for x in range(bx)]
        self.warps = [[Lane(t, i, warp=w // 32, flat=w + i) for i, t in enumerate(tids[w:w + 32])]
                      for w in range(0, len(tids), 32)]
        self.steps = 0
        self.bars: dict[int, _Barrier] = {}

    # ---- values ------------------------------------------------------------------

    def get(self, ln: Lane, op, bits: int) -> int:
        if isinstance(op, ImmI):
            return op.value & M(bits)
        if isinstance(op, ImmF):
            if op.width == bits:
                return op.bits
            # 0f literal used with a 64-bit type: convert value exactly
            return fp.encode(fp.decode(op.bits, fp.F32), fp.F64)
        if isinstance(op, Var):
            n = op.name
            if n in self.sym:
                return self.sym[n][1]
            if n in self._local_decls:
                return self._local_decls[n]
            if n.startswith("%") and n not in ln.regs and n.split(".")[0] in SREGS:
                return self.sreg(ln, n) & M(bits)
            if n not in ln.regs:
                if ln.undefined:
                    return 0  # value is already poisoned for this lane; result is skipped
                raise NotModeled(f"read of undefined register {n}")
            return ln.regs[n] & M(bits)
        raise NotModeled(f"operand {op}")

    def put(self, ln: Lane, op, value: int, bits: int) -> None:
        if isinstance(op, Var) and op.name != "_":
            ln.regs[op.name] = value & M(bits)

    def pred(self, ln: Lane, name: str) -> bool:
        neg = name.startswith("!")
        return bool(ln.regs.get(name.lstrip("!"), 0)) != neg

    def sreg(self, ln: Lane, name: str) -> int:
        L, (x, y, z) = self.L, ln.tid
        base, _, comp = name.partition(".")
        idx = "xyz".index(comp) if comp else 0
        table = {"%tid": ln.tid[idx], "%ntid": L.block[idx], "%ctaid": L.ctaid[idx], "%nctaid": L.grid[idx],
                 "%laneid": ln.lane, "%lanemask_eq": 1 << ln.lane, "%lanemask_lt": (1 << ln.lane) - 1,
                 "%lanemask_le": (1 << (ln.lane + 1)) - 1, "%lanemask_ge": M(32) & ~((1 << ln.lane) - 1),
                 "%lanemask_gt": M(32) & ~((1 << (ln.lane + 1)) - 1)}
        if base not in table:
            raise NotModeled(f"special register {name}")
        return table[base]

    def addr(self, ln: Lane, a: Addr) -> int:
        if a.base in self.sym:
            return self.sym[a.base][1] + a.off
        if a.base in [p for _, p in self.k.params]:
            raise NotModeled("address of param")
        return (ln.regs[a.base] + a.off) & M(64)

    def space_mem(self, ins: Instr, ln: Lane):
        for s, mem in (("shared", self.smem), ("local", ln.local), ("const", self.cmem), ("global", self.gmem)):
            if s in ins.mods:
                return mem
        if "param" in ins.mods:
            return None
        return self.gmem  # generic addressing: probes only use global pointers generically

    # ---- execution -------------------------------------------------------------------

    def run(self, max_steps: int = 2_000_000) -> None:
        from openassembler.ptx import semantics as S

        # Per warp: {pc: lanes}. Always run the entry with the lowest pc; entries
        # reaching the same pc merge (reconvergence). A lane parked at a CTA barrier
        # is in no entry until the barrier completes and puts it back at pc + 1.
        self._groups: list[dict[int, list[Lane]]] = [{0: list(w)} for w in self.warps]
        progress = True
        while progress:
            progress = False
            for groups in self._groups:
                while groups:
                    pc = min(groups)
                    lanes = [ln for ln in groups.pop(pc) if not ln.done]
                    if not lanes:
                        continue
                    progress = True
                    if pc >= len(self.k.body):
                        raise NotModeled("fell off the end of the kernel")
                    ins = self.k.body[pc]
                    if S.cta_barrier(ins):
                        nxt = self._bar_arrive(ins, lanes, pc)
                    else:
                        nxt = S.execute(self, ins, lanes, pc)
                        if ins.op in ("ret", "exit") and self.bars:
                            self._bar_try_all()      # exited threads count as arrived
                    for target, tl in nxt.items():
                        groups.setdefault(target, []).extend(tl)
                    self.steps += 1
                    if self.steps > max_steps:
                        raise NotModeled(f"step limit exceeded ({max_steps} instruction groups)")
        stuck = {b: br for b, br in self.bars.items() if any(a.waits for a in br.arrivals)}
        if stuck:
            raise NotModeled("CTA barrier deadlock: " + "; ".join(
                self._bar_describe(b, br) for b, br in sorted(stuck.items())))

    # ---- CTA barriers ------------------------------------------------------------------

    def _bar_arrive(self, ins: Instr, lanes: list[Lane], pc: int) -> dict[int, list[Lane]]:
        """Threads `lanes` (one warp, same pc) execute bar/barrier .sync/.arrive/.red.

        Rules (PTX ISA "bar, barrier", sm_70+ per-thread semantics), and where we
        refuse instead of guessing:
          * guard-false threads don't execute it and just move on;
          * full-CTA form (no thread count): completes when every non-exited
            thread of the CTA has arrived; exited threads count as arrived;
          * counted form (bar.sync b, n / bar.arrive b, n): the spec counts
            arrivals in warp-size units, so a counted arrival must come from a
            whole, non-exited, converged warp -- anything else is NotModeled, as
            is a count that isn't a positive multiple of 32 or exceeds the CTA;
          * .aligned (all bar.* forms, barrier.*.aligned): every thread of a warp
            must execute the same barrier instruction; one warp arriving from
            different pcs is rejected when the barrier completes (UB per spec).
            barrier.sync (non-aligned) allows it;
          * a thread may not arrive twice before the barrier completes;
          * bar.red reduces over the threads that arrived (exited ones have no
            predicate): popc.u32 / and.pred / or.pred. Mixing reductions, or a
            reduction and a plain sync, on one barrier id is rejected."""
        from openassembler.ptx import semantics as S

        on = [ln for ln in lanes if self.guard(ln, ins)]
        off = [ln for ln in lanes if not self.guard(ln, ins)]
        nxt: dict[int, list[Lane]] = {pc + 1: off} if off else {}
        if not on:
            return nxt
        f = S.barrier_form(self, ins, on)
        n = self._nthreads()
        if f.count is not None and (f.count <= 0 or f.count % 32 or f.count > -(-n // 32) * 32):
            raise NotModeled(f"{_txt(ins)}: thread count {f.count} is not a positive multiple of 32 "
                             f"covered by the CTA ({n} threads)")
        br = self.bars.get(f.bid)
        if br is None:
            br = self.bars[f.bid] = _Barrier(f.count, f.red)
        elif (br.count, br.red) != (f.count, f.red):
            raise NotModeled(f"{_txt(ins)}: barrier {f.bid} is already pending as count={br.count}, "
                             f"red={br.red}; mixing barrier forms on one id is not modeled")
        seen = {id(a.ln) for a in br.arrivals}
        if any(id(ln) in seen for ln in on):
            raise NotModeled(f"{_txt(ins)}: a thread arrived twice at barrier {f.bid} before it completed")
        if f.count is not None:
            w = self.warps[on[0].warp]
            if len(on) != len(w) or any(ln.done for ln in w):
                raise NotModeled(f"{_txt(ins)}: counted barrier reached by a partial warp ({len(on)} "
                                 f"of {len(w)} threads); warp-granular counting is not modeled")
            br.n_arrived += 32
        for ln in on:
            br.arrivals.append(_Arrival(ln, pc, ins, f.waits, f.preds.get(id(ln), False), f.aligned))
        if not f.waits:
            nxt.setdefault(pc + 1, []).extend(on)
        self._bar_try(f.bid)
        return nxt

    def _nthreads(self) -> int:
        return sum(len(w) for w in self.warps)

    def _bar_complete(self, br: _Barrier) -> bool:
        if br.count is not None:
            return br.n_arrived >= br.count       # increments of 32, so exactly at count
        arrived = {id(a.ln) for a in br.arrivals}
        return all(ln.done or id(ln) in arrived for w in self.warps for ln in w)

    def _bar_try_all(self) -> None:
        for b in list(self.bars):
            self._bar_try(b)

    def _bar_try(self, bid: int) -> None:
        br = self.bars.get(bid)
        if br is None or not br.arrivals or not self._bar_complete(br):
            return
        pcs: dict[int, set[int]] = {}
        for a in br.arrivals:
            if a.aligned:
                pcs.setdefault(a.ln.warp, set()).add(a.pc)
        for w, ps in sorted(pcs.items()):
            if len(ps) > 1:
                lines = sorted(self.k.body[p].line for p in ps)
                raise NotModeled(f"barrier {bid}: warp {w} reached this .aligned barrier from different "
                                 f"instructions (PTX statements {lines}), undefined behavior per the PTX ISA; "
                                 "use barrier.sync (non-aligned) for divergent code")
        del self.bars[bid]
        r = 0
        if br.red is not None:
            vals = [a.pred for a in br.arrivals]
            r = {"popc": sum(vals), "and": int(all(vals)), "or": int(any(vals))}[br.red]
        for a in br.arrivals:
            if br.red is not None:
                self.put(a.ln, a.ins.args[0], r, 32 if br.red == "popc" else 1)
            if a.waits:
                self._groups[a.ln.warp].setdefault(a.pc + 1, []).append(a.ln)

    def _bar_describe(self, bid: int, br: _Barrier) -> str:
        if br.count is not None:
            return (f"barrier {bid} expects {br.count} threads but only {br.n_arrived} arrived "
                    "and no other thread can make progress")
        arrived = {id(a.ln) for a in br.arrivals}
        missing = [ln for w in self.warps for ln in w if not ln.done and id(ln) not in arrived]
        eg = ", ".join(str(ln.tid) for ln in missing[:4])
        return (f"barrier {bid}: {len(br.arrivals)} threads wait but {len(missing)} non-exited threads "
                f"never arrive (e.g. tid {eg}); they wait at a different barrier or diverged around it")

    def guard(self, ln: Lane, ins: Instr) -> bool:
        if ins.guard is None:
            return True
        name, neg = ins.guard
        return bool(ln.regs.get(name, 0)) != neg


def _txt(ins: Instr) -> str:
    return ".".join((ins.op,) + ins.mods) + (f" (PTX statement {ins.line})" if ins.line else "")


SREGS = {"%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%lanemask_eq", "%lanemask_lt",
         "%lanemask_le", "%lanemask_ge", "%lanemask_gt"}


def run_kernel(src_mod: Module, kernel: str, launch: Launch, gmem: Memory) -> Interp:
    it = Interp(src_mod, kernel, launch, gmem)
    it.run()
    return it


__all__ = ["Interp", "Launch", "Lane", "Undefined", "NotModeled", "run_kernel", "sext", "M",
           "Fraction", "math", "PairDst", "Vec"]
