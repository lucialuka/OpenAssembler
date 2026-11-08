"""SM80 tensor PTX selection and source-order dependency scheduling.

This is the validated A100 tensor path: dense m16n8k16 FP16/BF16 with FP32
accumulators, matrix shared loads and 16-byte asynchronous operand copies.
Scalar-only kernels retain their existing backend. Unsupported tensor forms
raise NotSupported and never silently lower to different arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses

from openassembler.backend import ra
from openassembler.backend.tensor_encoding import TensorEncoder
from openassembler.ptx.parse import Addr, ImmI, Var, Vec


@dataclass(frozen=True)
class TensorConfig:
    """Validated per-call scheduling policy; no mutable module switches."""
    issue_gap: int = 8
    epoch_drain: int = 15
    reuse: bool = True
    pointer_mode: str = "legacy"
    pointer_advance: str = "late"

    def __post_init__(self):
        if self.issue_gap != 8 or self.epoch_drain != 15:
            raise ValueError("the production SM80 tensor path requires HMMA8 and drain15")
        if self.pointer_mode not in ("legacy", "direct"):
            raise ValueError("tensor pointer mode must be legacy or direct")
        if self.pointer_advance not in ("late", "early", "auto"):
            raise ValueError("tensor pointer advance must be late, early or auto")


def needs_tensor(kernel) -> bool:
    """Detect tensor/copy PTX, including unsupported forms for explicit refusal."""
    return any(ins.op in ("mma", "wgmma", "ldmatrix")
               or ins.op == "cp" and ins.mods[:1] == ("async",)
               for ins in kernel.body)


def loop_carried_registers(kernel):
    """Exact liveness for our single-latch-loop PTX subset, including vector defs.

    Other control-flow shapes are refused: this optimization does not replace
    the scalar backend's general guarded-definition rules.
    """
    from openassembler.ptx.parse import PairDst
    body = kernel.body
    branches = [(i, ins) for i, ins in enumerate(body) if ins.op == "bra"]
    if len(branches) != 1:
        raise ra.NotSupported("local-home optimization requires exactly one loop branch")
    latch, branch = branches[0]
    head = kernel.labels.get(branch.args[0].name, latch + 1)
    if head >= latch or branch.guard is None or any(
            ins.guard is not None and i != latch for i, ins in enumerate(body)):
        raise ra.NotSupported("local-home optimization requires a single unguarded loop body")

    def regs(operand):
        if isinstance(operand, Var):
            return {operand.name} if operand.name.startswith("%") else set()
        if isinstance(operand, Addr):
            return {operand.base} if operand.base.startswith("%") else set()
        if isinstance(operand, Vec):
            return set().union(*(regs(e) for e in operand.elems))
        if isinstance(operand, PairDst):
            return regs(operand.a) | regs(operand.b)
        return set()
    uses, defs = [], []
    for ins in body:
        writes = ins.op not in ("st", "bra", "ret", "bar", "cp", "red", "membar")
        dst = regs(ins.args[0]) if writes and ins.args else set()
        srcs = ins.args[1:] if writes else ins.args
        read = set().union(*(regs(a) for a in srcs))
        if ins.guard:
            read.add(ins.guard[0])
        uses.append(read); defs.append(dst)
    live = [set() for _ in body]
    changed = True
    while changed:
        changed = False
        for i in range(len(body) - 1, -1, -1):
            successors = [] if body[i].op == "ret" else ([i + 1] if i + 1 < len(body) else [])
            if i == latch:
                successors.append(head)
            after = set().union(*(live[j] for j in successors))
            before = uses[i] | (after - defs[i])
            if before != live[i]:
                live[i] = before
                changed = True
    return live[head] & set().union(*defs[head:latch + 1])


class TensorSelector(ra.Selector):
    def __init__(self, kernel, config=None):
        super().__init__(kernel)
        self.config = config or TensorConfig()

    def wide(self, group):
        # A loop-carried 64-bit home already has stable component identities.
        # Keeping this pair as an IR operand lets allocation impose alignment
        # directly and exposes LDGSTS's true address WAR to the scheduler. The
        # generic wide() path instead copies both halves to a temporary pair.
        # This mode is explicit: it wins on measured MLP shapes but not on all
        # large GEMMs. Non-home pairs and fragment quads retain generic rules.
        if (self.config.pointer_mode == "direct" and len(group) == 2
                and group in getattr(self, "homes", {}).values()):
            return group
        return super().wide(group)

    def _init_tensor_homes(self):
        if getattr(self, "_tensor_homes_ready", False):
            return
        self._tensor_homes_ready = True
        self.tensor_groups = {}
        if self.has_cfg:
            carried = loop_carried_registers(self.k)
            self.homes = {name: value for name, value in self.homes.items() if name in carried}
        if not self.has_cfg:
            return
        for ins in self.k.body:
            if ins.op != "mma":
                continue
            if not ins.args or not isinstance(ins.args[0], Vec):
                raise ra.NotSupported("MMA destination must be a register vector")
            names = tuple(v.name for v in ins.args[0].elems if isinstance(v, Var))
            if len(names) != 4 or len(set(names)) != 4:
                raise ra.NotSupported("MMA needs four distinct accumulator destinations")
            for name in names:
                old = self.tensor_groups.setdefault(name, names)
                if old != names:
                    raise ra.NotSupported("overlapping MMA accumulator groups")
                if self.homes.get(name) is None:
                    self.homes[name] = self.new()

    def select(self, ins):
        self._init_tensor_homes()
        if (ins.guard is None and ins.op=='add' and ins.mods in (('u64',),('s64',))
                and len(ins.args)==3 and isinstance(ins.args[0],Var) and ins.args[0]==ins.args[1]
                and isinstance(ins.args[2],ImmI) and 0<=ins.args[2].value<2**31):
            name=ins.args[0].name
            pair=self.pair(name)
            if self.homes.get(name)==pair and all(isinstance(v,ra.V) for v in pair):
                lo,hi=pair;carry=self.newp()
                self.emit('IADD3 {d0}, {p0}, {s0}, {s1}, RZ',[lo,carry],[lo,ra.I(ins.args[2].value)])
                self.emit('IADD3.X {d0}, {s0}, RZ, RZ, {q0}, !PT',[hi],[hi,(carry,False)])
                self.env[name]=pair
                return
        if ins.op == "mma":
            return self.mma(ins)
        if ins.op == "ldmatrix":
            return self.ldmatrix(ins)
        if ins.op == "cp":
            return self.async_copy(ins)
        return super().select(ins)

    def load_global(self, ins, t):
        if (ins.guard is None and ins.mods == ("global", "v2", "f32")
                and len(ins.args) == 2 and isinstance(ins.args[0], Vec)
                and len(ins.args[0].elems) == 2
                and all(isinstance(value, Var) for value in ins.args[0].elems)):
            base, off = self._addr(ins.args[1])
            group = (self.new(), self.new())
            self.emit(f"LDG.E.64 {{d0}}, [{{s0}}.64{off}]", [group], [base], var_lat=True)
            for dest, value in zip(ins.args[0].elems, group):
                self.bind(dest.name, value)
            return
        return super().load_global(ins, t)

    def async_copy(self, ins):
        if ins.mods != ("async", "cg", "shared", "global"):
            return self._copy_group(ins)
        if (ins.guard is not None or len(ins.args) != 3
                or not all(isinstance(arg, Addr) for arg in ins.args[:2])
                or not isinstance(ins.args[2], ImmI) or ins.args[2].value != 16):
            raise ra.NotSupported("expected unguarded 16-byte async copy")
        shared, source = ins.args[:2]
        if (not -524288 <= shared.off < 524288 or shared.off % 16
                or not -2048 <= source.off < 2048 or source.off % 16):
            raise ra.NotSupported("LDGSTS displacement range/alignment")
        sr = self.reg32(Var(shared.base))
        gr = self.wide(self.pair(source.base))
        fmt = lambda off: (f"+{off:#x}" if off >= 0 else f"-{abs(off):#x}") if off else ""
        self.emit(f"LDGSTS.E.BYPASS.128 [{{s0}}{fmt(shared.off)}], "
                  f"[{{s1}}.64{fmt(source.off)}]", [], [sr, gr], side=True)

    def _copy_group(self, ins):
        if ins.guard is not None:
            raise ra.NotSupported("tensor async-copy groups must be unguarded")
        if ins.mods == ("async", "commit_group") and not ins.args:
            self.emit("LDGDEPBAR", side=True)
            return
        if (ins.mods == ("async", "wait_group") and len(ins.args) == 1
                and isinstance(ins.args[0], ImmI) and ins.args[0].value in (0,1,2)):
            self.emit(f"DEPBAR.LE SB0, {ins.args[0].value:#x}", side=True)
            return
        raise ra.NotSupported("only 16-byte cp.async.cg, commit_group and wait_group 0/1/2 are supported")

    def mma(self, ins):
        # Reject every unimplemented shape, type and layout, rather than silently
        # discarding a modifier with arithmetic or synchronization semantics.
        mods = ins.mods
        prefix = ("sync", "aligned", "m16n8k16", "row", "col", "f32")
        if (mods[:6] != prefix or len(mods) != 9 or mods[6] not in ("f16", "bf16")
                or mods[7] != mods[6] or mods[8] != "f32" or ins.guard is not None):
            raise ra.NotSupported(f"unsupported SM80 MMA modifiers/guard: {mods}")
        if len(ins.args) != 4 or any(not isinstance(x, Vec) for x in ins.args):
            raise ra.NotSupported("MMA requires D/A/B/C register vectors")
        d, a, b, c = ins.args
        if tuple(len(x.elems) for x in (d, a, b, c)) != (4, 4, 2, 4):
            raise ra.NotSupported("m16n8k16 fragment widths must be 4/4/2/4")
        if any(not isinstance(x, Var) for x in d.elems):
            raise ra.NotSupported("MMA destination must contain registers")
        if len({value.name for value in d.elems}) != 4:
            raise ra.NotSupported("MMA needs four distinct accumulator destinations")
        aa = self.wide(tuple(self.reg32(v) for v in a.elems))
        bv = tuple(self.reg32(v) for v in b.elems)
        # A matrix x4 result contains two aligned B pairs. Passing the pair as
        # a new tuple would conflict with the enclosing quad's allocation.
        # Print its first scalar and explicitly retain both scalar readers.
        quads = [group for op in self.ops for group in op.defs
                 if isinstance(group, tuple) and len(group) == 4]
        native_half = any(bv == group[:2] or bv == group[2:] for group in quads)
        bb = bv[0] if native_half else self.wide(bv)
        names = tuple(v.name for v in d.elems)
        if self.has_cfg:
            dd = tuple(self.homes[n] for n in names)
            cc0 = tuple(self.reg32(v) for v in c.elems)
            cc = cc0 if cc0 == dd else self.wide(cc0)
        else:
            dd = tuple(self.new() for _ in range(4))
            cc = self.wide(tuple(self.reg32(v) for v in c.elems))
        suffix = ".BF16" if mods[6] == "bf16" else ""
        self.emit(f"HMMA.16816.F32{suffix} {{d0}}, {{s0}}, {{s1}}, {{s2}}",
                  [dd], [aa, bb, cc] + ([bv[1]] if native_half else []))
        for name, value in zip(names, dd):
            if self.has_cfg:
                self.env[name] = value
            else:
                self.bind(name, value)

    def ldmatrix(self, ins):
        allowed = {
            ("sync", "aligned", "m8n8", "x4", "shared", "b16"): (4, "M88"),
            ("sync", "aligned", "m8n8", "x2", "shared", "b16"): (2, "M88"),
            ("sync", "aligned", "m8n8", "x2", "trans", "shared", "b16"): (2, "MT88"),
        }
        if ins.mods not in allowed or ins.guard is not None:
            raise ra.NotSupported(f"unsupported SM80 LDSM modifiers/guard: {ins.mods}")
        width, layout = allowed[ins.mods]
        if (len(ins.args) != 2 or not isinstance(ins.args[0], Vec)
                or len(ins.args[0].elems) != width or not isinstance(ins.args[1], Addr)):
            raise ra.NotSupported("ldmatrix fragment/address")
        ad = ins.args[1]
        if ad.off < 0 or ad.off % 16 or ad.off >= 16384:
            raise ra.NotSupported("ldmatrix offset must be aligned and below 16 KiB")
        address = self.reg32(Var(ad.base))
        group = tuple(self.new() for _ in range(width))
        offset = f"+{ad.off:#x}" if ad.off else ""
        self.emit(f"LDSM.16.{layout}.{width} {{d0}}, [{{s0}}{offset}]", [group], [address], var_lat=True)
        for dst, value in zip(ins.args[0].elems, group):
            if not isinstance(dst, Var):
                raise ra.NotSupported("ldmatrix destinations must be registers")
            self.bind(dst.name, value)


    def store_global(self, ins, t):
        if (ins.guard is None and ins.mods == ("global", "v2", "f32")
                and len(ins.args) == 2 and isinstance(ins.args[1], Vec)
                and len(ins.args[1].elems) == 2
                and all(isinstance(x, Var) for x in ins.args[1].elems)):
            vals = tuple(self.reg32(x) for x in ins.args[1].elems)
            quads = [group for op in self.ops for group in op.defs
                     if isinstance(group, tuple) and len(group) == 4]
            if any(vals in (group[:2], group[2:]) for group in quads):
                base, off = self._addr(ins.args[0])
                self.emit(f"STG.E.64 [{{s0}}.64{off}], {{s1}}", [],
                          [base, vals[0], vals[1]], side=True)
                return
        return super().store_global(ins, t)


def _fixed_stalls(code, sched, join_at):
    """Measured SM80 scalar distances, retaining tensor/async dependencies.

    Same fixed-pipe model as the scalar backend, without its corpus-discovery
    imports. Tensor execution/observer latency is handled separately below.
    """
    from openassembler.backend.compile import _opname, _pdefs, _puses, hw_latency
    from openassembler.isa.word import Control
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


def tensor_schedule(body, labels):
    """Source-order scheduler with SB0 reserved for cp.async completion groups.

    ptxas's LDGDEPBAR has W0; DEPBAR.LE SB0 observes this async-copy counter.
    Ordinary memory scoreboards use 1..5, and live IDs retire before recycling.
    """
    from openassembler.backend.compile import (
        MEM_ASYNC, _opname, _pdefs, _puses, _branch_pred_latency)
    from openassembler.isa.word import Control
    pending_w, pending_r = {}, {}
    out = []
    joins = set(labels.values())
    next_sb = 1
    async_group = 0
    committed_group_seen = False
    active = set()

    for i, op in enumerate(body):
        name = _opname(op.text)
        p_read = [("p", p) for p in _puses(op.text)]
        p_write = [("p", p) for p in _pdefs(op.text)]
        wait = 0
        if i in joins or name in ("BRA", "EXIT", "BAR"):
            for b in active:
                wait |= 1 << b
        for reg in op.reads + op.writes + p_read + p_write:
            if reg in pending_w:
                wait |= 1 << pending_w[reg]
        for reg in op.writes:
            if reg in pending_r:
                for b in pending_r[reg]:
                    wait |= 1 << b

        def retire(mask):
            for reg in [r for r,b in pending_w.items() if mask & (1<<b)]:
                pending_w.pop(reg)
            for reg, bars in list(pending_r.items()):
                bars.difference_update(b for b in tuple(bars) if mask & (1<<b))
                if not bars:pending_r.pop(reg)
            active.difference_update(b for b in tuple(active) if mask & (1<<b))

        retire(wait)
        def take():
            nonlocal next_sb, wait
            b = next_sb
            next_sb = 1 + next_sb % 5
            if b in active:
                wait |= 1 << b
                retire(1 << b)
            active.add(b)
            return b

        wbar = rbar = 7
        if op.var_lat:
            wbar = take()
            for reg in op.writes + p_write:
                pending_w[reg] = wbar
        if name in MEM_ASYNC and op.reads and name != "LDSM":
            rbar = take()
            for reg in op.reads:
                pending_r.setdefault(reg,set()).add(rbar)
        elif name == "LDSM":
            # A completed LDSM result implies its address has been consumed.
            # Reuse its write-completion scoreboard to protect address WARs.
            # This omits the hardware read barrier without dropping the
            # dependency, including when several loads share one address reg.
            if wbar == 7:raise AssertionError("LDSM needs a completion scoreboard")
            for reg in op.reads:
                pending_r.setdefault(reg,set()).add(wbar)
        if name == "LDGDEPBAR":
            wbar = 0
            async_group += 1
            committed_group_seen = True
        if name == "DEPBAR":
            if not committed_group_seen:
                raise ra.NotSupported("DEPBAR without a tensor copy-group commit")
            threshold=int(op.text.rsplit(' ',1)[1],16)
            async_group=min(async_group,threshold)
        stall = 2 if op.var_lat else 6
        if name in ("STG", "BRA", "EXIT"):
            stall = 5
        out.append((op.text, Control(stall=stall, yield_=False, wbar=wbar, rbar=rbar, wait=wait, reuse=0)))
    out = _fixed_stalls(body, out, joins)
    # The scalar cost model treats synchronization as an ordinary no-result
    # instruction and otherwise shrinks these to S1. ptxas uses S6 after
    # deferred CTA barriers and S4 after DEPBAR in this exact SM80 family.
    # Preserve those issue gaps while testing them as separate semantics.
    for i, (text, ctl) in enumerate(out):
        minimum = {"BAR": 6, "DEPBAR": 4, "LDSM": 4}.get(_opname(text), 1)
        if ctl.stall < minimum:
            out[i] = (text, dataclasses.replace(ctl, stall=minimum))
    return _branch_pred_latency(out)


def policy(config=None) -> dict:
    """Describe the validated tensor policy without returning mutable state."""
    config = config or TensorConfig()
    if not isinstance(config, TensorConfig):
        raise TypeError("config must be an immutable TensorConfig")
    return {"backend": "sm80-tensor", "schedule": "source-order-hmma8-epoch-drain15",
            "copy_group_scoreboard": 0, "protected_scoreboard_recycling": True,
            "ldsm_address_war_uses_write_completion": True,
            "tensor_reuse_hints": config.reuse, "pointer_mode": config.pointer_mode,
            "pointer_advance": config.pointer_advance}


def generate(kernel, arch, opt_level, *, config=None):
    """Return (scheduled code, labels, words, exits, regcount) for SM80 PTX.

    All public optimization levels use the same tensor-aware dependency
    pipeline. The scalar list scheduler has no validated tensor latency model
    and is never applied to this path.
    """
    from openassembler.backend.compile import Out
    from openassembler.container.cubin import required_regcount
    from openassembler.emu.program import parse_insn

    if arch != "sm_80":
        raise ra.NotSupported("the tensor backend supports only sm_80")
    if opt_level not in ("safe", "mined", "sched", "opt"):
        raise ra.NotSupported(f"unsupported tensor optimization level: {opt_level}")
    if not needs_tensor(kernel):
        raise ra.NotSupported("kernel does not require the tensor backend")
    config = config or TensorConfig()
    if not isinstance(config, TensorConfig):
        raise TypeError("config must be an immutable TensorConfig")
    selector = TensorSelector(kernel, config)
    ops = ra.coalesce(ra.dce(selector.run()))
    phys = ra.allocate(ops, selector.r1)
    lowered = ra.lower(ops, phys)
    drained = []
    for i, item in enumerate(lowered):
        drained.append(item)
        if item[0].startswith("HMMA") and (
                i + 1 == len(lowered) or not lowered[i + 1][0].startswith("HMMA")):
            # Tensor-chain interlocks do not protect ordinary observers. The
            # clocked result-to-store sweeps require an epoch drain before a
            # scalar op, memory access, register reuse or control-flow edge.
            drained.append(("NOP", False, [], []))
    labels, body = {}, []
    for text, variable, reads, writes in drained:
        if text.startswith("<label "):
            labels[text[7:-1]] = len(body)
        else:
            body.append(Out(text, variable, reads, writes))
    code = tensor_schedule(body, labels)
    for i, (text, ctl) in enumerate(code):
        if text.startswith("HMMA"):
            code[i] = (text, dataclasses.replace(ctl, stall=config.issue_gap, yield_=False))
        elif text == "NOP":
            code[i] = (text, dataclasses.replace(ctl, stall=config.epoch_drain, yield_=True))
    if config.reuse:
        for i in range(len(code) - 1):
            current = parse_insn(0, code[i][0], 0)
            following = parse_insn(0, code[i + 1][0], 0)
            if current.op != "HMMA" or following.op != "HMMA":
                continue
            reuse = 0
            written = set(body[i].writes)
            for operand, width in ((1, 4), (2, 2)):
                old, new = current.ops[operand], following.ops[operand]
                if old == new and not written.intersection(range(old.idx, old.idx + width)):
                    reuse |= 1 << (operand - 1)
            text, ctl = code[i]
            code[i] = (text, dataclasses.replace(ctl, reuse=reuse))
    encoder = TensorEncoder()
    words = ra.encode_code(encoder, code, labels)
    words.append(encoder.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
    while len(words) % 8:
        words.append(encoder.encode("NOP"))
    exits = [16 * i for i, (text, _) in enumerate(code) if text.endswith("EXIT")]
    return code, labels, words, exits, required_regcount([text for text, _ in code])

