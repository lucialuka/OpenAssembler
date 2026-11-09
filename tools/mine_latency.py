"""M2: extract ptxas's machine model from its control codes.

Usage: uv run --no-project python tools/mine_latency.py [sm_80] [--corpus DIR ...]

For straight-line code (within a basic block), for every register RAW dependency
producer -> consumer:
  * if the consumer waits on a scoreboard the producer set   -> "variable latency" op
  * otherwise the issue distance (sum of stall counts from producer up to, not
    including, the consumer) is a fixed-latency guarantee ptxas relies on.
The minimum distance seen per producer class is ptxas's latency for that class.

Also reports the scoreboard policy (which SB index each op class uses for writes
and reads) and how often scoreboards are shared.
Writes corpus/<arch>/machine_model.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openassembler.emu.operands import Mem, Pred, Reg  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.isa.word import control  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BRANCHY = {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSYNC", "BSSY", "WARPSYNC", "BAR", "BPT"}

# Number of 32-bit registers written / read per operand, by opcode modifiers.
def width_of(insn, op, is_dst: bool) -> int:
    mods = set(insn.mods)
    if isinstance(op, Mem):
        return 2 if op.wide else 1
    base = insn.op
    if base in ("LDG", "LDS", "LDL", "LD", "LDC", "ULDC", "LDSM") and is_dst:
        for m, w in (("128", 4), ("64", 2)):
            if m in mods:
                return w
        if base == "LDSM":
            return {"1": 1, "2": 2, "4": 4}.get(insn.mods[-1], 1)
        return 1
    if base in ("STG", "STS", "STL", "ST") and not is_dst:
        for m, w in (("128", 4), ("64", 2)):
            if m in mods:
                return w
    if base == "IMAD" and "WIDE" in mods and is_dst:
        return 2
    if base in ("DADD", "DMUL", "DFMA", "DSETP", "F2F", "I2F", "F2I") or base.startswith("D"):
        return 2
    if base in ("HMMA", "IMMA", "DMMA") and is_dst:
        return 4
    return 1


def regs(insn, op, is_dst: bool) -> list[tuple[str, int]]:
    if isinstance(op, Reg):
        if (op.file == "R" and op.idx == 255) or (op.file == "UR" and op.idx == 63):
            return []
        return [(op.file, op.idx + k) for k in range(width_of(insn, op, is_dst))]
    if isinstance(op, Mem) and op.base is not None and op.base.idx != 255:
        n = 2 if op.wide else 1
        out = [(op.base.file, op.base.idx + k) for k in range(n)]
        if op.ur is not None:
            out.append(("UR", op.ur))
        return out
    if isinstance(op, Pred) and op.idx != 7:
        return [(op.file, op.idx)]
    return []


def defs_uses(insn):
    """Destination operands: leading Reg/Pred operands, except for stores/branches/etc."""
    ops = insn.ops
    no_dst = insn.op in ("STG", "STS", "STL", "ST", "RED", "BRA", "EXIT", "BAR", "BSSY", "BSYNC",
                          "CALL", "RET", "NOP", "DEPBAR", "MEMBAR", "LDGSTS", "CCTL", "WARPSYNC")
    dsts, srcs = [], []
    if no_dst:
        srcs = ops
    else:
        k = 0
        while k < len(ops) and isinstance(ops[k], (Reg, Pred)):
            # predicate destinations come right after a register destination (IADD3 R, P, ...)
            k += 1
            if isinstance(ops[k - 1], Reg):
                while k < len(ops) and isinstance(ops[k], Pred) and _pred_dst_slot(insn, k):
                    k += 1
                break
        dsts, srcs = ops[:k], ops[k:]
        if insn.op in ("ISETP", "FSETP", "DSETP", "PLOP3", "UISETP", "VOTE", "FCHK", "R2P"):
            dsts, srcs = ops[:2], ops[2:]
    d = [r for o in dsts for r in regs(insn, o, True)]
    s = [r for o in srcs for r in regs(insn, o, False)]
    if insn.guard is not None and insn.guard.idx != 7:
        s.append((insn.guard.file, insn.guard.idx))
    return d, s


def _pred_dst_slot(insn, k) -> bool:
    return insn.op in ("IADD3", "IMAD", "LEA", "LOP3", "SHFL", "UIADD3", "ULEA")


def op_class(insn) -> str:
    m = list(insn.mods)
    keep = [x for x in m if x in ("WIDE", "HI", "X", "MOV", "IADD", "SHL", "64", "128", "U32", "MMA", "F32",
                                  "RCP", "RSQ", "EX2", "LG2", "SIN", "COS", "SQRT", "TANH", "RCP64H", "RSQ64H")]
    return ".".join([insn.op] + keep)


def blocks(prog):
    """Yield basic blocks as lists of insns (split at labels and after control flow)."""
    targets = set(prog.labels.values())
    cur = []
    for addr in sorted(prog.insns):
        ins = prog.insns[addr]
        if addr in targets and cur:
            yield cur
            cur = []
        cur.append(ins)
        if ins.op in BRANCHY:
            yield cur
            cur = []
    if cur:
        yield cur


def mine(files):
    fixed = defaultdict(Counter)     # producer class -> Counter(distance) for SB-free deps
    pair_min = {}                    # (producer, consumer) -> min distance
    variable = Counter()             # producer class -> # deps covered by a scoreboard
    wsb, rsb = defaultdict(Counter), defaultdict(Counter)
    stall_by_op = defaultdict(Counter)
    n_insn = 0
    for f in files:
        prog = load_listing(Path(f).read_text())
        for blk in blocks(prog):
            last_write = {}  # reg -> (index in blk, insn)
            ctl = [control(i.word) for i in blk]
            for j, ins in enumerate(blk):
                n_insn += 1
                c = ctl[j]
                cls = op_class(ins)
                stall_by_op[cls][c.stall] += 1
                if c.wbar != 7:
                    wsb[cls][c.wbar] += 1
                if c.rbar != 7:
                    rsb[cls][c.rbar] += 1
                d, s = defs_uses(ins)
                seen_prod = set()
                for r in s:
                    if r not in last_write:
                        continue
                    i, prod = last_write[r]
                    if i in seen_prod:
                        continue
                    seen_prod.add(i)
                    pc = ctl[i]
                    pcls = op_class(prod)
                    covered = pc.wbar != 7 and any(ctl[k].wait >> pc.wbar & 1 for k in range(i + 1, j + 1))
                    if covered:
                        variable[pcls] += 1
                        continue
                    dist = sum(ctl[k].stall for k in range(i, j))
                    fixed[pcls][dist] += 1
                    key = (pcls, cls)
                    pair_min[key] = min(pair_min.get(key, 99), dist)
                for r in d:
                    last_write[r] = (j, ins)
    return fixed, pair_min, variable, wsb, rsb, stall_by_op, n_insn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", nargs="?", default="sm_80")
    args = ap.parse_args()
    corpus = ROOT / "corpus" / args.arch
    files = sorted((corpus / "sass").glob("*.sass")) + sorted((corpus / "divconst" / "sass").glob("*.sass"))
    fixed, pair_min, variable, wsb, rsb, stall_by_op, n = mine(files)

    model = {}
    for cls in sorted(set(fixed) | set(variable)):
        dists = fixed.get(cls, Counter())
        model[cls] = {
            "fixed_min": min(dists) if dists else None,
            "fixed_samples": sum(dists.values()),
            "fixed_hist": dict(sorted(dists.items())),
            "scoreboarded": variable.get(cls, 0),
            "wsb": dict(wsb.get(cls, {})), "rsb": dict(rsb.get(cls, {})),
        }
    out = corpus / "machine_model.json"
    out.write_text(json.dumps({"insns": n, "classes": model,
                               "pairs": {f"{a}->{b}": v for (a, b), v in sorted(pair_min.items())}}, indent=1))

    print(f"{args.arch}: {n} instructions in {len(files)} listings, {len(model)} producer classes")
    print(f"{'producer class':24} {'min':>4} {'samples':>8} {'sb-deps':>8}  distance histogram / write-SB")
    for cls, m in sorted(model.items(), key=lambda kv: (-(kv[1]['fixed_samples'] + kv[1]['scoreboarded']))):
        if m["fixed_samples"] + m["scoreboarded"] < 3:
            continue
        h = " ".join(f"{d}:{c}" for d, c in list(m["fixed_hist"].items())[:7])
        print(f"{cls:24} {str(m['fixed_min']):>4} {m['fixed_samples']:>8} {m['scoreboarded']:>8}  {h}  {m['wsb'] or ''}")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
