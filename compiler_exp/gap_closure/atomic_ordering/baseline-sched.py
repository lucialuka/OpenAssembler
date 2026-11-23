"""Basic-block list scheduler: reorders lowered instructions before control bits are set.

Input/output: [(text, var_lat, reads, writes)] with physical registers and "<label X>"
markers (openassembler.backend.ra.lower). Within each block (labels and branches end blocks)
instructions are reordered by critical path, using the A100-measured latencies
(compile.hw_latency, corpus/sm_80/hw_latency.json):
  * register RAW / WAR / WAW, predicate def/use (a @P guard needs 13 cycles),
  * memory ops and barriers keep their relative order (no alias analysis),
  * ULDC.64 UR4 (the global-memory descriptor) precedes every memory op,
  * the block's final BRA/EXIT/RET stays last.
"""

from __future__ import annotations

from openassembler.backend.compile import PRED_TO_GUARD, _opname, _pdefs, _puses, hw_latency

ORDERED = ("LDG", "STG", "LDS", "STS", "LDL", "STL", "ATOMG", "ATOMS", "RED", "BAR", "LDGSTS", "LDSM",
           "MEMBAR", "DEPBAR", "S2R", "CS2R", "WARPSYNC", "SHFL")
VAR_LAT = {"LDG": 30, "LDS": 24, "LDL": 30, "S2R": 20, "LDC": 12, "MUFU": 14, "I2F": 12, "F2I": 12}
TERMINATORS = ("BRA", "EXIT", "RET", "BRX", "JMP", "CALL")


def _lat(prod: str, cons: str, var_lat: bool) -> int:
    if var_lat:
        return VAR_LAT.get(prod, 20)
    return hw_latency(prod, cons)


def _blocks(body):
    cur = []
    for item in body:
        if item[0].startswith("<label "):
            if cur:
                yield cur
            yield [item]
            cur = []
            continue
        cur.append(item)
        if _opname(item[0]) in TERMINATORS:
            yield cur
            cur = []
    if cur:
        yield cur


def _schedule_block(blk):
    if len(blk) < 3 or blk[0][0].startswith("<label "):
        return blk
    n = len(blk)
    term = n - 1 if _opname(blk[-1][0]) in TERMINATORS else None
    ops = [_opname(t) for t, *_ in blk]
    succ = [dict() for _ in range(n)]   # i -> {j: latency}

    def edge(i, j, lat):
        succ[i][j] = max(succ[i].get(j, 0), lat)

    last_w, readers, last_pw, preaders = {}, {}, {}, {}
    last_mem, uldc = None, None
    for j, (t, vl, rd, wr) in enumerate(blk):
        for r in rd:
            if r in last_w:
                i = last_w[r]
                edge(i, j, _lat(ops[i], ops[j], blk[i][1]))
        for r in wr:
            if r in last_w:
                edge(last_w[r], j, 1)
            for i in readers.get(r, ()):
                edge(i, j, 1 if ops[i] not in ORDERED else 2)
        pu, pd = _puses(t), _pdefs(t)
        guard = t.startswith("@")
        for p in pu:
            if p in last_pw:
                i = last_pw[p]
                edge(i, j, PRED_TO_GUARD if guard or ops[j] == "BRA" else 4)
        for p in pd:
            if p in last_pw:
                edge(last_pw[p], j, 1)
            for i in preaders.get(p, ()):
                edge(i, j, 1)
        if ops[j] in ORDERED or ops[j] in TERMINATORS:
            if last_mem is not None:
                edge(last_mem, j, 1)
            last_mem = j
            if uldc is not None:
                edge(uldc, j, 1)
        if t.startswith("ULDC"):
            uldc = j
        if term is not None and j != term:
            edge(j, term, 1)
        for r in rd:
            readers.setdefault(r, []).append(j)
        for r in wr:
            last_w[r] = j
            readers[r] = []
        for p in pu:
            preaders.setdefault(p, []).append(j)
        for p in pd:
            last_pw[p] = j
            preaders[p] = []
    # critical-path priority (longest latency path to the block end)
    prio = [0] * n
    for i in range(n - 1, -1, -1):
        prio[i] = max([lat + prio[j] for j, lat in succ[i].items()], default=0)
    npred = [0] * n
    for i in range(n):
        for j in succ[i]:
            npred[j] += 1
    ready_at = [0] * n
    ready = [i for i in range(n) if npred[i] == 0]
    order, cycle = [], 0
    while ready:
        # prefer ops whose inputs are available now; then longest path; then source order
        ready.sort(key=lambda i: (ready_at[i] > cycle, -prio[i], i))
        i = ready.pop(0)
        cycle = max(cycle, ready_at[i]) + 1
        order.append(i)
        for j, lat in succ[i].items():
            ready_at[j] = max(ready_at[j], cycle - 1 + lat)
            npred[j] -= 1
            if npred[j] == 0:
                ready.append(j)
    assert len(order) == n, "dependence cycle"
    return [blk[i] for i in order]


def list_schedule(body):
    out = []
    for blk in _blocks(body):
        out.extend(_schedule_block(blk))
    return out
