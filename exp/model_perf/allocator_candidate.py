"""Test spill-free pressure scheduling before searching spill slots."""
from openassembler.backend import pressure, ra, spill


baseline = spill.allocate_kernel


def try_without_spills(selector, ops, limit):
    try:
        phys = ra.allocate(ops, selector.r1, max_reg=min(250, limit - 2))
        ra.allocate_preds(ops)
    except ra.NotSupported as error:
        if str(error) not in ('out of registers', 'out of predicate registers'):
            raise
        return None
    return ops, phys, dict(local_bytes=selector.local_frame.bytes, spill_slots=0, register_limit=limit)


def allocate_kernel(selector, ops):
    limit = selector.k.directives.get('maxnreg', (252,))[0]
    if limit < 16:
        return baseline(selector, ops)
    unchanged = try_without_spills(selector, ops, limit)
    if unchanged is not None:
        return unchanged
    ordered = pressure.schedule(ops)
    if any(a is not b for a, b in zip(ops, ordered)):
        try:
            result = try_without_spills(selector, ordered, limit)
        except ra.NotSupported:
            pass
        else:
            if result is not None:
                return result
    return baseline(selector, ops)


def original_intervals(ops):
    """Frozen pre-optimization scan for the experiment and production replay."""
    start, end = {}, {}
    lv = ra.cfg_liveness(ops) if any(op.target for op in ops) else None
    for i, op in enumerate(ops):
        here = list(ra._flat(op.defs)) + [u for u in ra._flat(op.uses) if isinstance(u, ra.V)]
        if lv is not None:
            here += [v for v in lv[i] if isinstance(v, ra.V)]
        for v in here:
            if isinstance(v, ra.V):
                start.setdefault(v, i)
                start[v] = min(start[v], i)
                end[v] = max(end.get(v, i), i)
    return start, end


def intervals(ops):
    start, end = {}, {}
    live_out = ra.cfg_liveness(ops) if any(op.target for op in ops) else None
    for i, op in enumerate(ops):
        for operands in (op.defs, op.uses):
            for value in ra._flat(operands):
                if isinstance(value, ra.V):
                    start.setdefault(value, i)
                    end[value] = i
        if live_out is not None:
            for value in live_out[i]:
                if isinstance(value, ra.V):
                    start.setdefault(value, i)
                    end[value] = i
    return start, end
