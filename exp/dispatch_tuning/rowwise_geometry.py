"""Bounded CTA-geometry search for current native rowwise kernels.

Only the number of independent rows assigned to a CTA changes. Every row
still belongs to a complete warp, with exactly the existing reduction tree,
math, neutral tail values and synchronization. This is an experimental source
and launch certificate, not an inferred production rowwise contract.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib

from openassembler.kernels.reductions import ReductionPlan, generate

WARPS = (1, 2, 4, 8)
SHAPES = (
    (1, 1), (1, 31), (1, 32), (1, 33),
    (3, 127), (3, 128), (4, 129), (7, 31), (7, 33), (9, 127),
    (32, 128), (32, 512), (32, 513),
    (129, 127), (129, 129), (129, 511), (129, 513), (129, 1023),
    (1024, 31), (1024, 128), (1024, 512), (1024, 513), (1024, 1024),
    (4096, 1), (4096, 32), (4096, 128), (4096, 512), (4096, 513), (4096, 1024),
)
TOLERANCES = {"softmax": {"rtol": 5e-5, "atol": 1e-6},
              "layernorm": {"rtol": 1e-4, "atol": 2e-5}}


def source(plan: ReductionPlan, warps: int) -> tuple[str, dict]:
    if not isinstance(plan, ReductionPlan) or type(warps) is not int or warps not in WARPS:
        raise ValueError("requires a validated ReductionPlan and 1/2/4/8 whole warps")
    baseline = generate(plan)
    old = "mad.lo.u32 %r4,%r1,4,%r3;"
    if baseline.count(old) != 1:
        raise ValueError("production row-coordinate emitter no longer matches the audited source")
    replacement = f"mad.lo.u32 %r4,%r1,{warps},%r3;"
    text = baseline.replace(old, replacement)
    # Every byte outside the one CTA-row multiplier is preserved. In particular
    # there is no FMA contraction, tree reorder or synchronization removal.
    if text.replace(replacement, old) != baseline:
        raise AssertionError("rowwise arithmetic or synchronization changed")
    certificate = {"schema": 1, "plan": asdict(plan), "warps_per_cta": warps,
                   "launch": {"grid": [(plan.rows + warps - 1) // warps, 1, 1],
                              "threads": 32 * warps, "smem": 0},
                   "manual_launch_experiment": True,
                   "production_launch_contract_inferred": False,
                   "only_cta_row_multiplier_changed": True,
                   "lanes_per_row": 32, "arithmetic_and_sync_order_preserved": True,
                   "input_alignment": 4, "output_alignment": 4,
                   "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "baseline_source_sha256": hashlib.sha256(baseline.encode()).hexdigest()}
    return text, certificate


def ownership(rows: int, warps: int) -> list[int]:
    """Independent scalar ownership model; tails must exit whole warps."""
    if type(rows) is not int or rows < 1 or type(warps) is not int or warps not in WARPS:
        raise ValueError("invalid row-geometry contract")
    result = []
    for block in range((rows + warps - 1) // warps):
        for warp in range(warps):
            row = block * warps + warp
            if row < rows:
                result.append(row)
    return result
