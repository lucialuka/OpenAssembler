from __future__ import annotations

from collections import Counter

import pytest

from openassembler.kernels.reductions import ReductionPlan, generate
from openassembler.ptx.parse import parse_module
from openassembler.build import generate as compile_native
from exp.dispatch_tuning.rowwise_geometry import SHAPES, WARPS, ownership, source


@pytest.mark.parametrize("op", ("softmax", "layernorm"))
@pytest.mark.parametrize("warps", WARPS)
def test_only_whole_warp_cta_row_mapping_changes(op, warps):
    plan = ReductionPlan(129, 513, op)
    text, cert = source(plan, warps)
    assert text.replace(f"mad.lo.u32 %r4,%r1,{warps},%r3;",
                        "mad.lo.u32 %r4,%r1,4,%r3;") == generate(plan)
    assert cert["launch"] == {"grid": [(129 + warps - 1) // warps, 1, 1],
                               "threads": 32 * warps, "smem": 0}
    assert cert["arithmetic_and_sync_order_preserved"] is True
    assert cert["manual_launch_experiment"] is True
    baseline = parse_module(generate(plan)).kernels[op]
    candidate = parse_module(text).kernels[op]
    assert len(baseline.body) == len(candidate.body)
    assert Counter(ins.op for ins in baseline.body) == Counter(ins.op for ins in candidate.body)
    assert text.count("shfl.sync.bfly.b32") == (10 if op == "softmax" else 10)


@pytest.mark.parametrize("warps", WARPS)
def test_cta_mapping_covers_each_row_once_and_has_only_uniform_tail_exits(warps):
    for rows, width in SHAPES:
        assert ownership(rows, warps) == list(range(rows))
        for block in range((rows + warps - 1) // warps):
            for warp in range(warps):
                rows_by_lane = [block * warps + (32 * warp + lane) // 32 for lane in range(32)]
                assert len(set(rows_by_lane)) == 1
                assert len({row >= rows for row in rows_by_lane}) == 1


@pytest.mark.parametrize("op", ("softmax", "layernorm"))
def test_all_candidate_geometry_sources_compile_in_current_scalar_encoder(op):
    for warps in WARPS:
        text, cert = source(ReductionPlan(7, 33, op), warps)
        code, labels, words, exits, registers = compile_native(
            parse_module(text).kernels[op], "sm_80", "opt")
        assert words and exits and 0 < registers <= 255
        assert any("WARPSYNC" in instruction for instruction, control in code)
        assert not any(instruction.startswith(("LDL", "STL", "CALL")) for instruction, control in code)


@pytest.mark.parametrize("warps", (0, 3, 16, True, 4.0, "4"))
def test_invalid_cta_geometry_fails_before_source_emission(warps):
    with pytest.raises(ValueError):
        source(ReductionPlan(7, 33, "softmax"), warps)
