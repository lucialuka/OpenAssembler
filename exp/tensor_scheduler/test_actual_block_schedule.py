from __future__ import annotations

from openassembler.kernels.gemm import GemmPlan

from actual_block_schedule import analyze


def test_actual_oa_lowered_blocks_are_consumed_by_experiment_scheduler():
    result = analyze(GemmPlan(128, 128, 128, dtype="f16", schedule="stagepair"))
    assert result["summary"]["blocks"] >= 1
    assert result["summary"]["hmma_blocks"] >= 1
    assert result["summary"]["production_promoted"] is False
    # The fail-closed adapter intentionally rejects the final terminator block;
    # every block containing real HMMA work must still be schedulable.
    assert all(row["accepted"] and row["k_order_preserved"]
               for row in result["blocks"] if row["hmma"])
    assert any(not row["accepted"] for row in result["blocks"])

