from __future__ import annotations

from openassembler.kernels.gemm import GemmPlan

from actual_block_schedule import analyze


def test_actual_oa_lowered_blocks_are_consumed_by_experiment_scheduler():
    result = analyze(GemmPlan(128, 128, 128, dtype="f16", schedule="stagepair"))
    assert result["summary"]["blocks"] >= 1
    assert result["summary"]["hmma_blocks"] >= 1
    assert result["summary"]["production_promoted"] is False
    # These original regions include synchronization/visibility and implicit
    # state that the old adapter silently dropped. They must not be accepted
    # merely because a permutation preserves the HMMA accumulator subsequence.
    assert all(not row["accepted"] and row["rejection_reason"]
               for row in result["blocks"] if row["hmma"])
    assert result["encoding_allowed"] is False
    assert result["latency_profile"]["calibrated_fields"] == ()


def test_clocked_profile_does_not_turn_actual_block_model_into_hardware_validation():
    result = analyze(GemmPlan(128, 128, 128), profile="a100_clocked_24_8")
    profile = result["latency_profile"]
    assert profile["latency"]["accum_raw"] == 24
    assert profile["latency"]["hmma_issue"] == 8
    assert profile["hardware_validated_schedule"] is False
    assert profile["cycle_predictions_are_hardware_measurements"] is False
    assert result["encoding_allowed"] is False

