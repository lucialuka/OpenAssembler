from __future__ import annotations

from openassembler.kernels.gemm import GemmPlan, generate

from load_next_compute_current import (
    candidate_compute,
    candidate_source,
    fragment_events,
    schedule_summary,
    validate_events,
)


def test_event_schedule_preserves_k_order_and_alternates_slots():
    events = fragment_events(4)
    validate_events(events)
    loads = [event for event in events if event.kind == "load"]
    assert [(event.k, event.slot) for event in loads] == [(0, 0), (1, 1), (2, 0), (3, 1)]
    assert sum(event.kind == "hmma" for event in events) == 4 * 4 * 8


def test_candidate_compute_has_load_between_current_hmma_halves():
    lines = candidate_compute("f16", "0")
    first_load = next(index for index, line in enumerate(lines) if line.startswith("ldmatrix"))
    next_load = next(index for index, line in enumerate(lines[first_load + 1:], first_load + 1)
                     if line.startswith("ldmatrix") and "%r464" in line)
    first_half_hmma = [index for index, line in enumerate(lines[:next_load]) if line.startswith("mma.")]
    assert first_half_hmma
    assert next_load > first_half_hmma[-1]
    assert sum(line.startswith("mma.") for line in lines) == 4 * 4 * 8


def test_candidate_source_is_experiment_only_and_keeps_launch_contract():
    plan = GemmPlan(128, 128, 128, dtype="f16", schedule="stagepair")
    baseline = generate(plan)
    candidate = candidate_source(plan)
    assert candidate != baseline
    assert candidate.count("mma.sync.aligned.m16n8k16") == baseline.count("mma.sync.aligned.m16n8k16")
    assert candidate.count("ldmatrix.sync.aligned.m8n8.x4.shared.b16") == baseline.count(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
    )
    assert "cp.async.wait_group 0; bar.sync 0;" in candidate
    assert "ret;" in candidate


def test_integrated_exact_shape_schedule_matches_validated_experiment_source():
    for dtype in ("f16", "bf16"):
        plan = GemmPlan(1024, 4096, 1024, dtype=dtype, fragment_schedule="load_next")
        baseline = GemmPlan(1024, 4096, 1024, dtype=dtype)
        assert generate(plan) == candidate_source(baseline)


def test_summary_records_full_output_gate_scope():
    summary = schedule_summary()
    assert summary["k_order_preserved"] is True
    assert summary["register_windows"] == 2
    assert summary["hmma"] == 128
