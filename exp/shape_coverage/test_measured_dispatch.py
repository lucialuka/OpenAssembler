"""Contract and evidence failures that must block experimental dispatch."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from exp.shape_coverage.measured_dispatch import (
    GATES, Requirements, evaluate_candidate, select_experiment,
)


@pytest.fixture
def proposal():
    workload = {
        "role": "ffn_up", "m": 1024, "n": 4096, "k": 1024, "batch_count": 1,
        "dtype": "f16", "arch": "sm_80",
        "operand_layouts": ["contiguous_MK", "contiguous_NK", "contiguous_MN"],
        "alignment_bytes": [16, 16, 8], "tail_policy": "reject",
        "workspace_bytes": 0, "accumulation_dtype": "f32", "output_dtype": "f32",
        "alpha": 1.0, "beta": 0.0, "numerical_policy": "ordered_hmma_fp32",
        "timing_scope": "resident_gemm_no_packing",
    }
    policy = {"tile_mnk": [128, 128, 64], "warps": 4, "stages": 2,
              "cta_order": "m_fast", "fragment_schedule": "load_next",
              "pointer_mode": "legacy", "pointer_advance": "late"}
    source = {"git_head": "test-only-not-measurement",
              "sha256": {"openassembler/kernels/gemm.py": "a" * 64,
                         "exp/shape_coverage/a100_role_gemm_modal.py": "b" * 64}}
    entry = {"schema": 1, "id": "synthetic_gate_fixture_not_hardware_evidence",
             "workload": workload, "policy": policy, "source_identity": source,
             "runs": []}
    for split in ("selection", "heldout"):
        entry["runs"].append({
            "id": f"fixture-{split}", "session_id": f"session-{split}",
            "split": split, "frozen_before_measurement": True,
            "workload": deepcopy(workload), "policy": deepcopy(policy),
            "source_identity": deepcopy(source),
            "gates": dict.fromkeys(GATES, True),
            "input_families": ["zero", "integer", "normal"],
            "artifact": {"cubin_sha256": "c" * 64, "sass_sha256": "d" * 64,
                         "raw_result_path": f"fixture-{split}.json",
                         "raw_result_sha256": "e" * 64, "entry": "gemm"},
            "resources": {"registers": 160, "shared_bytes": 65536,
                          "local_bytes": 0, "spill_bytes": 0},
            "comparators": {
                "same_source_ptxas": {"name": "ptxas 12.8", "matched_contract": True,
                                      "full_output_passed": True},
                "library": {"name": "cuBLAS GemmEx", "matched_contract": True,
                            "full_output_passed": True},
            },
            "timing": {"scope": workload["timing_scope"], "paired": True,
                       "randomized_order": True,
                       "rounds": [{"samples": 8, "oa_baseline_ms": 0.09,
                                   "oa_candidate_ms": 0.08, "ptxas_ms": 0.06,
                                   "library_ms": 0.05} for _ in range(2)]},
        })
    return {"workload": deepcopy(workload), "current_source": deepcopy(source),
            "entries": [entry]}


def test_complete_fixture_selects_only_an_experiment_and_exposes_control_losses(proposal):
    result = select_experiment(**proposal)
    assert result["status"] == "selected"
    assert result["production_promoted"] is False
    decision = result["decisions"][0]
    assert not decision["reasons"]
    assert all(row["oa_over_ptxas"] > 1 and row["oa_over_library"] > 1
               for row in decision["ratios"])


@pytest.mark.parametrize("field,value", [
    ("role", "qkv_projection"), ("m", 2048), ("dtype", "bf16"),
    ("operand_layouts", ["transposed_MK", "contiguous_NK", "contiguous_MN"]),
    ("alignment_bytes", [8, 16, 8]), ("workspace_bytes", 1024),
    ("tail_policy", "padding"), ("beta", 1.0),
    ("timing_scope", "gemm_including_packing"),
])
def test_same_dimensions_or_role_name_do_not_authorize_a_different_contract(proposal, field, value):
    proposal["workload"][field] = value
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert "exact workload contract mismatch" in result["decisions"][0]["reasons"]


def test_changed_harness_source_invalidates_previously_passing_results(proposal):
    proposal["current_source"]["sha256"]["exp/shape_coverage/a100_role_gemm_modal.py"] = "f" * 64
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert "stale or missing source identity" in result["decisions"][0]["reasons"]


@pytest.mark.parametrize("gate", GATES)
def test_each_individual_integrity_gate_blocks_selection(proposal, gate):
    del proposal["entries"][0]["runs"][1]["gates"][gate]
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert any(gate in reason for reason in result["decisions"][0]["reasons"])


def test_policy_name_does_not_replace_encoded_stream_verification(proposal):
    entry = proposal["entries"][0]
    entry["policy"]["pointer_mode"] = "direct"
    # The claimed direct policy still points at the old legacy measurement.
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert any("policy mismatch" in reason for reason in result["decisions"][0]["reasons"])


@pytest.mark.parametrize("comparator", ["library", "same_source_ptxas"])
def test_missing_or_unmatched_comparator_blocks_selection(proposal, comparator):
    proposal["entries"][0]["runs"][0]["comparators"][comparator]["matched_contract"] = False
    assert select_experiment(**proposal)["status"] == "fallback"


def test_two_rounds_from_one_process_are_not_independent_heldout_evidence(proposal):
    entry = proposal["entries"][0]
    entry["runs"][1]["session_id"] = entry["runs"][0]["session_id"]
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert "selection and held-out evidence reuse a session" in result["decisions"][0]["reasons"]


def test_posthoc_heldout_label_and_one_round_regression_block_selection(proposal):
    run = proposal["entries"][0]["runs"][1]
    run["frozen_before_measurement"] = False
    run["timing"]["rounds"][1]["oa_candidate_ms"] = 0.091
    result = select_experiment(**proposal)
    assert result["status"] == "fallback"
    assert any("not frozen" in reason for reason in result["decisions"][0]["reasons"])
    assert any("minimum OA baseline speedup" in reason
               for reason in result["decisions"][0]["reasons"])


@pytest.mark.parametrize("field,value", [("spill_bytes", 4), ("local_bytes", 16),
                                       ("registers", 256), ("shared_bytes", 163841)])
def test_resource_limit_requires_evidence_and_rejects_spills(proposal, field, value):
    proposal["entries"][0]["runs"][0]["resources"][field] = value
    assert select_experiment(**proposal)["status"] == "fallback"


@pytest.mark.parametrize("field,value", [("oa_candidate_ms", float("nan")),
                                       ("library_ms", 0.0), ("samples", 1)])
def test_invalid_or_under_sampled_timing_cannot_win(proposal, field, value):
    proposal["entries"][0]["runs"][0]["timing"]["rounds"][0][field] = value
    assert select_experiment(**proposal)["status"] == "fallback"


def test_old_raw_role_results_fail_closed_without_manufacturing_evidence():
    root = Path(__file__).parent
    raw = json.loads((root / "a100-role-gemm_20261109T083359.463266Z.json").read_text())
    result = evaluate_candidate(raw, workload={}, current_source=raw["source_identity"])
    assert result["status"] == "rejected"
    assert "no evidence runs" in result["reasons"]
    assert "current source identity lacks compiler or harness hashes" in result["reasons"]


def test_missing_and_malformed_nested_evidence_is_rejected_without_selecting(proposal):
    entry = proposal["entries"][0]
    entry["runs"][0].update({"split": [], "gates": None, "timing": [],
                              "resources": None, "artifact": None,
                              "comparators": None, "input_families": None})
    assert select_experiment(**proposal)["status"] == "fallback"


def test_requiring_no_win_or_no_rounds_is_not_a_valid_acceptance_policy():
    with pytest.raises(ValueError):
        Requirements(min_speedup=1)
    with pytest.raises(ValueError):
        Requirements(min_rounds=0)
