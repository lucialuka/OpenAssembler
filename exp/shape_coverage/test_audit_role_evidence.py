from __future__ import annotations

import json
from pathlib import Path

from exp.shape_coverage.audit_role_evidence import (
    CORE_EXPERIMENT_SOURCES, audit_file, build_ledger,
)


def _fixture(schema=3):
    methods = ("oa_pointer_direct", "ptxas_pointer_direct", "cublas")
    checks = [{"method": method, "input": family, "passed": True,
               "canaries": True, "inputs_unchanged": True,
               "ptxas_bitwise": True if method != "cublas" else None,
               "baseline_cross_policy_bitwise": True if method != "cublas" else None}
              for method in methods for family in ("zero", "integer", "normal")]
    statics = {method: {"registers": 128, "cubin_sha256": "a" * 64,
                        "sass_sha256": "b" * 64, "hmma_stall_controls": [8],
                        "post_hmma_nop_drains": [{"stall": 15, "yield": True}]}
               for method in methods if method != "cublas"}
    return {
        "schema": schema, "device": "fixture-not-hardware", "runtime": {},
        "source_identity": {"sha256": {"openassembler/kernels/gemm.py": "c" * 64,
                                         **{name: "d" * 64 for name in CORE_EXPERIMENT_SOURCES}},
                            "harness_sha256": "d" * 64},
        "source_unchanged_during_run": True, "methodology": {},
        "cases": [{"case": {"role": "ffn_up", "m": 1024, "n": 4096,
                             "k": 1024, "dtype": "f16"},
                   "checks": checks,
                   "poisoned_graph_checks": [{"method": method, "passed": True,
                                               "poisoned_output": True} for method in methods],
                   "resources": {method: {"num_regs": 128, "local_bytes": 0}
                                 for method in methods if method != "cublas"},
                   "static": statics,
                   "verified_compile_contracts": {
                       "pointer_direct": {"pointer_mode": "direct", "output_dtype": "f32",
                                          "accumulation": "f32", "input_layout": "contiguous RCR",
                                          "cubin_sha256": {"oa": "a" * 64, "ptxas": "a" * 64}}},
                   "timings": [{"median_ms": dict.fromkeys(methods, 0.1),
                                "paired_orders": [list(methods)]}],
                   "counters": {"status": "permission_denied", "profiles": []}}],
    }


def _write(tmp_path, name, raw):
    path = tmp_path / name
    path.write_text(json.dumps(raw))
    return path


def test_invalid_legacy_pointer_labels_cannot_be_rescued_by_passing_numerics(tmp_path):
    raw = _fixture(1)
    raw.pop("schema")
    raw["cases"][0].pop("verified_compile_contracts")
    for stamp in ("082832.131819", "083147.885491", "083359.463266"):
        path = _write(tmp_path, f"a100-role-gemm_20261109T{stamp}Z.json", raw)
        before = path.read_bytes()
        result = audit_file(path)
        assert path.read_bytes() == before
        assert result["all_reported_checks_passed"] is True
        case = result["cases"][0]
        assert case["invalid_method_labels"] == ["oa_pointer_direct", "ptxas_pointer_direct"]
        assert all(policy["label_status"] == "INVALID-direct-pointer-label"
                   for policy in case["policy_evidence"])
        assert result["production_promoted"] is False
        assert any("Known invalid" in reason for reason in result["audit_blockers"])


def test_schema2_records_pointer_metadata_but_not_encoded_or_full_source_scope(tmp_path):
    raw = _fixture(2)
    raw["source_identity"]["sha256"] = {"openassembler/kernels/gemm.py": "c" * 64}
    for static in raw["cases"][0]["static"].values():
        static.pop("hmma_stall_controls")
        static.pop("post_hmma_nop_drains")
        static.pop("cubin_sha256")
    result = audit_file(_write(tmp_path, "a100-role-gemm_schema2.json", raw))
    assert result["all_reported_checks_passed"]
    assert result["cases"][0]["policy_evidence"][0]["requested_pointer_mode_matches_record"] is True
    assert result["cases"][0]["policy_evidence"][0]["encoded_control_recorded"] is False
    assert not result["source_scope"]["unchanged_flag_includes_known_harness_dependencies"]
    assert any("Schema 2" in reason for reason in result["audit_blockers"])


def test_schema3_counts_and_control_hashes_are_inventory_not_promotion(tmp_path):
    raw = _fixture()
    result = audit_file(_write(tmp_path, "a100-role-gemm_current.json", raw))
    case = result["cases"][0]
    assert result["reported_check_count"] == 9
    assert case["check_counts"]["native_cross_policy_bitwise_all_passed"] is True
    assert case["graph_counts"]["passed"] == 3
    assert case["policy_evidence"][0]["native_hmma8_drain15_record_consistent"] is True
    assert case["policy_evidence"][0]["cubin_hash_matches_compile_record"] is True
    assert case["counter_status"] == "permission_denied"
    assert case["matrix_shape_key"]["A_shape_from_case"] == [1024, 1024]
    assert case["matrix_shape_key"]["C_shape_from_case"] == [1024, 4096]
    assert result["source_scope"]["unchanged_flag_includes_known_harness_dependencies"] is True
    assert result["production_promoted"] is False


def test_changed_sources_block_reuse_without_relabeling_numerical_passes_as_failures(tmp_path):
    raw = _fixture()
    raw["source_unchanged_during_run"] = False
    result = audit_file(_write(tmp_path, "a100-role-gemm_changed.json", raw))
    assert result["all_reported_checks_passed"] is True
    assert any("changed during run" in reason for reason in result["audit_blockers"])
    assert not result["source_scope"]["unchanged_flag_includes_known_harness_dependencies"]


def test_dynamic_scan_adds_new_files_and_records_malformed_raw_json(tmp_path):
    first = _write(tmp_path, "a100-role-gemm_first.json", _fixture())
    assert build_ledger(tmp_path)["raw_run_count"] == 1
    second = tmp_path / "a100-role-gemm_second.json"
    second.write_text("{bad json")
    (tmp_path / "evidence-ledger.json").write_text("{}")
    ledger = build_ledger(tmp_path)
    assert ledger["raw_run_count"] == 2
    assert len(ledger["expected_raw_paths_from_dynamic_scan"]) == 2
    assert ledger["runs"][1]["readable"] is False
    assert ledger["runs"][0]["raw_sha256"] != ledger["runs"][1]["raw_sha256"]
    assert first.exists() and ledger["raw_evidence_modified"] is False
