"""Semantic failures that block the one exact tile64 experiment proposal."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest

from exp.shape_coverage.import_tile64_evidence import (
    EvidenceError, HELDOUT, POLICY, PROTOCOL, SELECTION, WORKLOAD, build_proposal,
)
from exp.shape_coverage.measured_dispatch import POLICY_FIELDS, WORKLOAD_FIELDS, select_experiment


@pytest.fixture
def inputs(tmp_path):
    paths = []
    for source in (SELECTION, HELDOUT, PROTOCOL):
        target = tmp_path / source.name
        shutil.copy2(source, target)
        paths.append(target)
    return paths


def _mutate(path: Path, change):
    stat = path.stat()
    value = json.loads(path.read_bytes())
    change(value)
    path.write_text(json.dumps(value))
    # Preserve metadata so these tests isolate their semantic condition.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_actual_raw_import_selects_a_bounded_experiment_and_reports_remaining_losses():
    proposal = build_proposal()
    assert set(WORKLOAD) == WORKLOAD_FIELDS
    assert set(POLICY) == POLICY_FIELDS
    assert len(WORKLOAD) + len(POLICY) == 24
    result = select_experiment(proposal["entries"], workload=proposal["workload"],
                               current_source=proposal["current_source"])
    assert result["status"] == "selected"
    assert result["production_promoted"] is False
    ratios = result["decisions"][0]["ratios"]
    assert len(ratios) == 6
    assert all(1.93 <= row["oa_baseline_speedup"] <= 1.96 for row in ratios)
    assert all(row["oa_over_ptxas"] > 1.28 and row["oa_over_library"] > 1.42 for row in ratios)
    selection, repeat = proposal["entries"][0]["runs"]
    assert "ptxas_tile64 /" in repeat["comparators"]["same_source_ptxas"]["name"]
    assert selection["session_id"] != repeat["session_id"]
    assert repeat["provenance"]["container_id"] is None
    assert repeat["provenance"]["independent_shape"] is False
    assert "not embedded" in repeat["provenance"]["modal_id_source"]
    assert all(value["best_pointer_over_tile64_speedup"] > 1.87
               for value in repeat["additional_controls"])
    assert repeat["resources"] == {"registers": 88, "shared_bytes": 32768,
                                    "local_bytes": 0, "spill_bytes": 0}
    assert repeat["resource_proof"]["ldl_count"] == repeat["resource_proof"]["stl_count"] == 0
    assert "no ptxas spill report is claimed" in repeat["resource_proof"]["spill_derivation"]


def test_failed_full_output_gate_cannot_be_normalized_to_true(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["checks"][0].update(passed=False))
    with pytest.raises(EvidenceError, match="numerical/canary/input gate failed"):
        build_proposal(*inputs)


def test_missing_input_family_cannot_be_normalized_to_full_coverage(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["checks"].pop())
    with pytest.raises(EvidenceError, match="coverage is incomplete"):
        build_proposal(*inputs)


def test_cubin_hash_must_match_compile_manifest_and_static_record(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["static"]["oa_tile64"].update(cubin_sha256="0" * 64))
    with pytest.raises(EvidenceError, match="cubin hash differs"):
        build_proposal(*inputs)


def test_frozen_protocol_cannot_admit_another_shape(inputs):
    _mutate(inputs[2], lambda d: d["workload"].update(n=1024))
    with pytest.raises(EvidenceError, match="protocol exact workload mismatch"):
        build_proposal(*inputs)


def test_freeze_after_repeat_cannot_be_claimed_as_predeclared(inputs):
    _mutate(inputs[2], lambda d: d.update(frozen_utc="2026-11-09T12:00:00+00:00"))
    with pytest.raises(EvidenceError, match="UTC order is invalid"):
        build_proposal(*inputs)


def test_checkout_or_copy_reset_mtimes_are_diagnostic_only(inputs):
    equal_mtime = 1_800_000_000_000_000_000
    for path in inputs:
        os.utime(path, ns=(equal_mtime, equal_mtime))
    proposal = build_proposal(*inputs)
    protocol = proposal["entries"][0]["frozen_protocol"]
    assert protocol["mtime_order_supports_history"] is False
    assert "not durable provenance" in protocol["mtime_scope"]
    assert "not cryptographic attestation" in protocol["timestamp_scope"]
    result = select_experiment(proposal["entries"], workload=proposal["workload"],
                               current_source=proposal["current_source"])
    assert result["status"] == "selected"


def test_driver_and_cubin_register_count_must_agree(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["resources"]["oa_tile64"].update(num_regs=89))
    with pytest.raises(EvidenceError, match="register count differs"):
        build_proposal(*inputs)


def test_encoded_drain_control_is_verified_instead_of_assumed(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["static"]["oa_tile64"]["post_hmma_nop_drains"][0].update(stall=14))
    with pytest.raises(EvidenceError, match="encoded HMMA8/drain15"):
        build_proposal(*inputs)


def test_spill_zero_is_rejected_when_current_native_stream_has_local_memory(monkeypatch):
    import openassembler.backend.tensor as tensor
    original = tensor.generate

    def with_local_memory(*args, **kwargs):
        code, labels, words, exits, registers = original(*args, **kwargs)
        return code + [("LDL R0, [R1]", code[0][1])], labels, words, exits, registers

    monkeypatch.setattr(tensor, "generate", with_local_memory)
    with pytest.raises(EvidenceError, match="spill proof failed"):
        build_proposal()


def test_selector_still_rejects_tampered_normalized_spill_resource():
    proposal = deepcopy(build_proposal())
    proposal["entries"][0]["runs"][1]["resources"]["spill_bytes"] = 4
    result = select_experiment(proposal["entries"], workload=proposal["workload"],
                               current_source=proposal["current_source"])
    assert result["status"] == "fallback"
    assert any("spill_bytes" in reason for reason in result["decisions"][0]["reasons"])


def test_nonfinite_recorded_output_error_blocks_import(inputs):
    _mutate(inputs[1], lambda d: d["cases"][0]["checks"][0].update(relative_l2_fp32=float("inf")))
    with pytest.raises(EvidenceError, match="nonfinite/out-of-budget"):
        build_proposal(*inputs)
