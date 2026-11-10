"""Offline promotion gates over retained real cubins, never synthetic GPU wins."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from openassembler import gemm_promotion as promotion
from openassembler import gemm_tune


@pytest.fixture
def retained_sessions(tmp_path, monkeypatch):
    """Real binary evidence with explicitly simulated independent provenance.

    Test timestamps/session labels exercise audit logic; they are never written
    to the research evidence or claimed as new hardware measurements.
    """
    root = Path(__file__).resolve().parent.parent
    original = root / "exp/instruction_overhead/a100-production-autotune_20261110.json"
    record = json.loads(original.read_text())
    identity = record["compiler_identity"]
    monkeypatch.setattr(promotion, "compiler_identity", lambda: identity)
    monkeypatch.setattr(gemm_tune, "compiler_identity", lambda: identity)
    paths = []
    for split, completed in (("selection", "2026-11-10T00:00:00+00:00"),
                             ("heldout", "2026-11-10T02:00:00+00:00")):
        directory = tmp_path / split
        directory.mkdir()
        archive_name = record["artifact_archive"]["path"]
        (directory / archive_name).write_bytes((original.parent / archive_name).read_bytes())
        item = deepcopy(record)
        item["completed_utc"] = completed
        item["session"] = {"id": f"simulated-{split}", "split": split,
                           "modal_app_id": f"ap-simulated-{split}", "container_id": None}
        path = directory / "result.json"
        path.write_text(json.dumps(item, indent=2))
        paths.append(path)
    protocol_path = tmp_path / "frozen-protocol.json"
    promotion.freeze_selection(paths[0], protocol_path,
                               now=datetime(2026, 11, 10, 1, tzinfo=timezone.utc))
    return paths[0], paths[1], protocol_path


def _rewrite(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, indent=2, allow_nan=True))


def test_complete_source_binary_and_two_session_audit_promotes_only_frozen_choice(retained_sessions):
    result = promotion.promote(*retained_sessions)
    assert result["status"] == "promoted"
    assert result["policy"] == "tile64_direct"
    assert result["entry"]["gates_passed"] is True
    assert result["entry"]["contract"]["m"] == 128
    assert len(result["rounds"]) == 6
    assert all(row["baseline_over_selected"] > 2 for row in result["rounds"])
    assert {value["split"] for value in result["entry"]["evidence"]} == {"selection", "heldout"}
    assert all(value["native_reencoding"]["registers"] == 88
               for value in result["entry"]["evidence"])
    assert "no absolute-time pooling" in result["entry"]["selection_scope"]


def test_protocol_cannot_be_refrozen_after_reading_heldout(retained_sessions):
    _, _, protocol = retained_sessions
    _rewrite(protocol, lambda value: value.update(frozen_utc="2026-11-10T03:00:00+00:00"))
    with pytest.raises(promotion.PromotionError, match="timestamp order"):
        promotion.promote(*retained_sessions)


def test_selected_source_record_cannot_change_after_freeze(retained_sessions):
    selection, _, _ = retained_sessions
    _rewrite(selection, lambda value: value.update(unrelated_metadata="changed after freeze"))
    with pytest.raises(promotion.PromotionError, match="changed after the freeze"):
        promotion.promote(*retained_sessions)


@pytest.mark.parametrize("field,value,match", [
    ("id", "simulated-selection", "independent session"),
    ("modal_app_id", "ap-simulated-selection", "reuses the selection Modal app"),
    ("split", "selection", "session id/split"),
])
def test_another_filename_does_not_establish_independent_hardware_session(retained_sessions, field, value, match):
    _, heldout, _ = retained_sessions
    _rewrite(heldout, lambda item: item["session"].update({field: value}))
    with pytest.raises(promotion.PromotionError, match=match):
        promotion.promote(*retained_sessions)


def test_heldout_winner_cannot_replace_predeclared_selection(retained_sessions):
    _, heldout, protocol = retained_sessions
    _rewrite(heldout, lambda value: [row["samples_ms"].update(
        tile64=[sample * 0.5 for sample in row["samples_ms"]["tile64"]]) for row in value["timings"]])
    result = promotion.promote(*retained_sessions)
    assert result["policy"] == json.loads(protocol.read_text())["selected_policy"] == "tile64_direct"
    assert result["policy"] != gemm_tune.choose_policy(json.loads(heldout.read_text()))


def test_frozen_winner_must_repeat_baseline_win_in_every_heldout_round(retained_sessions):
    _, heldout, _ = retained_sessions
    _rewrite(heldout, lambda value: value["timings"][1]["samples_ms"].update(
        tile64_direct=value["timings"][1]["samples_ms"]["baseline"]))
    result = promotion.promote(*retained_sessions)
    assert result["status"] == "not-promoted"
    assert result["entry"] is None
    assert any("minimum baseline speedup" in reason for reason in result["reasons"])


def test_near_tie_retains_existing_incumbent_instead_of_chasing_noise(retained_sessions):
    selection, heldout, protocol = retained_sessions
    # Tile64/direct won the selection round, but its held-out gain against an
    # existing tile64 incumbent is below the admitted 2% threshold.
    _rewrite(protocol, lambda value: value.update(incumbent_policy="tile64"))
    _rewrite(heldout, lambda value: [row["samples_ms"].update(
        tile64_direct=[sample * 0.99 for sample in row["samples_ms"]["tile64"]])
        for row in value["timings"]])
    result = promotion.promote(selection, heldout, protocol)
    assert result["status"] == "not-promoted"
    assert result["policy"] == "tile64"
    assert any("incumbent" in reason for reason in result["reasons"])


def test_different_a100_variants_are_separate_measured_cohorts_not_pooled_times(retained_sessions):
    _, heldout, _ = retained_sessions
    _rewrite(heldout, lambda value: value["device"].update(name="NVIDIA A100 80GB PCIe", mem_bytes=80 << 30))
    result = promotion.promote(*retained_sessions)
    assert result["status"] == "promoted"
    assert len(result["entry"]["measurement_cohorts"]) == 2
    assert {row["device"] for row in result["rounds"]} == {
        "NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB PCIe"}


@pytest.mark.parametrize("field,value", [("relative_l2_fp32", float("nan")),
                                         ("relative_l2_fp32", 1.0), ("canaries", False)])
def test_passed_boolean_cannot_hide_numeric_or_canary_failure(retained_sessions, field, value):
    _, heldout, _ = retained_sessions
    _rewrite(heldout, lambda item: item["candidates"]["tile64_direct"]["checks"][0].update({field: value}))
    with pytest.raises(promotion.PromotionError, match="record validation failed"):
        promotion.promote(*retained_sessions)


def test_changed_binary_archive_is_rejected_before_performance_admission(retained_sessions):
    _, heldout, _ = retained_sessions
    result = json.loads(heldout.read_text())
    archive = heldout.parent / result["artifact_archive"]["path"]
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(promotion.PromotionError, match="archive is missing or changed"):
        promotion.promote(*retained_sessions)


def test_rehashed_tampered_disassembly_still_must_match_actual_cubin(retained_sessions):
    _, heldout, _ = retained_sessions
    result = json.loads(heldout.read_text())
    archive = heldout.parent / result["artifact_archive"]["path"]
    with zipfile.ZipFile(archive) as old:
        files = {name: old.read(name) for name in old.namelist()}
    path = "tile64_direct/native.sass"
    # Preserve its syntax and update both hashes: the independently retained
    # cubin's text still exposes the modified instruction word.
    text = files[path].decode()
    import re
    match = re.search(r"/\* 0x([0-9a-f]{16}) \*/", text)
    assert match
    replacement = f"{int(match.group(1), 16) ^ 1:016x}"
    files[path] = (text[:match.start(1)] + replacement + text[match.end(1):]).encode()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as new:
        for name, data in files.items():
            new.writestr(name, data)
    result["artifact_archive"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    result["candidates"]["tile64_direct"]["artifacts"]["native"]["sass_sha256"] = hashlib.sha256(files[path]).hexdigest()
    heldout.write_text(json.dumps(result, indent=2))
    with pytest.raises(promotion.PromotionError, match="SASS listing disagrees"):
        promotion.promote(*retained_sessions)


def test_missing_provenance_is_a_block_not_a_new_hardware_claim(retained_sessions):
    _, heldout, _ = retained_sessions
    _rewrite(heldout, lambda value: value.pop("session"))
    with pytest.raises(promotion.PromotionError, match="session provenance"):
        promotion.promote(*retained_sessions)


def test_freeze_refuses_to_overwrite_an_existing_protocol(retained_sessions):
    selection, _, protocol = retained_sessions
    with pytest.raises(promotion.PromotionError, match="already exists"):
        promotion.freeze_selection(selection, protocol,
                                   now=datetime(2026, 11, 10, 1, tzinfo=timezone.utc))
