from __future__ import annotations

import json

import pytest

from exp.dispatch_tuning import promote as campaign


def test_installation_ignores_cached_decision_and_reaudits_archive(tmp_path, monkeypatch):
    """A cached winner may never bypass current archive validation on install."""
    monkeypatch.setattr(campaign, "compiler_identity", lambda: "current")
    case = "128x512x512_f16"
    selection = tmp_path / "selection" / case
    heldout = tmp_path / "heldout" / case
    selection.mkdir(parents=True)
    heldout.mkdir(parents=True)
    for path in (selection / "result.json", heldout / "result.json", selection / "frozen-protocol.json"):
        path.write_text("{}\n")
    (tmp_path / "frozen-selection.json").write_text(json.dumps({
        "compiler_identity": "current", "cases": [{"id": case,
        "selection_record": f"selection/{case}/result.json",
        "protocol": f"selection/{case}/frozen-protocol.json"}]}))
    cache = tmp_path / "promotion"
    cache.mkdir()
    inputs = {"compiler_identity": "current",
              "selection_sha256": campaign._digest(selection / "result.json"),
              "heldout_sha256": campaign._digest(heldout / "result.json"),
              "protocol_sha256": campaign._digest(selection / "frozen-protocol.json")}
    (cache / f"{case}.json").write_text(json.dumps({"audit_inputs": inputs,
        "status": "promoted", "policy": "forged", "production_promoted": True,
        "entry": {"forged": True}}))
    archive = heldout / "tuning.artifacts.zip"
    archive.write_bytes(b"tampered archive")
    calls = []

    def reject_archive(*args):
        calls.append(args)
        raise ValueError("archive hash mismatch")

    monkeypatch.setattr(campaign, "promote", reject_archive)
    table = tmp_path / "production.json"
    with pytest.raises(ValueError, match="archive hash mismatch"):
        campaign.promote_campaign(tmp_path, table_path=table)
    assert len(calls) == 1
    assert not table.exists()
