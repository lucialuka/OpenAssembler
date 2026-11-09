import hashlib
import json

import pytest

from latency_profiles import CLOCKED_EVIDENCE, latency_profile


def test_clocked_profile_has_traceable_partial_calibration_and_no_encoding_permission():
    profile = latency_profile("a100_clocked_24_8")
    assert profile.latency.accum_raw == 24
    assert profile.latency.hmma_issue == 8
    assert set(profile.calibrated_fields) == {"accum_raw", "hmma_issue"}
    assert {"ldsm_to_hmma", "async_to_wait", "wait_to_ldsm", "address_war", "observer"} <= set(
        profile.uncalibrated_fields)
    assert profile.provenance["sha256"] == hashlib.sha256(CLOCKED_EVIDENCE.read_bytes()).hexdigest()
    assert profile.provenance["observed_effective_cycles_per_hmma"]["1"] == 24.0234375
    assert profile.encoding_allowed is False
    assert profile.record()["cycle_predictions_are_hardware_measurements"] is False


def test_legacy_toy_distances_are_explicitly_uncalibrated():
    profile = latency_profile()
    assert profile.calibrated_fields == ()
    assert "accum_raw" in profile.uncalibrated_fields
    assert profile.provenance["hardware_evidence"] is None


def test_zero_work_or_optimized_away_probe_cannot_supply_a_calibrated_profile(tmp_path):
    data = json.loads(CLOCKED_EVIDENCE.read_bytes())
    for row in data["clocked"]:
        if row["kind"] == "hmma":
            row["cycles_per_hmma"] = 0
    path = tmp_path / "optimized-away.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="degenerate"):
        latency_profile("a100_clocked_24_8", evidence_path=path)
