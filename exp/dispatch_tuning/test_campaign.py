from dataclasses import asdict
import hashlib

from exp.dispatch_tuning.campaign import case_id, cases, local_identity, protocol
from openassembler.gemm_dispatch import candidate_plans
from openassembler.kernels.gemm import GemmPlan


def test_campaign_covers_unique_roles_legacy_shapes_and_64_only_geometry():
    rows = cases()
    assert len(rows) == 42
    assert len({row["id"] for row in rows}) == len(rows)
    by = {row["id"]: row for row in rows}
    for key in ("128x512x512_f16", "4096x2048x8192_bf16",
                "16384x4096x4096_f16", "4096x4096x4096_bf16",
                "4096x1024x1024_f16", "64x192x256_f16", "192x320x384_bf16"):
        assert key in by
        plan = GemmPlan(**by[key]["plan"])
        assert case_id(asdict(plan)) == key
        assert len(candidate_plans(plan)) >= 2
    assert by["64x192x256_f16"]["plan"]["tile"] == "64x64"


def test_protocol_freezes_candidates_identity_and_metadata_without_claiming_speed():
    rows = cases()[:2]
    value = protocol("test", "selection", rows)
    assert value["rounds"] == 3 and value["samples"] == 8
    assert value["graph_launches"] == 16
    assert value["source_identity"]["compiler_identity"] == local_identity()["compiler_identity"]
    assert len(value["source_identity"]["compiler_identity"]) == hashlib.sha256().digest_size * 2
    assert set(value["source_identity"]["harness_sha256"]) == {"campaign.py", "modal_campaign.py"}
    for row in value["cases"]:
        assert row["candidate_names"] == list(candidate_plans(GemmPlan(**row["plan"])))
    assert "global fastest implementation" in value["not_claimed"]
