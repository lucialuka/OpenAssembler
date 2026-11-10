from __future__ import annotations

import json

from exp.dispatch_tuning.feature_inventory import ROOT, RULES, inventory


def test_every_implemented_catalogue_feature_has_an_explicit_contract_audit():
    result = inventory()
    expected = {row["id"] for row in json.loads((ROOT / "docs/research/catalogue.json").read_text())["ideas"]
                if row["status"] == "implemented"}
    assert {row["id"] for row in result["features"]} == expected == set(RULES)
    assert result["implemented_count"] == len(expected)
    assert len(result["features"]) == len(expected)
    assert result["not_an_all_shapes_performance_proof"] is True
    for row in result["features"]:
        assert row["applicability"] and row["current_selection"] and row["next_action"]
        assert row["code_paths"] and row["retained_references"]
        assert (ROOT / row["dossier"]).is_file()


def test_correctness_and_numeric_semantics_are_not_optional_shape_performance_rules():
    rows = {row["id"]: row for row in inventory()["features"]}
    for name in ("scalar-dependency-correctness", "state-verification-diagnostics", "cubin-metadata-splicing"):
        assert rows[name]["classification"] in {"mandatory-correctness", "mandatory-validation", "correctness-infrastructure"}
        assert rows[name]["tuning_axes"] == []
    for name in ("native-tensor-gemm", "small-grid-tensor-cta-tiles", "cooperative-layernorm",
                 "cooperative-softmax", "grouped-n-cta-swizzle", "morton-cta-order"):
        assert rows[name]["classification"] == "shape-sensitive-kernel"
        assert rows[name]["tuning_axes"]
    pending = inventory()["pending_nonimplemented_priority"]
    assert pending[0]["id"] == "fused-backward-reductions"
    assert pending[0]["status"] == "validated-not-integrated"
