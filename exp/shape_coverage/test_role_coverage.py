from __future__ import annotations

from exp.shape_coverage.role_coverage import (
    MODELS, _candidate_options, _roles, build_manifest, forward_gemm_cases,
)


def test_forward_roles_are_generated_from_model_shapes_and_backward_are_layout_gated():
    model = next(x for x in MODELS if x["id"] == "medium_long")
    roles = {x["role"]: x for x in _roles(model)}
    assert (roles["qkv_projection"]["m"], roles["qkv_projection"]["n"],
            roles["qkv_projection"]["k"]) == (1024, 3072, 1024)
    assert roles["qkv_projection"]["direct_rcr"] is True
    assert roles["ffn_up_dW"]["direct_rcr"] is False
    assert (roles["qkv_dW"]["m"], roles["qkv_dW"]["n"],
            roles["qkv_dW"]["k"]) == (3072, 1024, 1024)
    assert roles["attention_output_dX"]["direct_rcr"] is False
    assert roles["attention_av_single_head"]["direct_rcr"] is False


def test_load_next_is_admitted_only_for_measured_exact_shapes_and_precision():
    supported = _candidate_options((1024, 4096, 1024), "f16")
    names = {x["name"] for x in supported["eligible_options"]}
    assert "load_next" in names
    rejected = _candidate_options((1024, 3072, 1024), "f16")
    assert "load_next" not in {x["name"] for x in rejected["eligible_options"]}


def test_manifest_separates_shape_admission_from_role_route_admission():
    result = build_manifest()
    rows = result["cases"]
    assert result["summary"]["cases"] == len(MODELS) * 15 * 2
    qk = next(r for r in rows if r["model_case"] == "large_long"
              and r["role"] == "attention_qk_single_head" and r["dtype"] == "f16")
    assert qk["gemm_plan_dimensions_accepted"] is True
    assert qk["native_cli_role_route_admitted"] is False
    assert qk["batch_launch_supported"] is False
    assert qk["batch_count"] == 64
    assert qk["head_dim"] == 128
    output = next(r for r in rows if r["role"] == "attention_output_projection")
    assert output["attention_mask_contract"] is None


def test_ragged_and_non_multiple_shapes_fail_closed_without_claiming_speedups():
    result = build_manifest()
    ragged = [r for r in result["cases"] if r["model_case"] == "ragged_microbatch"]
    assert all(not r["gemm_plan_dimensions_accepted"] for r in ragged)
    assert all(r["evidence_scope"].startswith("static current-source") for r in result["cases"])


def test_forward_measurement_matrix_adds_heldout_small_and_large_roles():
    current = forward_gemm_cases()
    heldout = forward_gemm_cases(heldout=True)
    assert len(current) == 8
    assert len(heldout) == 32
    assert {case["model_case"] for case in current} == {"medium_long"}
    assert {case["model_case"] for case in heldout} == {
        "medium_long", "base_report_context", "small_fp32_context", "large_long"
    }
    assert all(case["role"] in {"qkv_projection", "attention_output_projection",
                                "ffn_up", "ffn_down"} for case in heldout)
