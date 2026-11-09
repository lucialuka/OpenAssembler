"""Audit transformer GEMM roles against the current OpenAssembler plan contract.

This is a static coverage experiment. It distinguishes dimension geometry from
operand layout, batching, tails and per-shape performance evidence; passing a
``GemmPlan`` shape check alone does not make a model role runnable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openassembler.kernels.gemm import GemmPlan  # noqa: E402


MODELS = (
    {"id": "base_report_context", "batch": 8, "sequence": 128, "hidden": 512,
     "heads": 8, "vocab": 512, "precision": ("f16", "bf16")},
    {"id": "small_fp32_context", "batch": 1, "sequence": 128, "hidden": 512,
     "heads": 8, "vocab": 32768, "precision": ("f16", "bf16")},
    {"id": "mid_gqa_style", "batch": 4, "sequence": 128, "hidden": 768,
     "heads": 12, "vocab": 50257, "precision": ("f16", "bf16")},
    {"id": "medium_long", "batch": 4, "sequence": 256, "hidden": 1024,
     "heads": 16, "vocab": 50257, "precision": ("f16", "bf16")},
    {"id": "large_long", "batch": 4, "sequence": 1024, "hidden": 2048,
     "heads": 16, "vocab": 50257, "precision": ("f16", "bf16")},
    {"id": "ragged_microbatch", "batch": 3, "sequence": 127, "hidden": 768,
     "heads": 12, "vocab": 50257, "precision": ("f16", "bf16")},
)


def _roles(model: dict) -> list[dict]:
    m = model["batch"] * model["sequence"]
    h = model["hidden"]
    f = 4 * h
    v = model["vocab"]
    s = model["sequence"]
    d = h // model["heads"]
    return [
        {"role": "qkv_projection", "m": m, "n": 3 * h, "k": h,
         "direct_rcr": True, "batch_count": 1, "reason": "merged QKV linear; weights [3H,H]"},
        {"role": "attention_output_projection", "m": m, "n": h, "k": h,
         "direct_rcr": True, "batch_count": 1, "reason": "dense output linear; weights [H,H]"},
        {"role": "ffn_up", "m": m, "n": f, "k": h,
         "direct_rcr": True, "batch_count": 1, "reason": "first FFN linear; weights [4H,H]"},
        {"role": "ffn_down", "m": m, "n": h, "k": f,
         "direct_rcr": True, "batch_count": 1, "reason": "second FFN linear; weights [H,4H]"},
        {"role": "vocabulary_projection", "m": m, "n": v, "k": h,
         "direct_rcr": True, "batch_count": 1, "reason": "dense language-model head; vocab output"},
        {"role": "qkv_dX", "m": m, "n": h, "k": 3 * h,
         "direct_rcr": False, "batch_count": 1,
         "reason": "dQKV[M,3H] @ Wqkv[3H,H]; weight transpose/packing required for RCR"},
        {"role": "qkv_dW", "m": 3 * h, "n": h, "k": m,
         "direct_rcr": False, "batch_count": 1,
         "reason": "dQKV.T @ X; transpose/layout preparation required"},
        {"role": "attention_output_dX", "m": m, "n": h, "k": h,
         "direct_rcr": False, "batch_count": 1,
         "reason": "dY[M,H] @ Wo[H,H]; square dimensions do not remove weight transpose requirement"},
        {"role": "attention_output_dW", "m": h, "n": h, "k": m,
         "direct_rcr": False, "batch_count": 1,
         "reason": "dY.T @ attention_output; transpose/layout preparation required"},
        {"role": "ffn_up_dX", "m": m, "n": h, "k": f,
         "direct_rcr": False, "batch_count": 1, "reason": "dY[M,4H] @ W[4H,H]; W must be transposed/packed for RCR"},
        {"role": "ffn_up_dW", "m": f, "n": h, "k": m,
         "direct_rcr": False, "batch_count": 1, "reason": "dY.T @ X; both operands require transpose/layout preparation"},
        {"role": "ffn_down_dX", "m": m, "n": f, "k": h,
         "direct_rcr": False, "batch_count": 1, "reason": "dY[M,H] @ W[H,4H]; W requires transpose/packing"},
        {"role": "ffn_down_dW", "m": h, "n": f, "k": m,
         "direct_rcr": False, "batch_count": 1, "reason": "dY.T @ X; both operands require transpose/layout preparation"},
        {"role": "attention_qk_single_head", "m": s, "n": s, "k": d,
         "direct_rcr": True, "batch_count": model["batch"] * model["heads"],
         "reason": "one Q[S,D] @ K[S,D].T is RCR; model requires per-head batched launch/offsets and masking"},
        {"role": "attention_av_single_head", "m": s, "n": d, "k": s,
         "direct_rcr": False, "batch_count": model["batch"] * model["heads"],
         "reason": "P[S,S] @ V[S,D] needs V.T packing for RCR and per-head batched launch"},
    ]


def forward_gemm_cases(*, heldout: bool = False) -> list[dict]:
    """Return role-labelled linear GEMMs that can run through OA's RCR CLI."""
    selected = [next(model for model in MODELS if model["id"] == "medium_long")]
    if heldout:
        selected.extend(model for model in MODELS
                        if model["id"] in {"base_report_context", "small_fp32_context", "large_long"})
    rows = []
    for model in selected:
        for role in _roles(model):
            if not role["direct_rcr"] or role["batch_count"] != 1:
                continue
            if role["role"] not in {"qkv_projection", "attention_output_projection",
                                    "ffn_up", "ffn_down"}:
                continue
            for dtype in model["precision"]:
                try:
                    GemmPlan(role["m"], role["n"], role["k"], dtype=dtype)
                except ValueError:
                    continue
                rows.append({"model_case": model["id"], "role": role["role"],
                             "m": role["m"], "n": role["n"], "k": role["k"],
                             "dtype": dtype, "batch": model["batch"],
                             "sequence": model["sequence"], "hidden": model["hidden"]})
    return rows


def _candidate_options(shape: tuple[int, int, int], dtype: str) -> dict:
    m, n, k = shape
    accepted = []
    rejected = []
    try:
        GemmPlan(m, n, k, dtype=dtype)
        accepted.append({"name": "source_m_fast", "production_option": True,
                         "performance_status": "no exact per-role timing evidence"})
    except (ValueError, TypeError) as exc:
        rejected.append({"name": "source_m_fast", "reason": str(exc)})
        return {"eligible_options": accepted, "rejected_options": rejected}

    if n // 128 % 8 == 0:
        try:
            GemmPlan(m, n, k, dtype=dtype, cta_order="grouped_n8")
            accepted.append({"name": "grouped_n8", "production_option": True,
                             "performance_status": "opt-in; measure this exact role/shape"})
        except ValueError as exc:
            rejected.append({"name": "grouped_n8", "reason": str(exc)})
    else:
        rejected.append({"name": "grouped_n8", "reason": "N tile count is not divisible by eight"})

    try:
        GemmPlan(m, n, k, dtype=dtype, cta_order="morton8")
        accepted.append({"name": "morton8", "production_option": True,
                         "performance_status": "opt-in; measure this exact role/shape"})
    except ValueError as exc:
        rejected.append({"name": "morton8", "reason": str(exc)})

    try:
        GemmPlan(m, n, k, dtype=dtype, fragment_schedule="load_next")
        accepted.append({"name": "load_next", "production_option": True,
                         "performance_status": "opt-in; exact-shape/dtype evidence exists, not role-level whole-step evidence"})
    except ValueError as exc:
        rejected.append({"name": "load_next", "reason": str(exc)})
    return {"eligible_options": accepted, "rejected_options": rejected}


def build_manifest() -> dict:
    rows = []
    for model in MODELS:
        for role in _roles(model):
            dims = (role["m"], role["n"], role["k"])
            for dtype in model["precision"]:
                options = _candidate_options(dims, dtype)
                dimensions_accepted = any(x["name"] == "source_m_fast" for x in options["eligible_options"])
                layout_accepted = bool(role["direct_rcr"])
                batching_accepted = role["batch_count"] == 1
                admitted = dimensions_accepted and layout_accepted and batching_accepted
                rows.append({
                    "model_case": model["id"], "role": role["role"],
                    "shape_mnk": list(dims), "dtype": dtype,
                    "role_description": role["reason"],
                    "gemm_plan_dimensions_accepted": dimensions_accepted,
                    "operand_layout_accepted_without_transform": layout_accepted,
                    "batch_launch_supported": batching_accepted,
                    "native_cli_role_route_admitted": admitted,
                    "required_preparation": [] if layout_accepted else ["transpose_or_pack"],
                    "batch_count": role["batch_count"],
                    "head_dim": model["hidden"] // model["heads"]
                    if role["role"] in {"attention_qk_single_head", "attention_av_single_head"} else None,
                    "attention_mask_contract": "not represented by GemmPlan" if role["role"]
                    in {"attention_qk_single_head", "attention_av_single_head"} else None,
                    "candidate_options": options,
                    "evidence_scope": "static current-source contract audit; no performance result",
                })
    source = ROOT / "openassembler/kernels/gemm.py"
    return {
        "schema": 1,
        "date": "2026-11-09",
        "purpose": "Map representative transformer GEMM roles to current OA shape/layout/launch contracts.",
        "production_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "current_contract": {
            "operation": "C[M,N] = A[M,K] @ B[N,K].T",
            "operand_layout": "contiguous row-major A[M,K] and B[N,K]",
            "dtypes": ["f16", "bf16"], "output_accumulation": "f32",
            "all_dimensions_multiple_of": 128,
            "alignment_bytes": {"A": 16, "B": 16, "C": 8},
            "tile_mnk": [128, 128, 64], "warps": 4, "stages": 2,
            "dynamic_shared_bytes": 65536,
            "arbitrary_strides_tails_transposes_batched_launch": False,
        },
        "manifest_key_proposal": [
            "role", "m", "n", "k", "batch_count", "head_dim", "dtype", "arch",
            "operand_layouts", "alignment_class", "tail_policy", "workspace_bytes",
            "tile_mnk", "stages", "cta_order", "fragment_schedule", "pointer_mode",
            "pointer_advance", "evidence_ids",
        ],
        "manifest_semantics": {
            "contract_supported": "All shape/layout/launch/numerical constraints pass.",
            "measured_candidate": "An exact workload key has full correctness/resource evidence and paired timing.",
            "fallback": "Use only the validated default where its contract applies; otherwise do not select native GEMM.",
            "no_auto_dispatch": "A syntactically accepted optional policy is not a measured winner for every role sharing its dimensions.",
        },
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "native_cli_role_routes_admitted": sum(r["native_cli_role_route_admitted"] for r in rows),
            "shape_supported_but_role_layout_or_batch_blocked": sum(
                r["gemm_plan_dimensions_accepted"] and not r["native_cli_role_route_admitted"] for r in rows),
            "dimension_contract_rejected": sum(not r["gemm_plan_dimensions_accepted"] for r in rows),
        },
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("role-coverage-manifest.json")
    manifest = build_manifest()
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(json.dumps(manifest["summary"], indent=2))
