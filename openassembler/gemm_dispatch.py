"""Exact-contract native GEMM selection and a bounded on-device search space.

The built-in table is measured evidence, not a universal performance model.
Unknown shapes keep the reference policy. GPU tuning can compare all admitted
policies, including ones that lost on other shapes, without changing arithmetic.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from openassembler.kernels.gemm import GemmPlan, generate

TABLE = Path(__file__).with_name("kernels") / "sm80_gemm_dispatch.json"
POLICIES = {
    "baseline": {},
    "tile64": {"tile": "64x64"},
    "tile64_direct": {"tile": "64x64", "pointer_mode": "direct"},
    "combined": {"schedule": "combined"},
    "morton8": {"cta_order": "morton8"},
    "grouped_n8": {"cta_order": "grouped_n8"},
    "pointer_direct": {"pointer_mode": "direct"},
    "pointer_early": {"pointer_advance": "early"},
    "pointer_direct_early": {"pointer_mode": "direct", "pointer_advance": "early"},
    "load_next": {"fragment_schedule": "load_next"},
}


def compiler_identity() -> str:
    """Content identity of GEMM code generation, validation and encoding data.

    An unrelated rowwise kernel or CLI help edit must not invalidate measured
    GEMM binaries. The generated body, checked plan and retained native text
    additionally bind each record to its actual implementation.
    """
    root = Path(__file__).resolve().parent.parent
    paths = []
    for folder in ("backend", "opt", "ptx", "isa", "container"):
        paths += list((root / "openassembler" / folder).rglob("*.py"))
    paths += [root / "openassembler" / name for name in
              ("gemm.py", "gemm_dispatch.py", "gemm_tune.py", "gemm_tune_modal.py", "build.py",
               "project.py", "oracle.py", "gpu/cuda_driver.py", "kernels/gemm.py", "kernels/_gemm64.py")]
    paths += list((root / "specs" / "sm_80").glob("*.json"))
    paths += [root / "corpus" / "sm_80" / name for name in ("pairs.jsonl", "machine_model.json")]
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(path.relative_to(root).as_posix().encode() + b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def contract_key(plan: GemmPlan) -> dict:
    """All variable semantics of this unbatched, aligned, zero-workspace API."""
    return {"m": plan.m, "n": plan.n, "k": plan.k, "dtype": plan.dtype,
            "arch": plan.arch, "layout": plan.layout, "output_dtype": plan.output_dtype,
            "accumulation": "f32", "epilogue": plan.epilogue,
            "alpha": 1, "beta": 0, "batch_count": 1,
            "a_alignment": 16, "b_alignment": 16, "c_alignment": 8,
            "workspace_bytes": 0, "tails": "none"}


def baseline(plan: GemmPlan) -> GemmPlan:
    # A manually requested 64 tile remains the reference if 128 is inadmissible.
    tile = "128x128" if plan.m % 128 == plan.n % 128 == 0 else "64x64"
    return replace(plan, tile=tile, schedule="stagepair", cta_order="m_fast",
                   pointer_mode="legacy", pointer_advance="late", fragment_schedule="source")


def policy_plan(plan: GemmPlan, policy: str) -> GemmPlan:
    if policy not in POLICIES:
        raise ValueError(f"unknown native GEMM policy: {policy}")
    return replace(baseline(plan), **POLICIES[policy])


def candidate_plans(plan: GemmPlan) -> dict[str, GemmPlan]:
    """Enumerate representable choices, never infer that a candidate is faster.

    Fused epilogues retain their own contract; unsupported combinations are
    refused by GemmPlan, not silently repaired. Shape-specific losers remain
    available for a new matched GPU trial.
    """
    result = {}
    seen = set()
    for name in POLICIES:
        try:
            candidate = policy_plan(plan, name)
        except ValueError:
            continue
        identity = json.dumps(asdict(candidate), sort_keys=True)
        if identity not in seen:
            result[name] = candidate
            seen.add(identity)
    return result


def select_plan(plan: GemmPlan, *, table: dict | None = None) -> tuple[GemmPlan, dict]:
    """Select an exact measured native policy, otherwise return the baseline.

    This function is deterministic and performs no remote work. Numerical
    validity of an option and performance admission are distinct. Source/data
    changes invalidate the table until its current kernels are revalidated.
    """
    reference = baseline(plan)
    decision = {"schema": 1, "mode": "measured", "policy": "baseline",
                "contract": contract_key(reference), "reason": "no exact measured entry",
                "scope": "native A100 kernel only; not a whole-model or global fastest guarantee"}
    if table is None:
        try:
            table = json.loads(TABLE.read_text())
        except (OSError, ValueError):
            return reference, {**decision, "reason": "measured table unavailable"}
    if not isinstance(table, dict) or table.get("schema") != 1:
        return reference, {**decision, "reason": "invalid measured table"}
    if table.get("compiler_identity") != compiler_identity():
        return reference, {**decision, "reason": "measured table belongs to a different compiler"}
    entries = table.get("entries")
    if not isinstance(entries, list):
        return reference, {**decision, "reason": "invalid measured entries"}
    matches = [x for x in entries if isinstance(x, dict)
               and x.get("contract") == contract_key(reference)]
    if len(matches) != 1:
        return reference, {**decision, "reason": "no unique exact measured entry"}
    entry = matches[0]
    if entry.get("status") != "promoted" or entry.get("gates_passed") is not True:
        return reference, {**decision, "reason": "entry has not passed promotion gates"}
    try:
        chosen = policy_plan(reference, entry["policy"])
    except (ValueError, KeyError, TypeError):
        return reference, {**decision, "reason": "invalid or unsupported measured policy"}
    if hashlib.sha256(generate(chosen).encode()).hexdigest() != entry.get("ptx_body_sha256"):
        return reference, {**decision, "reason": "generated policy source changed"}
    return chosen, {**decision, "policy": entry["policy"], "reason": "exact measured contract",
                    "evidence": entry.get("evidence", []), "device_scope": table.get("device_scope"),
                    "measured_speedup": entry.get("speedup_range")}
