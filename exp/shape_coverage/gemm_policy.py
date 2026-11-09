"""Pure source/manifest guards for the role-labelled A100 experiment."""
from __future__ import annotations

from dataclasses import asdict
import hashlib

from openassembler.gemm import source_for
from openassembler.kernels.gemm import GemmPlan

FRAGMENT_SPLITS = {f"fragment_split{split}": split for split in (2, 4, 6)}


def policy_source(plan: GemmPlan, policy: str) -> tuple[str, dict | None]:
    if policy == "tile64":
        from exp.shape_coverage.small_tile_gemm import source_for_tile64
        return source_for_tile64(plan)
    if policy in FRAGMENT_SPLITS:
        from exp.tensor_scheduler.fragment_split_search import source_for_split
        return source_for_split(plan, split_n=FRAGMENT_SPLITS[policy])
    if policy.startswith("fragment_split"):
        raise ValueError("unknown fragment split experiment")
    return source_for(plan), None


def verify_compile_record(plan: GemmPlan, policy: str, source: str, kernel: dict,
                          certificate: dict | None) -> dict:
    """Reject lost production policy annotations or unexpected native policy.

    A non-integrated fragment source intentionally has no checked launch
    annotation.  Only its bounded, separately certified baseline launch may
    be used by this experiment; it cannot become an inferred CLI route.
    """
    actual = kernel.get("gemm") or {}
    experiment = policy in FRAGMENT_SPLITS or policy == "tile64"
    if kernel.get("backend") != "openassembler":
        raise AssertionError({"not_native": kernel})
    tensor = kernel.get("tensor", {})
    expected_tensor = {
        "schedule": "source-order-hmma8-epoch-drain15",
        "pointer_mode": plan.pointer_mode,
        "pointer_advance": plan.pointer_advance,
    }
    for key, wanted in expected_tensor.items():
        if tensor.get(key) != wanted:
            raise AssertionError({"native_tensor_policy_mismatch": key,
                                  "wanted": wanted, "actual": tensor})
    if experiment:
        if actual or certificate is None:
            raise AssertionError("experimental fragment source must use explicit manual launch evidence")
        if certificate.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
            raise AssertionError("fragment source differs from its certificate")
        if (not certificate.get("full_work_preserved")
                or not certificate.get("k_order_preserved")
                or (policy in FRAGMENT_SPLITS and (
                    certificate.get("split_n") != FRAGMENT_SPLITS[policy]
                    or certificate.get("live_window_overwrite") is not False))):
            raise AssertionError("invalid bounded fragment certificate")
        if policy == "tile64" and (
                certificate.get("tile_mnk") != [64, 64, 64]
                or not certificate.get("output_ownership_exact_once")
                or not certificate.get("copy_coverage_exact_once")
                or not certificate.get("ldsm_aligned_in_bounds")
                or certificate.get("manual_launch") != {
                    "grid": [plan.m // 64, plan.n // 64, 1], "threads": 128, "smem": 32768}):
            raise AssertionError("invalid tile64 launch/layout certificate")
    else:
        for key in ("pointer_mode", "pointer_advance", "fragment_schedule", "cta_order"):
            if actual.get(key) != getattr(plan, key):
                raise AssertionError({"policy_contract_lost": key, "kernel": kernel})
    return {
        **actual,
        "checked_gemm_contract": not experiment,
        "manual_launch_experiment": experiment,
        "requested_plan": asdict(plan),
        "requested_tensor_policy": {**expected_tensor, "hmma_stall": 8, "epoch_drain": 15},
        "native_tensor_policy": tensor,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "fragment_certificate": certificate,
        "manual_launch": ((certificate or {}).get("manual_launch", plan.launch)) if experiment else None,
    }
