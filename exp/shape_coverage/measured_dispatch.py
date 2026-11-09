"""Fail-closed selection of exact GEMM experiments, never production dispatch.

The role coverage matrix answers whether an input is representable. This module
answers a different question: does a proposed *exact* policy have enough matched,
current evidence to justify another integration experiment? It deliberately does
not infer evidence from a policy name, a role name, a single median, or old raw
run files. Normalized evidence must retain links/hashes to the original results.

``select_experiment`` returns ``selected`` only when selection and independent
held-out sessions pass every contract, numerical, binary/resource and timing
gate. That result is not authorization for production promotion, and makes no
whole-step performance claim. A faster OA policy may still lose to ptxas/cuBLAS;
those ratios are returned explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = 1
HASH = re.compile(r"[0-9a-f]{64}\Z")
WORKLOAD_FIELDS = frozenset({
    "role", "m", "n", "k", "batch_count", "dtype", "arch", "operand_layouts",
    "alignment_bytes", "tail_policy", "workspace_bytes", "accumulation_dtype",
    "output_dtype", "alpha", "beta", "numerical_policy", "timing_scope",
})
POLICY_FIELDS = frozenset({
    "tile_mnk", "warps", "stages", "cta_order", "fragment_schedule",
    "pointer_mode", "pointer_advance",
})
GATES = (
    "full_output", "canaries", "inputs_unchanged", "poisoned_graph",
    "ptxas_bitwise", "finite_output", "disassembly_valid",
    "policy_matches_encoded_stream", "resources_checked",
)


@dataclass(frozen=True)
class Requirements:
    """Experimental acceptance thresholds, not A100 latency assumptions."""

    min_speedup: float = 1.02
    min_rounds: int = 2
    min_samples_per_round: int = 8
    max_registers: int = 255
    max_shared_bytes: int = 163840
    max_local_bytes: int = 0
    max_spill_bytes: int = 0

    def __post_init__(self) -> None:
        if (not _positive_number(self.min_speedup) or self.min_speedup <= 1
                or not _integer(self.min_rounds, 1)
                or not _integer(self.min_samples_per_round, 1)
                or not all(_integer(x) for x in (self.max_registers,
                    self.max_shared_bytes, self.max_local_bytes, self.max_spill_bytes))):
            raise ValueError("acceptance thresholds must require a positive speedup and evidence")


def _same_json(left: Any, right: Any) -> bool:
    # JSON type equality must distinguish True from 1 and preserve exact fields.
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _positive_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _valid_source(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    hashes = value.get("sha256")
    # Git HEAD alone does not identify a dirty tree or the benchmark harness.
    return (isinstance(hashes, Mapping) and bool(hashes)
            and all(isinstance(name, str) and HASH.fullmatch(str(digest))
                    for name, digest in hashes.items())
            and any(name.startswith("openassembler/") for name in hashes)
            and any(name.startswith("exp/shape_coverage/") for name in hashes))


def _workload_reasons(key: Any) -> list[str]:
    if not isinstance(key, Mapping) or set(key) != WORKLOAD_FIELDS:
        return ["workload key must contain all and only the exact contract fields"]
    reasons = []
    if not all(_integer(key[x], 1) for x in ("m", "n", "k", "batch_count")):
        reasons.append("workload dimensions and batch count must be positive integers")
    if not _integer(key["workspace_bytes"]):
        reasons.append("workspace_bytes must be a nonnegative integer")
    if (not isinstance(key["alignment_bytes"], list)
            or len(key["alignment_bytes"]) != 3
            or not all(_integer(x, 1) and x & (x - 1) == 0
                       for x in key["alignment_bytes"])):
        reasons.append("alignment_bytes must identify A/B/C power-of-two alignment")
    if (not isinstance(key["operand_layouts"], list)
            or len(key["operand_layouts"]) != 3
            or not all(isinstance(x, str) and x for x in key["operand_layouts"])):
        reasons.append("operand_layouts must identify exact A/B/C storage")
    if not all(isinstance(key[x], str) and key[x] for x in (
        "role", "dtype", "arch", "tail_policy", "accumulation_dtype",
        "output_dtype", "numerical_policy", "timing_scope",
    )):
        reasons.append("workload semantic fields must be nonempty strings")
    if not all(type(key[x]) in (int, float) and math.isfinite(key[x])
               for x in ("alpha", "beta")):
        reasons.append("alpha and beta must be finite numbers")
    return reasons


def _policy_reasons(policy: Any) -> list[str]:
    if not isinstance(policy, Mapping) or set(policy) != POLICY_FIELDS:
        return ["policy must contain all and only the exact implementation fields"]
    reasons = []
    if (not isinstance(policy["tile_mnk"], list) or len(policy["tile_mnk"]) != 3
            or not all(_integer(x, 1) for x in policy["tile_mnk"])):
        reasons.append("tile_mnk must have three positive integers")
    if not all(_integer(policy[x], 1) for x in ("warps", "stages")):
        reasons.append("warps and stages must be positive integers")
    if not all(isinstance(policy[x], str) and policy[x] for x in (
        "cta_order", "fragment_schedule", "pointer_mode", "pointer_advance",
    )):
        reasons.append("policy implementation fields must be nonempty strings")
    return reasons


def evaluate_candidate(
    entry: Mapping[str, Any], *, workload: Mapping[str, Any],
    current_source: Mapping[str, Any], requirements: Requirements = Requirements(),
) -> dict[str, Any]:
    """Check one normalized proposal; missing/mismatched evidence rejects it.

    An evidence run stores exact ``workload``, ``policy``, ``source_identity``,
    ``artifact`` (cubin/SASS hashes and original-result path/hash), numerical
    ``gates`` and input families, ``resources``, matched ``comparators`` and
    paired timing rounds. At least one selection run and one independent
    held-out run are required. Held-out run IDs must be declared before results
    are selected (``frozen_before_measurement``); this declaration is auditable
    provenance, not a replacement for reviewing the referenced raw evidence.
    """
    reasons = _workload_reasons(workload)
    if not isinstance(entry, Mapping):
        return {"status": "rejected", "reasons": ["candidate must be an object"],
                "production_promoted": False}
    name = entry.get("id")
    if not isinstance(name, str) or not name:
        reasons.append("candidate id is missing")
    if type(entry.get("schema")) is not int or entry.get("schema") != SCHEMA:
        reasons.append("unsupported or missing normalized evidence schema")
    if not _same_json(entry.get("workload"), workload):
        reasons.append("exact workload contract mismatch")
    policy = entry.get("policy")
    reasons += _policy_reasons(policy)
    if not _valid_source(current_source):
        reasons.append("current source identity lacks compiler or harness hashes")
    if not _same_json(entry.get("source_identity"), current_source):
        reasons.append("stale or missing source identity")
    runs = entry.get("runs")
    if not isinstance(runs, list) or not runs:
        reasons.append("no evidence runs")
        runs = []

    ratios: list[dict[str, Any]] = []
    sessions: dict[str, set[str]] = {"selection": set(), "heldout": set()}
    run_ids: set[str] = set()
    for index, run in enumerate(runs):
        prefix = f"run[{index}]"
        if not isinstance(run, Mapping):
            reasons.append(f"{prefix}: evidence must be an object")
            continue
        run_id, session = run.get("id"), run.get("session_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            reasons.append(f"{prefix}: missing or duplicate run id")
        else:
            run_ids.add(run_id)
        split = run.get("split")
        if not isinstance(split, str) or split not in sessions or not isinstance(session, str) or not session:
            reasons.append(f"{prefix}: missing selection/heldout session provenance")
        else:
            sessions[split].add(session)
        if split == "heldout" and run.get("frozen_before_measurement") is not True:
            reasons.append(f"{prefix}: held-out choice was not frozen before measurement")
        for field, expected in (("workload", workload), ("policy", policy),
                                ("source_identity", current_source)):
            if not _same_json(run.get(field), expected):
                reasons.append(f"{prefix}: {field} mismatch")
        gates = run.get("gates", {})
        if not isinstance(gates, Mapping):
            gates = {}
        for gate in GATES:
            if gates.get(gate) is not True:
                reasons.append(f"{prefix}: gate missing or failed: {gate}")
        families = run.get("input_families", [])
        if (not isinstance(families, list)
                or not {"zero", "integer", "normal"}.issubset(
                    x for x in families if isinstance(x, str))):
            reasons.append(f"{prefix}: required input families not checked")
        artifact = run.get("artifact", {})
        if not isinstance(artifact, Mapping):
            artifact = {}
        for field in ("cubin_sha256", "sass_sha256", "raw_result_sha256"):
            if not HASH.fullmatch(str(artifact.get(field, ""))):
                reasons.append(f"{prefix}: invalid or missing {field}")
        if not isinstance(artifact.get("raw_result_path"), str) or not artifact.get("raw_result_path"):
            reasons.append(f"{prefix}: original result path missing")
        if not isinstance(artifact.get("entry"), str) or not artifact.get("entry"):
            reasons.append(f"{prefix}: kernel entry missing")
        resources = run.get("resources", {})
        if not isinstance(resources, Mapping):
            resources = {}
        for field, lower, upper in (
            ("registers", 1, requirements.max_registers),
            ("shared_bytes", 0, requirements.max_shared_bytes),
            ("local_bytes", 0, requirements.max_local_bytes),
            ("spill_bytes", 0, requirements.max_spill_bytes),
        ):
            if not _integer(resources.get(field), lower) or resources[field] > upper:
                reasons.append(f"{prefix}: missing or out-of-budget resource: {field}")
        comparators = run.get("comparators", {})
        if not isinstance(comparators, Mapping):
            comparators = {}
        for comparator in ("same_source_ptxas", "library"):
            control = comparators.get(comparator, {})
            if not isinstance(control, Mapping):
                control = {}
            if (not isinstance(control.get("name"), str) or not control.get("name")
                    or control.get("matched_contract") is not True
                    or control.get("full_output_passed") is not True):
                reasons.append(f"{prefix}: missing or unmatched {comparator} control")
        timing = run.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        if timing.get("paired") is not True or timing.get("randomized_order") is not True:
            reasons.append(f"{prefix}: timing is not paired with randomized method order")
        if timing.get("scope") != workload.get("timing_scope"):
            reasons.append(f"{prefix}: timing scope mismatch")
        rounds = timing.get("rounds")
        if not isinstance(rounds, list) or len(rounds) < requirements.min_rounds:
            reasons.append(f"{prefix}: too few paired timing rounds")
            rounds = []
        for round_index, measurements in enumerate(rounds):
            label = f"{prefix}/round[{round_index}]"
            if not isinstance(measurements, Mapping):
                reasons.append(f"{label}: missing paired measurements")
                continue
            if (not _integer(measurements.get("samples"), requirements.min_samples_per_round)
                    or not all(_positive_number(measurements.get(field)) for field in (
                        "oa_baseline_ms", "oa_candidate_ms", "ptxas_ms", "library_ms"))):
                reasons.append(f"{label}: invalid timing values or insufficient samples")
                continue
            candidate_ms = measurements["oa_candidate_ms"]
            speedup = measurements["oa_baseline_ms"] / candidate_ms
            ratios.append({"run": run_id, "round": round_index,
                           "oa_baseline_speedup": speedup,
                           "oa_over_ptxas": candidate_ms / measurements["ptxas_ms"],
                           "oa_over_library": candidate_ms / measurements["library_ms"]})
            if speedup < requirements.min_speedup:
                reasons.append(f"{label}: candidate does not meet minimum OA baseline speedup")
    if not sessions["selection"] or not sessions["heldout"]:
        reasons.append("selection and held-out sessions are both required")
    if sessions["selection"] & sessions["heldout"]:
        reasons.append("selection and held-out evidence reuse a session")
    return {"id": name, "status": "rejected" if reasons else "eligible-experiment",
            "reasons": reasons, "ratios": ratios, "production_promoted": False}


def select_experiment(
    entries: Sequence[Mapping[str, Any]], *, workload: Mapping[str, Any],
    current_source: Mapping[str, Any], requirements: Requirements = Requirements(),
) -> dict[str, Any]:
    """Return the largest worst-round baseline win among fully gated entries.

    No eligible entry returns ``fallback``. This function cannot choose or run a
    production fallback: whether even the default native kernel supports the
    workload is a separate contract decision.
    """
    decisions = [evaluate_candidate(entry, workload=workload,
                                    current_source=current_source,
                                    requirements=requirements) for entry in entries]
    accepted = [decision for decision in decisions
                if decision["status"] == "eligible-experiment"]
    winner = max(accepted, key=lambda decision: min(
        row["oa_baseline_speedup"] for row in decision["ratios"]), default=None)
    return {"status": "selected" if winner else "fallback",
            "selected_experiment": winner["id"] if winner else None,
            "decisions": decisions, "production_promoted": False}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal", type=Path,
                        help="normalized JSON with entries, workload and current_source")
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    result = select_experiment(proposal.get("entries", []),
                               workload=proposal.get("workload", {}),
                               current_source=proposal.get("current_source", {}))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
