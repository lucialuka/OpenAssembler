"""Inventory immutable role-GEMM raw runs without promoting a policy.

This audit checks the *recorded evidence* and its internal consistency. It does
not rerun the GPU, redisassemble missing binaries, establish first divergence,
or turn a manually asserted boolean into an independent proof. Schema versions
have different evidence scopes; legacy passing numerical rows do not verify a
lost pointer-lowering annotation. Timing values are not pooled across workers,
GPU types, roles or numerical contracts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INVALID_POINTER_STAMPS = ("20261109T082832.", "20261109T083147.", "20261109T083359.")
INVALID_POINTER_POLICIES = {"pointer_direct", "pointer_direct_early"}
INPUT_FAMILIES = ("zero", "integer", "normal")
CORE_EXPERIMENT_SOURCES = {
    "exp/shape_coverage/a100_role_gemm_modal.py",
    "exp/shape_coverage/role_coverage.py",
    "exp/shape_coverage/gemm_policy.py",
    "exp/tensor_scheduler/fragment_split_search.py",
    "exp/tensor_scheduler/load_next_compute_current.py",
}


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _source_scope(raw: Mapping) -> dict:
    identity = _mapping(raw.get("source_identity"))
    hashes = _mapping(identity.get("sha256"))
    compiler = [name for name, digest in hashes.items()
                if isinstance(name, str) and name.startswith("openassembler/") and _sha(digest)]
    experiment = [name for name, digest in hashes.items()
                  if isinstance(name, str) and name.startswith("exp/") and _sha(digest)]
    required = set(CORE_EXPERIMENT_SOURCES)
    methodology = _mapping(raw.get("methodology"))
    if methodology.get("tile_sweep") is True:
        required.add("exp/shape_coverage/small_tile_gemm.py")
    if methodology.get("counter_replay_after_timing") is True:
        required.update({"exp/shape_coverage/profile_worker.py", "exp/a100_counters/run.py"})
    manifest_complete = bool(compiler) and all(_sha(hashes.get(name)) for name in required)
    return {
        "git_head": identity.get("git_head"), "source_file_hash_count": len(hashes),
        "compiler_file_hash_count": len(compiler),
        "experiment_file_hash_count": len(experiment),
        "harness_sha256": identity.get("harness_sha256"),
        "separate_experiment_hashes": identity.get("experiment_sha256"),
        "compiler_and_known_harness_dependencies_in_source_manifest": manifest_complete,
        "required_experiment_paths_missing_from_manifest": sorted(required - hashes.keys()),
        "source_unchanged_during_run_reported": raw.get("source_unchanged_during_run"),
        "unchanged_flag_includes_known_harness_dependencies": manifest_complete
            and raw.get("source_unchanged_during_run") is True,
        "scope": "Recorded file manifest and pre/post comparison flag; no claim that every "
                 "environment/library dependency was hashed or independently remeasured.",
    }


def _case_audit(row: Mapping, *, schema: int, invalid_pointer_labels: bool) -> dict:
    case = dict(_mapping(row.get("case")))
    checks = [_mapping(value) for value in _list(row.get("checks"))]
    graphs = [_mapping(value) for value in _list(row.get("poisoned_graph_checks"))]
    resources = _mapping(row.get("resources"))
    statics = _mapping(row.get("static"))
    contracts = _mapping(row.get("verified_compile_contracts"))
    timings = [_mapping(value) for value in _list(row.get("timings"))]
    methods = set(resources) | set(statics)
    for timing in timings:
        methods.update(_mapping(timing.get("median_ms")))
    methods.update(check.get("method") for check in checks if isinstance(check.get("method"), str))
    methods = sorted(str(method) for method in methods)
    valid_pairs = [(check.get("method"), check.get("input")) for check in checks
                   if isinstance(check.get("method"), str) and isinstance(check.get("input"), str)]
    pairs = Counter(valid_pairs)
    missing_pairs = [[method, family] for method in methods for family in INPUT_FAMILIES
                     if pairs[(method, family)] == 0]
    duplicate_pairs = [[method, family, count] for (method, family), count in pairs.items()
                       if count > 1]
    native_checks = [check for check in checks if str(check.get("method", "")).startswith(("oa_", "ptxas_"))]
    cross_present = bool(native_checks) and all("baseline_cross_policy_bitwise" in check for check in native_checks)
    graph_methods = Counter(graph.get("method") for graph in graphs
                            if isinstance(graph.get("method"), str))
    graph_complete = bool(methods) and all(graph_methods[method] == 1 for method in methods)
    policy_rows = []
    for method in methods:
        if not method.startswith(("oa_", "ptxas_")):
            continue
        policy = method.split("_", 1)[1]
        contract = _mapping(contracts.get(policy))
        static = _mapping(statics.get(method))
        resource = _mapping(resources.get(method))
        invalid = invalid_pointer_labels and policy in INVALID_POINTER_POLICIES
        expected_pointer = "direct" if policy in INVALID_POINTER_POLICIES else "legacy"
        recorded_plan = _mapping(contract.get("requested_plan")) or contract
        pointer_match = (recorded_plan.get("pointer_mode") == expected_pointer
                         if contract else None)
        raw_drains = _list(static.get("post_hmma_nop_drains"))
        encoded = (static.get("hmma_stall_controls") == [8] and bool(raw_drains)
                   and all(_mapping(drain).get("stall") == 15
                           and _mapping(drain).get("yield") is True for drain in raw_drains))
        control_recorded = "hmma_stall_controls" in static and "post_hmma_nop_drains" in static
        binary_hash = static.get("cubin_sha256")
        contract_binary = _mapping(contract.get("cubin_sha256")).get(method.split("_", 1)[0])
        binary_match = binary_hash == contract_binary if _sha(binary_hash) and _sha(contract_binary) else None
        policy_rows.append({
            "method": method, "policy": policy,
            "label_status": "INVALID-direct-pointer-label" if invalid else "no_known_label_invalidation",
            "invalid_reason": ("Entry rename lost pointer-mode annotation; numerical pass validates "
                               "the executed legacy lowering, not the claimed direct-pointer policy.") if invalid else None,
            "compile_contract_recorded": bool(contract),
            "requested_pointer_mode_matches_record": pointer_match,
            "compile_contract_kind": ("manual-launch-experiment" if contract.get("manual_launch_experiment") is True
                                      else "checked-gemm-metadata" if contract else "not-recorded"),
            "encoded_control_recorded": control_recorded,
            "native_hmma8_drain15_record_consistent": (encoded if method.startswith("oa_") and control_recorded else None),
            "cubin_sha256": binary_hash, "sass_sha256": static.get("sass_sha256"),
            "cubin_hash_matches_compile_record": binary_match,
            "register_count_matches_driver_record": (static.get("registers") == resource.get("num_regs")
                if static.get("registers") is not None and resource.get("num_regs") is not None else None),
            "static": {key: static.get(key) for key in ("instructions", "non_nop", "registers", "hmma", "ldsm")},
            "driver_resources": dict(resource),
            "launch": _mapping(row.get("launches")).get(method)
                or contract.get("manual_launch") or contract.get("launch"),
        })
    declared_methods_invalid = [value["method"] for value in policy_rows
                                if value["label_status"] == "INVALID-direct-pointer-label"]
    counters = _mapping(row.get("counters"))
    counter_profiles = [{"label": value.get("label"), "status": value.get("status"),
                         "kernel_count": len(_list(value.get("kernels"))),
                         "missing_metrics": value.get("missing_metrics"),
                         "returncode": value.get("returncode")}
                        for value in map(_mapping, _list(counters.get("profiles")))]
    canonical = next((contract for contract in contracts.values()
                      if isinstance(contract, Mapping) and contract.get("output_dtype")), {})
    dtype = case.get("dtype")
    m, n, k = case.get("m"), case.get("n"), case.get("k")
    return {
        "exact_raw_case_key": case,
        "matrix_shape_key": {"role": case.get("role"), "m": m, "n": n, "k": k,
                             "input_dtype": dtype,
                             "A_shape_from_case": [m, k], "B_shape_from_case": [n, k],
                             "C_shape_from_case": [m, n],
                             "recorded_output_dtype": canonical.get("output_dtype"),
                             "recorded_accumulation_dtype": canonical.get("accumulation"),
                             "recorded_layout": canonical.get("input_layout"),
                             "scope": "Shape formulas use the role harness RCR ABI; absent precision/layout metadata stays null."},
        "methods": methods, "reported_policies": row.get("policies"),
        "check_counts": {"reported": len(checks), "passed": sum(check.get("passed") is True for check in checks),
                         "failed": sum(check.get("passed") is False for check in checks),
                         "missing_pass_flag": sum(type(check.get("passed")) is not bool for check in checks),
                         "invalid_method_input_key_records": len(checks) - len(valid_pairs),
                         "expected_from_recorded_methods_and_three_families": len(methods) * 3,
                         "all_reported_checks_passed": bool(checks) and all(check.get("passed") is True for check in checks),
                         "canary_flags_all_passed": bool(checks) and all(check.get("canaries") is True for check in checks),
                         "input_immutability_flags_all_passed": bool(checks) and all(check.get("inputs_unchanged") is True for check in checks),
                         "native_matched_ptxas_bitwise_all_passed": bool(native_checks) and all(check.get("ptxas_bitwise") is True for check in native_checks),
                         "cross_policy_bitwise_recorded": cross_present,
                         "native_cross_policy_bitwise_all_passed": all(check.get("baseline_cross_policy_bitwise") is True for check in native_checks) if cross_present else None,
                         "missing_method_input_pairs": missing_pairs, "duplicate_method_input_pairs": duplicate_pairs},
        "graph_counts": {"reported": len(graphs), "passed": sum(graph.get("passed") is True for graph in graphs),
                         "all_reported_checks_passed": bool(graphs) and all(graph.get("passed") is True for graph in graphs),
                         "one_record_per_timed_method": graph_complete,
                         "poisoned_output_flags_all_passed": bool(graphs) and all(graph.get("poisoned_output") is True for graph in graphs),
                         "scope": "Per-method records" if graphs else "Not recorded; methodology declaration alone is not a per-method record"},
        "policy_evidence": policy_rows, "invalid_method_labels": declared_methods_invalid,
        "counter_status": counters.get("status", "not-recorded"),
        "counter_profiles": counter_profiles,
        "timing_metadata": {"rounds_recorded": len(timings),
                            "methods_per_round": [sorted(_mapping(timing.get("median_ms"))) for timing in timings],
                            "paired_order_recorded_every_round": bool(timings) and all(bool(timing.get("paired_orders")) for timing in timings),
                            "library_medians_recorded_every_round": bool(timings) and all("cublas" in _mapping(timing.get("median_ms")) for timing in timings),
                            "scope": "Inventory only; no winner selection or cross-run/GPU aggregation"},
    }


def audit_file(path: Path) -> dict:
    record: dict[str, Any] = {"path": _path_name(path), "name": path.name,
                              "production_promoted": False}
    try:
        content = path.read_bytes()
        record.update(raw_sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        raw = json.loads(content)
        if not isinstance(raw, Mapping):
            raise ValueError("raw result root is not an object")
    except (OSError, ValueError, UnicodeError) as error:
        return {**record, "readable": False, "audit_blockers": [str(error)], "cases": []}
    schema = raw.get("schema", 1)
    invalid_labels = any(stamp in path.name for stamp in INVALID_POINTER_STAMPS)
    cases = [_case_audit(_mapping(row), schema=schema, invalid_pointer_labels=invalid_labels)
             for row in _list(raw.get("cases"))]
    blockers = []
    if raw.get("source_unchanged_during_run") is False:
        blockers.append("Compiler/harness source comparison changed during run; recorded numerical passes remain numerical passes.")
    if any(case["invalid_method_labels"] for case in cases):
        blockers.append("Known invalid direct-pointer labels must not support direct-pointer claims.")
    if any(not case["check_counts"]["all_reported_checks_passed"] for case in cases):
        blockers.append("At least one case has failed or missing numerical check flags.")
    if any(case["check_counts"]["missing_method_input_pairs"]
           or case["check_counts"]["duplicate_method_input_pairs"]
           or case["check_counts"]["invalid_method_input_key_records"] for case in cases):
        blockers.append("Method/input-family records are incomplete or duplicated.")
    if any(not case["check_counts"]["canary_flags_all_passed"]
           or not case["check_counts"]["input_immutability_flags_all_passed"]
           or not case["check_counts"]["native_matched_ptxas_bitwise_all_passed"] for case in cases):
        blockers.append("At least one explicit canary/input-immutability/matched-ptxas flag is missing or failed.")
    if schema in (2, 3) and any(not case["graph_counts"]["one_record_per_timed_method"]
                              or not case["graph_counts"]["all_reported_checks_passed"] for case in cases):
        blockers.append("Per-method poisoned graph records are incomplete or failed.")
    if schema == 3:
        if any(case["check_counts"]["native_cross_policy_bitwise_all_passed"] is not True for case in cases):
            blockers.append("Schema 3 cross-policy numerical records are missing or failed.")
        policies = [policy for case in cases for policy in case["policy_evidence"]]
        if any(policy["native_hmma8_drain15_record_consistent"] is not True
               for policy in policies if policy["method"].startswith("oa_")):
            blockers.append("Schema 3 native HMMA8/drain15 records are missing or inconsistent.")
        if any(policy["cubin_hash_matches_compile_record"] is not True for policy in policies):
            blockers.append("Schema 3 cubin hashes do not all match compiler records.")
        if any(policy["register_count_matches_driver_record"] is not True for policy in policies):
            blockers.append("Static/driver register records are missing or inconsistent.")
    scope = _source_scope(raw)
    if not scope["unchanged_flag_includes_known_harness_dependencies"]:
        blockers.append("No unchanged compiler-plus-known-harness/dependency file manifest is established by this record.")
    if schema == 1:
        blockers.append("Schema 1 lacks compile-policy verification and encoded-control records.")
    elif schema == 2:
        blockers.append("Schema 2 records compiler policy metadata but not encoded HMMA/drain controls or full experiment source scope.")
    elif schema != 3:
        blockers.append("Unknown raw schema; no schema-specific completeness claim is made.")
    return {
        **record, "readable": True, "schema": schema, "device": raw.get("device"),
        "runtime": raw.get("runtime"), "methodology": raw.get("methodology"),
        "source_scope": scope, "raw_promotion_blocker": raw.get("promotion_blocker"),
        "cases": cases, "case_count": len(cases),
        "reported_check_count": sum(case["check_counts"]["reported"] for case in cases),
        "all_reported_checks_passed": bool(cases) and all(case["check_counts"]["all_reported_checks_passed"] for case in cases),
        "audit_blockers": blockers,
        "audit_scope": "Raw-record inventory and consistency checks only; no independent mechanical proof of all harness assertions, no dispatch promotion.",
    }


def build_ledger(directory: Path = HERE) -> dict:
    paths = sorted(directory.glob("a100-role-gemm_*.json"))
    runs = [audit_file(path) for path in paths]
    return {"schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "raw_glob": "a100-role-gemm_*.json", "raw_run_count": len(paths),
            "expected_raw_paths_from_dynamic_scan": [_path_name(path) for path in paths],
            "raw_evidence_modified": False, "production_promoted": False,
            "scope": "Immutable raw-file inventory. Device cohorts and cases remain separate; check counts are bookkeeping, not independent trials or performance aggregates.",
            "runs": runs}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence-ledger.json")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((ROOT / "exp").resolve()):
        parser.error("audit output must remain under exp/")
    ledger = build_ledger()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2) + "\n")
    print(args.out)
    print(json.dumps({"raw_run_count": ledger["raw_run_count"],
                      "readable_runs": sum(run["readable"] for run in ledger["runs"]),
                      "all_numerical_flags_passed_runs": sum(run.get("all_reported_checks_passed") is True for run in ledger["runs"])}))


if __name__ == "__main__":
    main()
