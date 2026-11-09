"""Audit one exact A100 tile64 proposal from raw, frozen and replay evidence.

The result can select a further integration experiment only.  It cannot route
production work, admit another shape/device, or claim a ptxas/library win.
Original cubins are not retained by these raw JSONs: their hashes are checked
for consistency across manifests, listings and independent runs, while the
current hash-matched source is re-encoded independently for the spill proof.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from exp.shape_coverage.measured_dispatch import GATES, select_experiment


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECTION = HERE / "a100-role-gemm_20261109T112122.548949Z.json"
HELDOUT = HERE / "a100-role-gemm_20261109T113154.637725Z.json"
PROTOCOL = HERE / "tile64-heldout-protocol.json"
HASH = re.compile(r"[0-9a-f]{64}\Z")
STAMP = re.compile(r"a100-role-gemm_(\d{8}T\d{6}\.\d{6}Z)\.json\Z")
CASE = {"model_case": "small_fp32_context", "role": "attention_output_projection",
        "m": 128, "n": 512, "k": 512, "dtype": "f16", "batch": 1,
        "sequence": 128, "hidden": 512}
WORKLOAD = {
    "role": "attention_output_projection", "m": 128, "n": 512, "k": 512,
    "batch_count": 1, "dtype": "f16", "arch": "sm_80",
    "operand_layouts": ["contiguous_MK", "contiguous_NK", "contiguous_MN"],
    "alignment_bytes": [16, 16, 8], "tail_policy": "reject", "workspace_bytes": 0,
    "accumulation_dtype": "f32", "output_dtype": "f32", "alpha": 1.0, "beta": 0.0,
    "numerical_policy": "ordered_hmma_fp32", "timing_scope": "resident_gemm_no_packing",
}
POLICY = {"tile_mnk": [64, 64, 64], "warps": 4, "stages": 2,
          "cta_order": "m_fast", "fragment_schedule": "source",
          "pointer_mode": "legacy", "pointer_advance": "late"}
EXPECTED_LAUNCH = {"grid": [2, 8, 1], "threads": 128, "smem": 32768}
# These application IDs were observed in the parent agent's Modal launch tool
# outputs. They are external provenance, not fields recovered from the raw JSON
# and not container IDs. Raw-result timestamp/hash IDs are retained alongside.
MODAL_PROVENANCE = {
    SELECTION.name: {"app_id": "ap-sAk42AdbKAykC7A2JQLIK2",
                     "device": "NVIDIA A100-SXM4-40GB"},
    HELDOUT.name: {"app_id": "ap-KUh2VWIURNhMlrL0qSbSvl",
                   "device": "NVIDIA A100-SXM4-80GB"},
}


class EvidenceError(ValueError):
    """A concrete raw-evidence or provenance condition was not established."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def exact(left: Any, right: Any) -> bool:
    """Compare JSON contracts without accepting booleans as numeric values."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def read_evidence(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path, "data": json.loads(raw), "sha256": digest(raw),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def _timestamp(path: Path) -> datetime:
    match = STAMP.fullmatch(path.name)
    require(match is not None, "raw result filename lacks its UTC completion timestamp")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)


def _source_identity(selection: dict, heldout: dict, protocol: dict, root: Path) -> dict:
    identity = selection["data"].get("source_identity")
    require(isinstance(identity, dict) and identity == heldout["data"].get("source_identity"),
            "selection and held-out source identities differ")
    hashes = identity.get("sha256", {})
    require(isinstance(hashes, dict) and hashes and
            "exp/shape_coverage/small_tile_gemm.py" in hashes and
            "openassembler/backend/tensor.py" in hashes, "source manifest is incomplete")
    for name, expected in hashes.items():
        path = root / name
        require(bool(HASH.fullmatch(str(expected))) and path.is_file()
                and digest(path.read_bytes()) == expected, f"current source hash mismatch: {name}")
    require(identity.get("harness_sha256") == hashes.get("exp/shape_coverage/a100_role_gemm_modal.py"),
            "harness identity disagrees with the source manifest")
    require(all(hashes.get(name) == expected for name, expected in
                identity.get("experiment_sha256", {}).items()), "experiment identity disagrees with source manifest")
    frozen = protocol["data"].get("source_sha256", {})
    require(isinstance(frozen, dict) and frozen and
            all(hashes.get(name) == expected for name, expected in frozen.items()),
            "frozen protocol source hashes do not match both raw runs")
    for required in ("exp/shape_coverage/a100_role_gemm_modal.py",
                     "exp/shape_coverage/small_tile_gemm.py",
                     "exp/shape_coverage/gemm_policy.py"):
        require(required in frozen, f"frozen protocol omits critical source: {required}")
    return deepcopy(identity)


def _protocol(selection: dict, heldout: dict, protocol: dict) -> dict:
    frozen = protocol["data"]
    require(frozen.get("schema") == 1 and frozen.get("production_promoted") is False,
            "protocol schema/scope is invalid")
    require(frozen.get("selection_result") == f"exp/shape_coverage/{selection['path'].name}",
            "protocol references a different selection result")
    expected = {"role": CASE["role"], "m": 128, "n": 512, "k": 512, "dtype": "f16",
                "layout": "contiguous_RCR", "accumulation": "f32", "output": "f32",
                "alpha": 1, "beta": 0, "packing_included": False}
    require(exact(frozen.get("workload"), expected), "protocol exact workload mismatch")
    require(frozen.get("candidate") == "tile64" and frozen.get("baseline") == "oa_m_fast"
            and frozen.get("additional_oa_control") == "oa_pointer_direct_early",
            "protocol policy/control mismatch")
    require(set(frozen.get("required_controls", [])) == {"ptxas_m_fast", "ptxas_tile64", "cublas"},
            "protocol does not require matched tile64 ptxas and library controls")
    require(frozen.get("minimum_baseline_speedup_each_round") == 1.02,
            "protocol speedup threshold changed")
    require(set(frozen.get("required_gates", [])) == {
        "zero_integer_normal_full_output", "same_policy_and_baseline_bitwise", "canaries",
        "inputs_unchanged", "poisoned_graph", "native_manifest", "encoded_hmma8_drain15",
        "resources", "source_unchanged"}, "protocol numerical/resource gates changed")
    timestamp = datetime.fromisoformat(frozen["frozen_utc"])
    require(timestamp.utcoffset().total_seconds() == 0, "protocol timestamp must be UTC")
    first, repeat = _timestamp(selection["path"]), _timestamp(heldout["path"])
    require(first < timestamp < repeat, "selection/freeze/held-out UTC order is invalid")
    mtimes = [datetime.fromisoformat(value["mtime_utc"]) for value in (selection, protocol, heldout)]
    # Checkout/copy tools may reset or equalize these times. They can
    # corroborate the original local observation, but cannot be a portable
    # evidence gate. Embedded completion/freeze timestamps remain checked.
    mtime_supports_history = (mtimes[0] < mtimes[1] < mtimes[2]
                              and abs((mtimes[1] - timestamp).total_seconds()) < 1)
    require(selection["sha256"] != heldout["sha256"], "selection and repeat reuse the same raw artifact")
    return {"path": "exp/shape_coverage/" + protocol["path"].name,
            "sha256": protocol["sha256"], "frozen_utc": frozen["frozen_utc"],
            "source_hash_count": len(frozen["source_sha256"]),
            "file_mtimes_utc": dict(zip(("selection", "protocol", "heldout"),
                                        (value["mtime_utc"] for value in (selection, protocol, heldout)))),
            "mtime_order_supports_history": mtime_supports_history,
            "mtime_scope": "diagnostic only: filesystem times are not durable provenance and may reset after checkout/copy",
            "start_time_limit": "raw filenames mark completion; protocol-before-Modal-launch order was observed by parent tool calls, not embedded as a launch timestamp",
            "timestamp_scope": "embedded UTC and observed launch-order checks establish consistency of recorded provenance, not cryptographic attestation of chronology"}


def _native_reencode() -> dict:
    from collections import Counter
    from openassembler.backend.tensor import TensorConfig, generate
    from openassembler.container.cubin import words_to_bytes
    from openassembler.kernels.gemm import GemmPlan
    from openassembler.ptx.parse import parse_module
    from exp.shape_coverage.small_tile_gemm import source_for_tile64
    source, certificate = source_for_tile64(GemmPlan(128, 512, 512, dtype="f16"))
    code, _, words, _, registers = generate(parse_module(source).kernels["gemm"],
                                           "sm_80", "opt", config=TensorConfig())
    opcodes = Counter(text.split()[0].split(".")[0] for text, _ in code)
    require(not any(name in opcodes for name in ("LDL", "STL", "CALL", "JCAL")),
            "spill proof failed: local-memory or call instructions in re-encoded stream")
    require(".local" not in source and registers == 88, "spill proof/resource re-encoding mismatch")
    require(all(ctl.stall == 8 for text, ctl in code if text.startswith("HMMA")),
            "native HMMA controls changed")
    drains = [ctl for index, (text, ctl) in enumerate(code) if text == "NOP"
              and index and code[index - 1][0].startswith("HMMA")]
    require(len(drains) == 16 and all(ctl.stall == 15 and ctl.yield_ for ctl in drains),
            "native epoch-drain controls changed")
    return {"source_sha256": digest(source.encode()), "certificate": certificate,
            "registers": registers, "instructions": len(words),
            # Native encoding appends a self-branch then pads with NOP words.
            "non_nop": sum(not text.startswith("NOP") for text, _ in code) + 1,
            "hmma": opcodes["HMMA"], "ldsm": opcodes["LDSM"],
            "text_sha256": digest(words_to_bytes(words)), "opcodes": dict(opcodes),
            "ldl_count": opcodes["LDL"], "stl_count": opcodes["STL"],
            "spill_bytes_derived": 0,
            "spill_derivation": "current hash-matched native source re-encodes with no LDL/STL/calls; both hardware function reports also have local_bytes=0; no ptxas spill report is claimed"}


def _checks(row: dict) -> set[str]:
    methods = set(row.get("static", {})) | {"cublas"}
    require(methods == set(row.get("resources", {})) | {"cublas"}, "resource/static method sets differ")
    require({"oa_m_fast", "ptxas_m_fast", "oa_tile64", "ptxas_tile64", "cublas"} <= methods,
            "candidate/baseline/matched controls missing")
    checks = row.get("checks", [])
    require(len(checks) == len(methods) * 3, "full-output family coverage is incomplete")
    seen = set()
    for check in checks:
        key = (check.get("method"), check.get("input"))
        require(key not in seen and key[0] in methods and key[1] in {"zero", "integer", "normal"},
                "duplicate/unknown method or numerical family")
        seen.add(key)
        require(all(check.get(field) is True for field in
                    ("passed", "canaries", "inputs_unchanged")), "numerical/canary/input gate failed")
        rel = check.get("relative_l2_fp32")
        require(type(rel) in (int, float) and math.isfinite(rel) and 0 <= rel < 5e-5,
                "nonfinite/out-of-budget FP32 output error")
        if key[0] != "cublas":
            require(check.get("ptxas_bitwise") is True and check.get("baseline_cross_policy_bitwise") is True,
                    "matched-source or cross-policy bitwise gate failed")
    graphs = row.get("poisoned_graph_checks", [])
    require(len(graphs) == len(methods) and {value.get("method") for value in graphs} == methods,
            "poisoned-graph method coverage is incomplete")
    for check in graphs:
        rel = check.get("relative_l2_fp32")
        require(check.get("passed") is True and check.get("poisoned_output") is True
                and type(rel) in (int, float) and math.isfinite(rel) and 0 <= rel < 5e-5,
                "poisoned graph/finite-output gate failed")
    return methods


def _binaries(row: dict, proof: dict) -> None:
    from exp.shape_coverage.gemm_policy import policy_source, verify_compile_record
    from openassembler.kernels.gemm import GemmPlan
    methods = set(row["static"])
    contracts = row.get("verified_compile_contracts", {})
    require(methods == {f"{backend}_{policy}" for policy in contracts for backend in ("oa", "ptxas")},
            "binary methods do not match native compile manifests")
    for policy, record in contracts.items():
        plan = GemmPlan(**record["requested_plan"])
        source, certificate = policy_source(plan, policy)
        require(record.get("source_sha256") == digest(source.encode()), "candidate PTX source hash mismatch")
        require(record.get("fragment_certificate") == certificate, "source/layout certificate mismatch")
        require(record.get("checked_gemm_contract") is (policy != "tile64")
                and record.get("manual_launch_experiment") is (policy == "tile64"),
                "manual/checked launch contract flags mismatch")
        try:
            verify_compile_record(plan, policy, source,
                                  {"backend": "openassembler", "tensor": record["native_tensor_policy"],
                                   "gemm": record if record["checked_gemm_contract"] else None}, certificate)
        except (AssertionError, ValueError) as exc:
            raise EvidenceError(f"native manifest policy/certificate failed: {exc}") from exc
        require(record.get("requested_tensor_policy", {}).get("hmma_stall") == 8
                and record.get("requested_tensor_policy", {}).get("epoch_drain") == 15,
                "requested HMMA/drain policy mismatch")
        for backend in ("oa", "ptxas"):
            method = backend + "_" + policy
            stats = row["static"][method]
            require(all(HASH.fullmatch(str(stats.get(field, ""))) for field in
                        ("cubin_sha256", "sass_sha256")), "missing binary/disassembly hash")
            require(stats["cubin_sha256"] == record.get("cubin_sha256", {}).get(backend),
                    "cubin hash differs between compile manifest and static listing")
            require(stats["registers"] == row["resources"][method]["num_regs"],
                    "register count differs between cubin and hardware attributes")
            if backend == "oa":
                require(stats.get("hmma_stall_controls") == [8] and stats.get("post_hmma_nop_drains")
                        and all(value == {"stall": 15, "yield": True}
                                for value in stats["post_hmma_nop_drains"]),
                        "encoded HMMA8/drain15 gate failed")
    contract = contracts["tile64"]
    require(contract["fragment_certificate"] == proof["certificate"], "tile64 certificate differs from current re-encoding")
    require(exact(contract.get("manual_launch"), EXPECTED_LAUNCH)
            and exact(row["launches"].get("oa_tile64"), EXPECTED_LAUNCH)
            and exact(row["launches"].get("ptxas_tile64"), EXPECTED_LAUNCH),
            "tile64 launch/shared-memory budget mismatch")
    stats, attrs = row["static"]["oa_tile64"], row["resources"]["oa_tile64"]
    require(all(stats.get(key) == proof[key] for key in
                ("registers", "instructions", "non_nop", "hmma", "ldsm")),
            "recorded tile64 counts differ from current native re-encoding")
    require(all(type(attrs.get(field)) is int for field in
                ("num_regs", "local_bytes", "shared_bytes", "max_threads", "binary_version"))
            and attrs.get("num_regs") == 88 and attrs.get("local_bytes") == 0
            and attrs.get("shared_bytes") == 0 and attrs.get("max_threads", 0) >= 128
            and attrs.get("binary_version") == 80,
            "tile64 hardware register/local/static-shared resource gate failed")


def _timings(row: dict, methods: set[str]) -> tuple[list[dict], list[dict]]:
    normalized, extra = [], []
    require(len(row.get("timings", [])) >= 3, "fewer than three paired timing rounds")
    for index, timing in enumerate(row["timings"]):
        orders, samples, medians = timing.get("paired_orders", []), timing.get("samples_ms", {}), timing.get("median_ms", {})
        require(timing.get("round") == index and set(samples) == methods and set(medians) == methods,
                "timing round/method set mismatch")
        require(len(orders) >= 12 and len({tuple(order) for order in orders}) > 1
                and all(len(order) == len(methods) and set(order) == methods for order in orders),
                "paired/randomized timing order evidence is incomplete")
        for name in methods:
            values = samples[name]
            require(len(values) == len(orders) and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values),
                    "missing/nonfinite paired timing samples")
            require(medians[name] == statistics.median(values), "recorded timing median does not match samples")
        normalized.append({"samples": len(orders), "oa_baseline_ms": medians["oa_m_fast"],
                           "oa_candidate_ms": medians["oa_tile64"],
                           "ptxas_ms": medians["ptxas_tile64"], "library_ms": medians["cublas"]})
        pointer = {name: value for name, value in medians.items() if name.startswith("oa_pointer_")}
        best = min(pointer, key=pointer.get) if pointer else None
        extra.append({"round": index, "best_pointer_control": best,
                      "best_pointer_ms": pointer[best] if best else None,
                      "best_pointer_over_tile64_speedup": pointer[best] / medians["oa_tile64"] if best else None,
                      "scope": "within this raw run only; absolute times are not averaged across devices"})
    return normalized, extra


def _normalize_run(raw: dict, split: str, identity: dict, proof: dict) -> dict:
    data = raw["data"]
    external = MODAL_PROVENANCE.get(raw["path"].name)
    require(external is not None and data.get("device") == external["device"],
            "unknown or mismatched independently observed Modal/device provenance")
    require(data.get("schema") == 3 and data.get("source_unchanged_during_run") is True
            and "promotion_blocker" not in data, "raw schema/source stability gate failed")
    require(len(data.get("cases", [])) == 1 and exact(data["cases"][0].get("case"), CASE),
            "raw exact role/shape/case mismatch")
    row, methodology = data["cases"][0], data.get("methodology", {})
    require(all(methodology.get(field) is True for field in
                ("full_output", "canaries", "input_immutability", "poisoned_graph", "baseline_cross_policy_bitwise")),
            "full-output methodology gates are not recorded")
    require(methodology.get("cublas") == "GemmEx, FP16/BF16 inputs, FP32 output/compute, alpha1 beta0, disallow reduced precision reduction",
            "library arithmetic contract is not matched")
    methods = _checks(row)
    if split == "heldout":
        require("oa_pointer_direct_early" in methods, "frozen additional pointer control missing")
    _binaries(row, proof)
    rounds, extra = _timings(row, methods)
    return {
        "id": raw["path"].stem + ":" + raw["sha256"],
        "session_id": "modal-app:" + external["app_id"], "split": split,
        "frozen_before_measurement": split == "heldout",
        "workload": deepcopy(WORKLOAD), "policy": deepcopy(POLICY),
        "source_identity": deepcopy(identity), "gates": dict.fromkeys(GATES, True),
        "gate_derivation": "all recorded full-family/graph/method gates verified against the hash-matched harness; finite_output follows its passed gate plus finite recorded error, not retained output arrays",
        "input_families": ["zero", "integer", "normal"],
        "artifact": {"entry": "gemm", "cubin_sha256": row["static"]["oa_tile64"]["cubin_sha256"],
                     "sass_sha256": row["static"]["oa_tile64"]["sass_sha256"],
                     "source_sha256": row["verified_compile_contracts"]["tile64"]["source_sha256"],
                     "raw_result_path": "exp/shape_coverage/" + raw["path"].name,
                     "raw_result_sha256": raw["sha256"]},
        "resources": {"registers": 88, "shared_bytes": 32768, "local_bytes": 0,
                      "spill_bytes": proof["spill_bytes_derived"]},
        "resource_proof": {"spill_derivation": proof["spill_derivation"],
                           "native_text_sha256": proof["text_sha256"],
                           "ldl_count": proof["ldl_count"], "stl_count": proof["stl_count"],
                           "static_shared_bytes": 0, "launch_dynamic_shared_bytes": 32768},
        "comparators": {
            "same_source_ptxas": {"name": "ptxas_tile64 / " + data["runtime"]["ptxas"],
                                  "matched_contract": True, "full_output_passed": True,
                                  "cubin_sha256": row["static"]["ptxas_tile64"]["cubin_sha256"]},
            "library": {"name": "cuBLAS GemmEx FP16/FP32 alpha1 beta0, no reduced-precision reduction",
                        "matched_contract": True, "full_output_passed": True},
        },
        "timing": {"scope": WORKLOAD["timing_scope"], "paired": True,
                   "randomized_order": True, "rounds": rounds},
        "additional_controls": extra,
        "provenance": {"device": data["device"], "raw_completion_utc": _timestamp(raw["path"]).isoformat(),
                       "retained_file_mtime_utc": raw["mtime_utc"], "modal_app_id": external["app_id"],
                       "modal_id_source": "parent agent's observed Modal launch tool output; not embedded in original raw JSON",
                       "container_id": None, "independent_shape": False,
                       "heldout_kind": "predeclared independent same-shape replay on a different A100 variant"},
    }


def build_proposal(selection_path: Path = SELECTION, heldout_path: Path = HELDOUT,
                   protocol_path: Path = PROTOCOL, *, source_root: Path = ROOT) -> dict[str, Any]:
    selection, heldout, protocol = map(read_evidence, (selection_path, heldout_path, protocol_path))
    freeze = _protocol(selection, heldout, protocol)
    identity = _source_identity(selection, heldout, protocol, source_root)
    proof = _native_reencode()
    runs = [_normalize_run(selection, "selection", identity, proof),
            _normalize_run(heldout, "heldout", identity, proof)]
    require(runs[0]["session_id"] != runs[1]["session_id"], "independent sessions are not established")
    require(runs[0]["artifact"]["cubin_sha256"] == runs[1]["artifact"]["cubin_sha256"]
            and runs[0]["artifact"]["sass_sha256"] == runs[1]["artifact"]["sass_sha256"],
            "candidate binary/disassembly identity changed across independent workers")
    entry = {"schema": 1, "id": "output_projection_128x512x512_f16_tile64_integration_experiment",
             "workload": deepcopy(WORKLOAD), "policy": deepcopy(POLICY),
             "source_identity": deepcopy(identity), "runs": runs,
             "frozen_protocol": freeze, "production_promoted": False,
             "deployment_scope": "NVIDIA A100 only; another SM80 device or workload requires new evidence"}
    return {"schema": 1, "workload": deepcopy(WORKLOAD), "current_source": identity,
            "entries": [entry], "production_promoted": False,
            "importer_source_sha256": digest(Path(__file__).read_bytes()),
            "selection_scope": "offline choice for a further A100-only integration experiment; no kernel routing",
            "evidence_limits": [
                "same-shape independent replay is not a held-out-shape generalization",
                "timestamp checks establish recorded provenance consistency, not cryptographic attestation; current filesystem mtimes are diagnostic only",
                "original cubin/output arrays were not retained: binary hashes and passed gates are source-bound runner attestations; current native text was independently re-encoded",
                "spill_bytes0 is derived from no local-memory/call operations plus driver local_bytes0, not a ptxas spill report",
                "counter and whole-training-step gates are not part of this isolated GEMM proposal",
                "same-source tile64 ptxas and cuBLAS remain faster than OA tile64",
            ]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--heldout", type=Path, default=HELDOUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        proposal = build_proposal(args.selection, args.heldout, args.protocol)
        result = select_experiment(proposal["entries"], workload=proposal["workload"],
                                   current_source=proposal["current_source"])
        result["selection_scope"] = proposal["selection_scope"]
        result["evidence_limits"] = proposal["evidence_limits"]
        result["additional_controls"] = {run["split"]: run["additional_controls"]
                                         for run in proposal["entries"][0]["runs"]}
        proposal_path = args.out_dir / "tile64-dispatch-proposal.json"
        proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
        result["proposal_path"] = str(proposal_path.relative_to(ROOT)) if proposal_path.is_relative_to(ROOT) else str(proposal_path)
        result["proposal_sha256"] = digest(proposal_path.read_bytes())
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as exc:
        result = {"status": "fallback", "selected_experiment": None,
                  "audit_rejected": True, "reasons": [str(exc)], "production_promoted": False}
    decision_path = args.out_dir / "tile64-dispatch-decision.json"
    decision_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
