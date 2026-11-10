"""Audit retained A100 tuning sessions before promoting an exact GEMM policy.

This module is deliberately offline: it neither launches a GPU nor changes the
dispatch table on import. Selection, freezing and held-out confirmation are
separate operations. Each timing ratio stays within its recorded GPU session;
absolute runtimes from different A100 variants are never pooled.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
from typing import Any
import zipfile

from openassembler.gemm_dispatch import (
    candidate_plans, compiler_identity, contract_key, policy_plan,
)
from openassembler.gemm_tune import choose_policy, validate_record, _verify_archive
from openassembler.kernels.gemm import GemmPlan, generate

HASH = re.compile(r"[0-9a-f]{64}\Z")


class PromotionError(ValueError):
    """One failed evidence or performance requirement; never a partial promotion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def _exact(left: Any, right: Any) -> bool:
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _read(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        result = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise PromotionError(f"cannot read {path}: {exc}") from None
    _require(isinstance(result, dict), f"record is not an object: {path}")
    return result, hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, label: str) -> datetime:
    _require(isinstance(value, str), f"{label} is not an aware ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PromotionError(f"invalid {label}") from None
    _require(timestamp.utcoffset() is not None, f"{label} lacks a timezone")
    return timestamp.astimezone(timezone.utc)


def _plan(record: dict) -> GemmPlan:
    candidates = record.get("candidates")
    baseline = candidates.get("baseline") if isinstance(candidates, dict) else None
    _require(isinstance(baseline, dict) and isinstance(baseline.get("plan"), dict),
             "record has no baseline plan")
    try:
        plan = GemmPlan(**baseline["plan"])
    except (TypeError, ValueError):
        raise PromotionError("record baseline plan is invalid") from None
    _require(_exact(record.get("contract"), contract_key(plan)),
             "exact workload/precision/layout contract mismatch")
    _require(plan.epilogue == "none", "promotion currently covers unfused GEMM only")
    return plan


def _session(record: dict, split: str) -> dict:
    session = record.get("session")
    _require(isinstance(session, dict), f"missing {split} session provenance")
    _require(isinstance(session.get("id"), str) and bool(session["id"])
             and session.get("split") == split,
             f"invalid {split} session id/split")
    for key in ("modal_app_id", "container_id"):
        _require(session.get(key) is None or isinstance(session.get(key), str),
                 f"invalid {split} {key}")
    return session


@lru_cache(maxsize=512)
def _native_proof(identity: str, source: str, pointer_mode: str,
                  pointer_advance: str) -> tuple[tuple[int, ...], int]:
    """Re-encode the current source without calling NVIDIA tools or a GPU."""
    from openassembler.backend.tensor import TensorConfig
    from openassembler.build import generate as native_generate
    from openassembler.ptx.parse import parse_module

    kernel = parse_module(source).kernels.get("gemm")
    _require(kernel is not None, "retained source does not contain gemm")
    _, _, words, _, registers = native_generate(
        kernel, "sm_80", "opt", tensor_config=TensorConfig(
            pointer_mode=pointer_mode, pointer_advance=pointer_advance))
    return tuple(words), registers


def audit_record(path: Path, *, split: str) -> dict:
    """Verify current source, full gates, resources, retained SASS and cubins.

    Hash agreement proves retained-file consistency, not that a claimed remote
    session occurred. The benchmark/session provenance is retained explicitly;
    current native text is independently re-encoded before promotion.
    """
    path = Path(path)
    record, raw_hash = _read(path)
    _require(type(record.get("schema")) is int and record["schema"] == 1,
             "unsupported tuning record schema")
    plan = _plan(record)
    identity = compiler_identity()
    _require(record.get("compiler_identity") == identity,
             "record belongs to a different current compiler")
    session = _session(record, split)
    completed = _timestamp(record.get("completed_utc"), f"{split} completed_utc")
    try:
        validate_record(plan, record)
        _verify_archive(path, record)
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
        raise PromotionError(f"{split} record validation failed: {exc}") from None
    expected = candidate_plans(plan)
    for name, candidate in expected.items():
        _require(_exact(record["candidates"][name].get("plan"), asdict(candidate)),
                 f"{split} exact candidate plan mismatch: {name}")
    device = record.get("device", {})
    for key in ("driver", "sm_count", "clock_khz", "mem_bytes"):
        _require(type(device.get(key)) is int and device[key] > 0,
                 f"{split} device metadata is missing/invalid: {key}")
    runtime = record.get("runtime")
    _require(isinstance(runtime, dict) and all(
        isinstance(runtime.get(key), str) and runtime[key]
        for key in ("torch", "cuda", "ptxas")), f"{split} runtime provenance is incomplete")
    _require(record.get("timing_scope") ==
             "sixteen-launch CUDA graphs, native kernels only; excludes build, transfers, packing, training",
             f"{split} timing scope differs from the production protocol")

    from openassembler import oracle
    from openassembler.container.cubin import regcount, text_words
    from openassembler.isa.word import control
    archive_path = path.parent / record["artifact_archive"]["path"]
    proofs = {}
    try:
        with zipfile.ZipFile(archive_path) as retained:
            expected_names = {f"{name}/{kind}.{extension}" for name in expected
                              for kind in ("native", "ptxas") for extension in ("cubin", "sass")}
            expected_names |= {f"{name}/gemm.ptx" for name in expected}
            _require(set(retained.namelist()) == expected_names,
                     f"{split} retained archive has missing or unexpected entries")
            for name, candidate in expected.items():
                proof = {}
                source = retained.read(f"{name}/gemm.ptx").decode()
                for kind in ("native", "ptxas"):
                    artifact = record["candidates"][name]["artifacts"][kind]
                    _require(all(HASH.fullmatch(str(artifact.get(key, ""))) for key in
                                 ("cubin_sha256", "sass_sha256")),
                             f"{split} artifact identities missing: {name}/{kind}")
                    cubin = retained.read(f"{name}/{kind}.cubin")
                    sass = retained.read(f"{name}/{kind}.sass").decode()
                    listing = oracle.parse_listing(sass)
                    words = text_words(cubin, "gemm")
                    _require(bool(listing) and len(listing) == len(words)
                             and all(ins.addr == 16 * index and ins.word == words[index]
                                     for index, ins in enumerate(listing)),
                             f"{split} SASS listing disagrees with cubin text: {name}/{kind}")
                    resources = artifact["resources"]
                    _require(regcount(cubin, "gemm") == resources["num_regs"],
                             f"{split} cubin/driver register count differs: {name}/{kind}")
                    _require(type(artifact.get("instructions")) is int
                             and artifact["instructions"] == len(listing),
                             f"{split} recorded instruction count differs: {name}/{kind}")
                    hmma = [ins for ins in listing if ins.text.startswith("HMMA")]
                    _require(bool(hmma), f"{split} no tensor work in {name}/{kind}")
                    if kind == "native":
                        _require(all(control(ins.word).stall == 8 for ins in hmma),
                                 f"{split} native HMMA issue control changed: {name}")
                        drains = [ins for index, ins in enumerate(listing) if index
                                  and ins.text.startswith("NOP") and listing[index - 1].text.startswith("HMMA")]
                        _require(bool(drains) and all(control(ins.word).stall == 15
                                 and control(ins.word).yield_ for ins in drains),
                                 f"{split} native observer drain changed: {name}")
                        _require(not any(ins.text.split()[0].split(".")[0]
                                         in {"LDL", "STL", "CALL", "JCAL"} for ins in listing),
                                 f"{split} native local-memory/call instruction: {name}")
                        current_words, current_registers = _native_proof(
                            identity, source, candidate.pointer_mode, candidate.pointer_advance)
                        _require(tuple(words) == current_words and resources["num_regs"] == current_registers,
                                 f"{split} retained native text differs from current encoding: {name}")
                    proof[kind] = {"registers": resources["num_regs"],
                                   "text_sha256": hashlib.sha256(b"".join(
                                       value.to_bytes(16, "little") for value in words)).hexdigest(),
                                   "instructions": len(words)}
                proofs[name] = proof
    except (OSError, UnicodeError, KeyError, TypeError, AttributeError, StopIteration,
            struct.error, zipfile.BadZipFile, IndexError) as exc:
        raise PromotionError(f"{split} binary audit failed: {exc}") from None
    return {"path": path, "sha256": raw_hash, "record": record, "plan": plan,
            "session": session, "completed": completed, "native_proof": proofs}


def freeze_selection(selection_path: Path, protocol_path: Path, *,
                     incumbent_policy: str = "baseline", now: datetime | None = None) -> dict:
    """Persist a source/hash-bound winner before running a held-out session."""
    audited = audit_record(Path(selection_path), split="selection")
    record, plan = audited["record"], audited["plan"]
    _require(incumbent_policy in record["candidates"], "incumbent is not in the measured candidate set")
    timestamp = now or datetime.now(timezone.utc)
    _require(timestamp.utcoffset() is not None and timestamp >= audited["completed"],
             "freeze precedes the completed selection record")
    protocol = {"schema": 1, "frozen_utc": timestamp.astimezone(timezone.utc).isoformat(),
                "compiler_identity": compiler_identity(), "contract": contract_key(plan),
                "selection_record_sha256": audited["sha256"],
                "selected_policy": choose_policy(record), "incumbent_policy": incumbent_policy,
                "session_id": audited["session"]["id"],
                "minimum_speedup": 1.02, "maximum_incumbent_regression": 1.02,
                "device_policy": "independent_a100_cohorts"}
    from openassembler.project import write_atomic
    _require(not Path(protocol_path).exists(), "frozen selection protocol already exists")
    write_atomic(Path(protocol_path), json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return protocol


def promote(selection_path: Path, heldout_path: Path, protocol_path: Path) -> dict:
    """Return one table entry/decision without mutating the production table.

    A later held-out winner cannot replace the frozen selection. A near tie
    keeps the incumbent; a losing or invalid candidate never gains promotion.
    ``measured-baseline`` entries deliberately record a no-win result.
    """
    protocol, protocol_hash = _read(Path(protocol_path))
    _require(type(protocol.get("schema")) is int and protocol["schema"] == 1,
             "invalid frozen protocol schema")
    selection = audit_record(Path(selection_path), split="selection")
    heldout = audit_record(Path(heldout_path), split="heldout")
    first, repeat = selection["record"], heldout["record"]
    identity = compiler_identity()
    _require(protocol.get("compiler_identity") == identity,
             "frozen protocol belongs to a different compiler")
    _require(_exact(first["contract"], repeat["contract"])
             and _exact(protocol.get("contract"), first["contract"]),
             "selection/freeze/held-out exact contracts differ")
    _require(protocol.get("selection_record_sha256") == selection["sha256"],
             "selection record changed after the freeze")
    _require(protocol.get("session_id") == selection["session"]["id"],
             "frozen selection session differs")
    _require(selection["sha256"] != heldout["sha256"] and
             selection["session"]["id"] != heldout["session"]["id"],
             "held-out evidence is not an independent session")
    app1, app2 = (value["session"].get("modal_app_id") for value in (selection, heldout))
    _require(not app1 or not app2 or app1 != app2, "held-out evidence reuses the selection Modal app")
    frozen = _timestamp(protocol.get("frozen_utc"), "frozen_utc")
    _require(selection["completed"] <= frozen < heldout["completed"],
             "selection/freeze/held-out timestamp order is invalid")
    _require(protocol.get("device_policy", "independent_a100_cohorts") == "independent_a100_cohorts",
             "unsupported device-cohort policy")
    minimum = protocol.get("minimum_speedup", 1.02)
    max_regression = protocol.get("maximum_incumbent_regression", 1.02)
    _require(type(minimum) in (int, float) and math.isfinite(minimum) and minimum >= 1.02,
             "frozen minimum speedup is weaker than 2 percent")
    _require(type(max_regression) in (int, float) and math.isfinite(max_regression)
             and 1 <= max_regression <= 1.02, "invalid incumbent regression allowance")
    selected = protocol.get("selected_policy")
    incumbent = protocol.get("incumbent_policy", "baseline")
    _require(isinstance(selected, str) and isinstance(incumbent, str)
             and selected in first["candidates"] and incumbent in first["candidates"],
             "frozen policy/incumbent is not a measured candidate")
    _require(selected == choose_policy(first), "frozen policy was not selected from the selection session")
    _require(set(first["candidates"]) == set(repeat["candidates"]), "held-out candidate set changed")

    rounds = []
    for split, audited in (("selection", selection), ("heldout", heldout)):
        for timing in audited["record"]["timings"]:
            values = {name: statistics.median(samples) for name, samples in timing["samples_ms"].items()}
            rounds.append({"split": split, "session_id": audited["session"]["id"],
                           "device": audited["record"]["device"]["name"], "round": timing["round"],
                           "baseline_over_selected": values["baseline"] / values[selected],
                           "selected_over_incumbent": values[selected] / values[incumbent],
                           "selected_ms": values[selected], "incumbent_ms": values[incumbent]})
    reasons = []
    if selected != "baseline" and any(row["baseline_over_selected"] < minimum for row in rounds):
        reasons.append("frozen candidate fails the minimum baseline speedup in a paired round")
    if any(row["selected_over_incumbent"] > max_regression for row in rounds):
        reasons.append("frozen candidate regresses the incumbent beyond the noise allowance")
    if selected != incumbent and incumbent != "baseline" and not all(
            row["selected_over_incumbent"] <= 1 / minimum for row in rounds):
        reasons.append("candidate does not establish a repeatable improvement over the incumbent")
    if selected == "baseline" and incumbent != "baseline":
        reasons.append("a baseline choice does not replace an existing promoted policy")
    if reasons:
        return {"schema": 1, "status": "not-promoted", "policy": incumbent,
                "frozen_policy": selected, "reasons": reasons, "rounds": rounds,
                "entry": None, "production_promoted": False}

    plan = policy_plan(selection["plan"], selected)
    evidence = []
    cohorts = []
    for split, audited in (("selection", selection), ("heldout", heldout)):
        record = audited["record"]
        artifact = record["candidates"][selected]
        evidence.append({"split": split, "raw_result": str(audited["path"]),
                         "raw_result_sha256": audited["sha256"], "session": audited["session"],
                         "completed_utc": record["completed_utc"],
                         "artifact_archive": record["artifact_archive"],
                         "native": artifact["artifacts"]["native"],
                         "ptxas": artifact["artifacts"]["ptxas"],
                         "native_reencoding": audited["native_proof"][selected]["native"],
                         "rounds": len(record["timings"]),
                         "samples_per_round": [len(row["paired_orders"]) for row in record["timings"]]})
        cohort = {**record["device"], "runtime": record["runtime"]}
        if not any(_exact(cohort, old) for old in cohorts):
            cohorts.append(cohort)
    entry = {"contract": contract_key(plan), "policy": selected, "status": "promoted",
             "gates_passed": True, "compiler_identity": identity,
             "ptx_body_sha256": hashlib.sha256(generate(plan).encode()).hexdigest(),
             "speedup_range": [min(row["baseline_over_selected"] for row in rounds),
                               max(row["baseline_over_selected"] for row in rounds)],
             "measurement_cohorts": cohorts, "evidence": evidence,
             "frozen_protocol": {"path": str(protocol_path), "sha256": protocol_hash,
                                 "frozen_utc": protocol["frozen_utc"], "selected_policy": selected,
                                 "incumbent_policy": incumbent},
             "selection_scope": "exact native GEMM contract; independent A100 cohorts checked separately; no absolute-time pooling"}
    return {"schema": 1, "status": "measured-baseline" if selected == "baseline" else "promoted",
            "policy": selected, "frozen_policy": selected, "reasons": [], "rounds": rounds,
            "entry": entry, "production_promoted": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = promote(args.selection, args.heldout, args.protocol)
    from openassembler.project import write_atomic
    write_atomic(args.out, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {result['policy']} -> {args.out}")


if __name__ == "__main__":
    main()
