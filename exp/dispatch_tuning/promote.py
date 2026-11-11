"""Audit every frozen contract and assemble a current exact-dispatch table.

Raw selection, freeze and held-out files are never rewritten. Per-case decisions
are saved immediately so a CPU audit interruption cannot discard earlier work.
The table is written only after all frozen contracts have independent evidence.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from openassembler.gemm_dispatch import compiler_identity
from openassembler.gemm_promotion import promote
from openassembler.project import write_atomic


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote_campaign(directory: Path, *, table_path: Path | None = None) -> dict:
    directory = Path(directory)
    frozen_path = directory / "frozen-selection.json"
    frozen = json.loads(frozen_path.read_text())
    identity = compiler_identity()
    if frozen["compiler_identity"] != identity:
        raise ValueError("current compiler differs from the frozen selection")
    expected = frozen["cases"]
    if len({row["id"] for row in expected}) != len(expected):
        raise ValueError("frozen contracts are not unique")
    destination = directory / "promotion"
    destination.mkdir(exist_ok=True)
    decisions = []
    missing = []
    for case in expected:
        selection = directory / case["selection_record"]
        protocol = directory / case["protocol"]
        heldout = directory / "heldout" / case["id"] / "result.json"
        if not heldout.is_file():
            missing.append(case["id"])
            continue
        inputs = {"compiler_identity": identity,
                  "selection_sha256": _digest(selection),
                  "heldout_sha256": _digest(heldout),
                  "protocol_sha256": _digest(protocol)}
        output = destination / f"{case['id']}.json"
        # Progress decisions are convenient for an interrupted reporting pass,
        # but they are not an authority for installation. Installing always
        # re-reads and audits every archive/native encoding through promote().
        if output.is_file() and table_path is None:
            saved = json.loads(output.read_text())
            if saved.get("audit_inputs") != inputs:
                raise ValueError(f"retained promotion inputs changed: {case['id']}")
            decision = saved
        else:
            decision = {**promote(selection, heldout, protocol), "case_id": case["id"],
                        "audit_inputs": inputs,
                        "audited_utc": datetime.now(timezone.utc).isoformat()}
            write_atomic(output, json.dumps(decision, indent=2, allow_nan=False) + "\n")
        decisions.append(decision)
        print(f"{len(decisions)}/{len(expected)} {case['id']}: {decision['status']} {decision['policy']}",
              flush=True)
    counts = dict(Counter(row["status"] for row in decisions))
    entries = [row["entry"] for row in decisions if row.get("production_promoted")]
    result = {"schema": 1, "campaign": directory.name, "compiler_identity": identity,
              "frozen_selection": {"path": str(frozen_path), "sha256": _digest(frozen_path)},
              "audited_utc": datetime.now(timezone.utc).isoformat(),
              "expected_contracts": len(expected), "audited_contracts": len(decisions),
              "missing_contracts": missing, "counts": counts,
              "policy_counts": dict(Counter(row["policy"] for row in decisions
                                            if row.get("production_promoted"))),
              "decisions": decisions,
              "comparison_scope": "Supplemental OA/ptxas/cuBLAS controls use each session's nominal selected policy; promotion uses the pre-frozen policy and all-candidate native timings. They are not interchangeable."}
    write_atomic(directory / "promotion-decisions.json",
                 json.dumps(result, indent=2, allow_nan=False) + "\n")
    if table_path is not None:
        if missing or len(decisions) != len(expected):
            raise ValueError("cannot install an incomplete campaign")
        table = {"schema": 1, "device_scope": "A100 SM80 measured independent cohorts; per-entry device/runtime metadata retained",
                 "compiler_identity": identity, "source": str(directory / "promotion-decisions.json"),
                 "evidence_policy": "Production-source native/PTXAS full outputs, input immutability, canaries, poisoned graphs, native reencoding, disassembly and resource gates; pre-frozen choice must win every paired round in two independent sessions.",
                 "table_regenerated_utc": datetime.now(timezone.utc).isoformat(),
                 "promotion_policy": {"minimum_baseline_speedup_each_round": 1.02,
                                      "maximum_incumbent_regression": 1.02,
                                      "rounds_per_session": 3, "minimum_samples_per_round": 8,
                                      "graph_launches": 16, "unknown_contract": "baseline"},
                 "campaign_counts": counts, "entries": entries}
        write_atomic(table_path, json.dumps(table, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    result = promote_campaign(args.directory, table_path=args.table)
    print(json.dumps({key: result[key] for key in ("counts", "policy_counts", "missing_contracts")}))
