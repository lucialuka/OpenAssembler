"""Freeze each successful selection before an independent held-out Modal app."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from openassembler.gemm_dispatch import (compiler_identity, contract_key, policy_plan,
                                         select_plan)
from openassembler.gemm_promotion import freeze_selection, _require
from openassembler.kernels.gemm import GemmPlan, generate


def freeze(directory: Path) -> dict:
    directory = Path(directory)
    output = directory / "frozen-selection.json"
    if output.exists():
        raise ValueError("frozen selection already exists; never overwrite an executed freeze")
    records = sorted((directory / "selection").glob("*/result.json"))
    if not records:
        raise ValueError("no successful selection records")
    entries = []
    for path in records:
        record = json.loads(path.read_text())
        plan = GemmPlan(**record["case"]["plan"])
        _, decision = select_plan(plan)
        incumbent = decision["policy"]
        # A stale global table correctly disables runtime dispatch, but a
        # previously integrated incumbent must still be compared during the
        # fresh A100 replay if its exact measured plan remains representable.
        table_path = Path("openassembler/kernels/sm80_gemm_dispatch.json")
        table_bytes = b""
        try:
            table_bytes = table_path.read_bytes()
            table = json.loads(table_bytes)
        except (OSError, ValueError):
            table = {}
        old = [entry for entry in table.get("entries", [])
               if isinstance(entry, dict) and entry.get("status") == "promoted"
               and entry.get("gates_passed") is True
               and entry.get("contract") == contract_key(plan)]
        old_incumbent = None
        if len(old) == 1:
            try:
                expected = hashlib.sha256(generate(policy_plan(plan, old[0]["policy"])).encode()).hexdigest()
            except (ValueError, KeyError, TypeError):
                expected = None
            if (old[0].get("policy") in record["candidates"]
                    and expected == old[0].get("ptx_body_sha256")):
                old_incumbent = old[0]
        incumbent_match = ({"policy": old_incumbent["policy"],
                            "basis": "exact_generated_body_hash",
                            "prior_table_sha256": hashlib.sha256(table_bytes).hexdigest(),
                            "prior_compiler_identity": table.get("compiler_identity"),
                            "prior_evidence": old_incumbent.get("evidence", [])}
                           if old_incumbent else
                           {"policy": "baseline", "basis": "no_exact_current_generated_body_match",
                            "prior_table_sha256": hashlib.sha256(table_bytes).hexdigest() if table_bytes else None})
        if old_incumbent:
            incumbent = old_incumbent["policy"]
        else:
            incumbent = "baseline"
        protocol = path.parent / "frozen-protocol.json"
        if protocol.is_file():
            frozen_record = json.loads(protocol.read_text())
            _require(frozen_record.get("selection_record_sha256") == hashlib.sha256(path.read_bytes()).hexdigest(),
                     "existing frozen protocol belongs to changed selection data")
            _require(frozen_record.get("selected_policy") == record["selected"]
                     and frozen_record.get("incumbent_policy") == incumbent,
                     "existing frozen policy/incumbent differs from audited selection")
            _require(frozen_record.get("incumbent_match") == incumbent_match,
                     "existing frozen incumbent basis differs from the current policy audit")
        else:
            freeze_selection(path, protocol, incumbent_policy=incumbent)
            frozen_record = json.loads(protocol.read_text())
            frozen_record["incumbent_match"] = incumbent_match
            protocol.write_text(json.dumps(frozen_record, indent=2) + "\n")
        entries.append({"id": record["case"]["id"], "selection_record": path.relative_to(directory).as_posix(),
                        "selection_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "protocol": protocol.relative_to(directory).as_posix(),
                        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        "selected_policy": record["selected"], "incumbent_policy": incumbent,
                        "incumbent_match": incumbent_match})
    result = {"schema": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "compiler_identity": compiler_identity(), "cases": entries,
              "device_policy": "independent_a100_cohorts",
              "scope": "per-contract frozen choice; no absolute runtime pooling between workers"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    value = freeze(args.directory)
    print(json.dumps({"frozen_cases": len(value["cases"]), "compiler_identity": value["compiler_identity"]}))
