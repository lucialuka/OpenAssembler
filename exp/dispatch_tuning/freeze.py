"""Freeze each successful selection before an independent held-out Modal app."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from openassembler.gemm_dispatch import compiler_identity, select_plan
from openassembler.gemm_promotion import freeze_selection
from openassembler.kernels.gemm import GemmPlan


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
        protocol = path.parent / "frozen-protocol.json"
        freeze_selection(path, protocol, incumbent_policy=incumbent)
        entries.append({"id": record["case"]["id"], "selection_record": path.relative_to(directory).as_posix(),
                        "selection_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "protocol": protocol.relative_to(directory).as_posix(),
                        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        "selected_policy": record["selected"], "incumbent_policy": incumbent})
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
