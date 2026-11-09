"""Rank experiment-only tensor schedule policies on synthetic dependency traces."""

from __future__ import annotations

import json
from pathlib import Path

from dependency_scheduler import TensorInstruction, search_candidates
from latency_profiles import latency_profile


def workload(chains: int, depth: int = 8, observer: str = "store") -> list[TensorInstruction]:
    rows: list[TensorInstruction] = []
    for step in range(depth):
        for chain in range(chains):
            group = chr(ord("A") + chain)
            previous = f"{group}{step - 1}" if step else None
            current = f"{group}{step}"
            rows.append(TensorInstruction(
                name=f"HMMA.{group}{step}",
                reads=(previous,) if previous else (),
                writes=(current,),
                kind="hmma",
                accumulator_group=group,
                source_index=len(rows),
            ))
    final = f"A{depth - 1}"
    kind = "store" if observer == "store" else "scalar_consumer"
    rows.append(TensorInstruction(observer.upper(), reads=(final,), kind=kind, source_index=len(rows)))
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("toy_uncalibrated", "a100_clocked_24_8"),
                        default="toy_uncalibrated")
    args = parser.parse_args()
    profile = latency_profile(args.profile)
    output = []
    for chains in (1, 2, 3):
        rows = search_candidates(workload(chains),
                                 None if args.profile == "toy_uncalibrated" else
                                 ((profile.name, profile.latency),))
        output.append({"chains": chains, "candidates": [
            {"name": row["name"], "order": row["order"], "metrics": row["metrics"].__dict__}
            for row in rows
        ]})
    print(json.dumps({"schema": 2, "latency_profile": profile.record(),
                      "evidence_scope": "static model predictions, not hardware measurements",
                      "experiments": output}, indent=2))


if __name__ == "__main__":
    main()

