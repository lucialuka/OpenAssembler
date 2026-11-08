"""Assemble the measured lowest-passing-control inventory.

This combines the retained scalar dependency sweeps with the new SM80 tensor
control probes. It reports measured coverage, not a claim that every SASS
opcode or every operand form has been swept.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def scalar_inventory() -> list[dict]:
    candidates = [
        ROOT / "exp/tensor_scheduler/control-sweep-a100_20261108.json",
        ROOT / "corpus/sm_80/gpu_a100_40gb_latsweep.json",
    ]
    path = next((p for p in candidates if p.exists()), candidates[-1])
    data = json.loads(path.read_text())
    grouped: dict[str, list[tuple[int, str]]] = {}
    for key, row in data["probes"].items():
        if "@p_lat_" not in key or not key.startswith("s"):
            continue
        prefix, name = key.split("@", 1)
        if prefix == "ptxas":
            continue
        try:
            stall = int(prefix[1:])
        except ValueError:
            continue
        grouped.setdefault(name, []).append((stall, row.get("status", "unknown")))
    rows = []
    for name, values in sorted(grouped.items()):
        values.sort()
        passing = [stall for stall, status in values if status == "match"]
        lowest = next((stall for stall in sorted(passing)
                       if all(status == "match" for s, status in values if s >= stall)), None)
        rows.append({"family": "scalar", "probe": name, "lowest_passing": lowest,
                     "sweep": [{"stall": s, "status": status} for s, status in values]})
    return rows


def tensor_inventory() -> list[dict]:
    path = sorted((ROOT / "exp/tensor_scheduler").glob("hardware-probes_*.json"))
    if not path:
        return []
    data = json.loads(path[-1].read_text())
    rows = []
    for case in data.get("cases", []):
        valid = [row["label"] for row in case.get("variants", [])
                 if row.get("disasm_ok") and row.get("output_match")]
        stalls = sorted(int(label[1:]) for label in valid if label.startswith("s"))
        lowest = next((stall for stall in stalls
                       if all(f"s{s}" in valid for s in range(stall, 12))), None)
        rows.append({"family": "tensor", "probe": case["name"],
                     "lowest_passing": lowest, "hmma_count": case.get("hmma_count"),
                     "ldsm_count": case.get("ldsm_count"), "valid_controls": valid})
    return rows


def main() -> None:
    rows = scalar_inventory() + tensor_inventory()
    # `ptx_probes.py` exposes a broader standalone opcode/form corpus. The
    # A100 replay below measures dependency pairs, not a control-field sweep
    # for every one of those standalone forms; record both counts explicitly.
    try:
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import ptx_probes
        standalone_forms = len(list(ptx_probes.probes("sm_80")))
    except Exception:
        standalone_forms = None
    out = ROOT / "exp/tensor_scheduler/control-inventory.json"
    out.write_text(json.dumps({"schema": 1, "coverage": {
        "scalar_pairs": sum(row["family"] == "scalar" for row in rows),
        "tensor_probes": sum(row["family"] == "tensor" for row in rows),
        "standalone_ptx_forms_available": standalone_forms,
        "scope": "A100 dependency-pair controls plus tensor probes; standalone PTX forms are inventory-only",
    }, "rows": rows}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

