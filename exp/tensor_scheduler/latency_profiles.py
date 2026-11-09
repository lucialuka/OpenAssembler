"""Explicit provenance for partial A100 tensor-model calibration.

These profiles only parameterize a static experiment model.  Even the clocked
profile is not an encoded-control policy or a full pipeline latency table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

try:
    from .dependency_scheduler import TensorLatency
except ImportError:
    from dependency_scheduler import TensorLatency


HERE = Path(__file__).resolve().parent
CLOCKED_EVIDENCE = HERE / "clocked-loop-probes_20261109T064627.431181Z.json"


@dataclass(frozen=True)
class LatencyProfile:
    name: str
    latency: TensorLatency
    calibrated_fields: tuple[str, ...]
    uncalibrated_fields: tuple[str, ...]
    provenance: dict
    encoding_allowed: bool = False
    hardware_validated_schedule: bool = False

    def record(self) -> dict:
        return {**asdict(self), "cycle_predictions_are_hardware_measurements": False,
                "scope": "static model; partial calibration does not validate reordering"}


def latency_profile(name: str = "toy_uncalibrated", *, evidence_path: Path | None = None) -> LatencyProfile:
    fields = tuple(TensorLatency.__dataclass_fields__)
    if name == "toy_uncalibrated":
        return LatencyProfile(name, TensorLatency(), (), fields, {
            "kind": "synthetic hypothesis", "hardware_evidence": None,
            "note": "Legacy toy distances are not measured A100 instruction latencies.",
        })
    if name != "a100_clocked_24_8":
        raise ValueError("unknown tensor latency profile")
    path = evidence_path or CLOCKED_EVIDENCE
    raw = path.read_bytes()
    data = json.loads(raw)
    rows = {row["chains"]: row for row in data["clocked"] if row["kind"] == "hmma"}
    if data.get("device") != "A100" or set(rows) != {1, 2, 3}:
        raise ValueError("calibration requires the bounded A100 one/two/three-chain record")
    observed = {chains: float(row["cycles_per_hmma"]) for chains, row in rows.items()}
    if (not all(row.get("reps") == 128 for row in rows.values())
            or not all(value > 0 for value in observed.values())
            or abs(observed[1] - 24) > 0.5
            or abs(observed[2] - 12) > 0.5
            or abs(observed[3] - 8) > 0.5):
        raise ValueError("clocked evidence is absent, degenerate, or outside the bounded 24/8 fit")
    return LatencyProfile(
        name, TensorLatency(accum_raw=24, hmma_issue=8),
        ("accum_raw", "hmma_issue"),
        tuple(field for field in fields if field not in {"accum_raw", "hmma_issue"}),
        {"kind": "partial fit to clocked ptxas loop probes",
         "artifact": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
         "device": data["device"], "repetitions": 128,
         "observed_effective_cycles_per_hmma": {str(k): v for k, v in observed.items()},
         "fit": {"accum_raw": "one-chain effective cycles/HMMA rounded to24",
                 "hmma_issue": "three-chain effective cycles/HMMA rounded to8"},
         "limitations": [
             "24/8 are approximate model distances, not lowest safe SASS stall controls",
             "the bounded ptxas loop result does not validate an OA reordered kernel",
             "LDSM, async-copy visibility, address WAR, observers and synchronization remain uncalibrated",
             "loop/control overhead and operand arrangement are specific to the retained probes",
         ]},
    )
