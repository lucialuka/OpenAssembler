"""Shape/alignment/workspace manifest for experiment-only GEMM dispatch.

This is a record of candidate policy and evidence provenance.  It deliberately
does not alter the OpenAssembler CLI or silently select an unmeasured kernel.
The manifest can be fed to a later held-out runner once grouped-N and
load-next/compute-current candidates have real A100 timings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:  # package import for tests
    from .cta_swizzle import TileGrid, SwizzleSpec, candidate_specs, locality_metrics
except ImportError:  # direct script execution from the repository root
    from cta_swizzle import TileGrid, SwizzleSpec, candidate_specs, locality_metrics


@dataclass(frozen=True)
class DispatchCase:
    name: str
    shape: tuple[int, int, int]
    dtype: str
    input_alignment: int
    output_alignment: int
    workspace_bytes: int
    candidate: str
    status: str
    rationale: str
    evidence: tuple[str, ...] = ()


def _candidate_name(spec: SwizzleSpec) -> str:
    return spec.name if spec.width is None else f"{spec.name}{spec.width}"


def build_cases() -> list[DispatchCase]:
    """Create a bounded matrix covering known tall, wide and square shapes."""

    shapes = [
        ("tall", (16384, 4096, 4096), ("f16", "bf16")),
        ("wide", (1024, 4096, 1024), ("f16", "bf16")),
        ("square", (4096, 4096, 4096), ("f16", "bf16")),
    ]
    rows: list[DispatchCase] = []
    for label, shape, dtypes in shapes:
        grid = TileGrid.from_shape(shape[0], shape[1])
        for dtype in dtypes:
            for alignment in (16, 128):
                for workspace in (0, 1 << 20, 64 << 20):
                    for spec in candidate_specs(grid):
                        name = f"{label}_{dtype}_a{alignment}_w{workspace}_{_candidate_name(spec)}"
                        if spec.name == "m_fast":
                            status = "baseline"
                            rationale = "Current OA traversal; required control for paired timing."
                        elif spec.name == "morton8":
                            status = "validated-on-some-shapes"
                            rationale = "Existing OA Morton evidence changes L2/DRAM locality; regressions remain."
                        else:
                            status = "untested"
                            rationale = "Public CUTLASS grouped-N mapping; requires source-inclusive A100 validation."
                        rows.append(DispatchCase(
                            name=name, shape=shape, dtype=dtype,
                            input_alignment=alignment, output_alignment=8,
                            workspace_bytes=workspace, candidate=_candidate_name(spec),
                            status=status, rationale=rationale,
                            evidence=("exp/cutlass_cublas_research.md",) if spec.name == "grouped_n" else (),
                        ))
    return rows


def build_manifest() -> dict:
    cases = build_cases()
    locality = {}
    for shape_name, shape, _ in (
        ("tall", (16384, 4096, 4096), ()),
        ("wide", (1024, 4096, 1024), ()),
        ("square", (4096, 4096, 4096), ()),
    ):
        grid = TileGrid.from_shape(shape[0], shape[1])
        locality[shape_name] = {
            _candidate_name(spec): locality_metrics(grid, spec)
            for spec in candidate_specs(grid)
        }
    return {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "experiment-only; no candidate is a production dispatch rule",
        "contract": {
            "arch": "sm_80",
            "tile_mnk": [128, 128, 64],
            "input_alignment_bytes": [16, 128],
            "output_alignment_bytes": [8],
            "workspace_bytes": [0, 1 << 20, 64 << 20],
            "required_gates": ["full_output", "canary", "poisoned_input", "valid_disassembly", "resource_report", "held_out_timing"],
        },
        "cases": [asdict(case) for case in cases],
        "cpu_locality_diagnostics": locality,
        "promotion_policy": "A candidate remains untested until source-inclusive A100 timing beats the current OA policy on held-out shapes with all gates passing.",
    }


def write(path: Path) -> None:
    path.write_text(json.dumps(build_manifest(), indent=2) + "\n")


if __name__ == "__main__":
    target = Path(__file__).with_name("shape-alignment-workspace-manifest.json")
    write(target)
    print(target)

