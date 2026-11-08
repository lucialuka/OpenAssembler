"""Compile grouped-N source candidates through OA for static screening.

This is intentionally a cheap pre-hardware gate.  It proves the rewritten
mapping survives OA parsing/encoding and records the instruction/register
delta.  It does not claim a speed result; the next stage must launch the
candidate with the remapped grid on A100 and run the full output gates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openassembler.backend.tensor import TensorConfig, generate as encode
from openassembler.kernels.gemm import GemmPlan, generate as source_generate
from openassembler.ptx.parse import parse_module

try:
    from .cta_swizzle import rewrite_grouped_n_prologue
except ImportError:
    from cta_swizzle import rewrite_grouped_n_prologue


def compile_candidate(shape: tuple[int, int, int], dtype: str = "f16") -> dict:
    m, n, k = shape
    baseline_source = source_generate(GemmPlan(m, n, k, dtype=dtype))
    grouped_source, mapping = rewrite_grouped_n_prologue(
        baseline_source, m // 128, n // 128, 8)
    baseline = encode(parse_module(baseline_source).kernels["gemm"], "sm_80", "opt", config=TensorConfig())
    grouped = encode(parse_module(grouped_source).kernels["gemm"], "sm_80", "opt", config=TensorConfig())
    return {
        "shape": list(shape), "dtype": dtype, "mapping": mapping,
        "baseline": {"instructions": len(baseline[0]),
                      "hmma": sum(text.startswith("HMMA") for text, _ in baseline[0]),
                      "registers": baseline[4]},
        "grouped_n8": {"instructions": len(grouped[0]),
                        "hmma": sum(text.startswith("HMMA") for text, _ in grouped[0]),
                        "registers": grouped[4]},
        "status": "static-encodable; hardware timing pending",
    }


def main() -> None:
    rows = [compile_candidate(shape, dtype)
            for shape in ((1024, 4096, 1024), (16384, 4096, 4096), (4096, 4096, 4096))
            for dtype in ("f16", "bf16")]
    out = Path(__file__).with_name("grouped-n-static-compare.json")
    out.write_text(json.dumps({"schema": 1,
                               "created_utc": datetime.now(timezone.utc).isoformat(),
                               "rows": rows}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

