"""Apply the lowered tensor dependency model to real OA basic blocks.

The production tensor backend remains unchanged. This adapter selects a real
GEMM through the current OA selector/allocation path and audits regions at
control/synchronization boundaries. Unsupported predicate, memory, async and
observer state is rejected with a reason; preserving HMMA K-order alone does
not establish those regions' semantic safety. No permutation is encoded.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openassembler.backend import ra
from openassembler.backend.tensor import TensorSelector
from openassembler.kernels.gemm import GemmPlan, generate
from openassembler.ptx.parse import parse_module

try:
    from .lowered_adapter import LoweredInstruction, schedule_lowered_block
    from .latency_profiles import latency_profile
except ImportError:
    from lowered_adapter import LoweredInstruction, schedule_lowered_block
    from latency_profiles import latency_profile


@dataclass(frozen=True)
class BlockResult:
    block: int
    instructions: int
    hmma: int
    moved: int
    baseline_order: tuple[int, ...]
    candidate_order: tuple[int, ...]
    k_order_preserved: bool
    accepted: bool
    rejection_reason: str | None = None


def _op(text: str) -> str:
    token = text.split()[1] if text.startswith("@") else text.split()[0]
    return token.split(".")[0]


def _blocks(plan: GemmPlan) -> list[list[LoweredInstruction]]:
    kernel = parse_module(generate(plan)).kernels["gemm"]
    selector = TensorSelector(kernel)
    ops = ra.coalesce(ra.dce(selector.run()))
    phys = ra.allocate(ops, selector.r1)
    lowered = ra.lower(ops, phys)
    blocks: list[list[LoweredInstruction]] = []
    current: list[LoweredInstruction] = []
    for text, var_lat, reads, writes in lowered:
        current.append(LoweredInstruction(text, var_lat, tuple(reads), tuple(writes)))
        if _op(text) in {"BRA", "EXIT", "RET", "BAR", "DEPBAR", "LDGDEPBAR"}:
            blocks.append(current); current = []
    if current:
        blocks.append(current)
    return blocks


def analyze(plan: GemmPlan, *, profile: str = "toy_uncalibrated") -> dict:
    model = latency_profile(profile)
    results: list[BlockResult] = []
    for index, block in enumerate(_blocks(plan)):
        if not block:
            continue
        try:
            order = schedule_lowered_block(block, model.latency)
        except ValueError as exc:
            # A block with a terminator or unsupported control is deliberately
            # rejected by the fail-closed adapter.
            results.append(BlockResult(index, len(block),
                                       sum(x.text.startswith("HMMA") for x in block),
                                       0, tuple(range(len(block))), tuple(), False, False,
                                       str(exc)))
            continue
        moved = sum(a != b for a, b in zip(range(len(block)), order))
        hmma = sum(x.text.startswith("HMMA") for x in block)
        # Check that every physical accumulator group retains its source order
        # after list scheduling. This is the K-order RAW gate for real HMMA
        # blocks; the graph must not merely produce a shorter permutation.
        source_pos = {source: pos for pos, source in enumerate(order)}
        baseline_by_accum: dict[tuple[int, ...], list[int]] = {}
        candidate_by_accum: dict[tuple[int, ...], list[int]] = {}
        for source, ins in enumerate(block):
            if not ins.text.startswith("HMMA"):
                continue
            key = tuple(ins.writes)
            baseline_by_accum.setdefault(key, []).append(source)
            candidate_by_accum.setdefault(key, []).append(source_pos[source])
        k_order = all(values == sorted(values) for values in candidate_by_accum.values())
        results.append(BlockResult(index, len(block), hmma, moved,
                                   tuple(range(len(block))), tuple(order), k_order,
                                   k_order))
    return {"schema": 2, "plan": plan.metadata, "blocks": [asdict(row) for row in results],
            "latency_profile": model.record(),
            "evidence_scope": "static adapter audit, not executed or encoded schedules",
            "accepted_means": "model syntax/explicit-register envelope only; not hardware safety",
            "encoding_allowed": False,
            "summary": {"blocks": len(results),
                        "hmma_blocks": sum(row.hmma > 0 for row in results),
                        "moved_instructions": sum(row.moved for row in results),
                        "accepted_blocks": sum(row.accepted for row in results),
                        "production_promoted": False}}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("toy_uncalibrated", "a100_clocked_24_8"),
                        default="toy_uncalibrated")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("actual-block-schedule.json"))
    args = parser.parse_args()
    out = args.out
    rows = [analyze(GemmPlan(128, 128, k, dtype="f16", schedule="stagepair"), profile=args.profile)
            for k in (128, 256, 512)]
    out.write_text(json.dumps({"schema": 2, "experiments": rows}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

