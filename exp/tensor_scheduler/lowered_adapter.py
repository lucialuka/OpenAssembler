"""Fail-closed adapter from lowered OA instructions to the tensor graph model.

This is an experiment boundary only. It does not replace
`openassembler.backend.tensor.tensor_schedule`, encode SASS or mutate physical
register assignments. Labels and control-flow instructions are rejected so a
caller cannot accidentally move across a basic-block boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from dependency_scheduler import TensorInstruction, TensorLatency, build_dependency_graph, schedule


@dataclass(frozen=True)
class LoweredInstruction:
    text: str
    var_lat: bool
    reads: tuple[int, ...]
    writes: tuple[int, ...]


def _opname(text: str) -> str:
    token = text.split()[1] if text.startswith("@") else text.split()[0]
    return token.split(".")[0]


def _kind(name: str) -> str:
    if name == "HMMA":
        return "hmma"
    if name == "LDSM":
        return "ldsm"
    if name in {"LDGSTS", "LDGDEPBAR"}:
        return "async_copy"
    if name == "DEPBAR":
        return "wait_group"
    if name in {"STG", "STS", "BAR", "WARPSYNC", "BRA", "EXIT", "RET"}:
        return "store" if name in {"STG", "STS"} else "control"
    return "scalar"


def to_tensor_instructions(body: Sequence[LoweredInstruction]) -> list[TensorInstruction]:
    """Convert one basic block of lowered instructions into graph nodes."""

    rows: list[TensorInstruction] = []
    for index, op in enumerate(body):
        name = _opname(op.text)
        if name in {"BRA", "EXIT", "RET", "BSSY", "BSYNC", "CALL", "JMP"}:
            raise ValueError("control flow must be scheduled as a separate basic block")
        kind = _kind(name)
        reads = tuple(f"R{r}" for r in op.reads)
        writes = tuple(f"R{r}" for r in op.writes)
        accumulator = ":".join(writes) if kind == "hmma" else None
        rows.append(TensorInstruction(
            name=op.text,
            reads=reads,
            writes=writes,
            kind=kind,
            accumulator_group=accumulator,
            operands=reads,
            source_index=index,
        ))
    return rows


def schedule_lowered_block(
    body: Sequence[LoweredInstruction],
    latency: TensorLatency | None = None,
) -> list[int]:
    """Return a candidate source-index order for one lowered basic block."""

    nodes = to_tensor_instructions(body)
    graph = build_dependency_graph(nodes, latency)
    return [item.instruction.source_index for item in schedule(graph, latency)]

