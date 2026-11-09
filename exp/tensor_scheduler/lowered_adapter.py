"""Fail-closed adapter from lowered OA instructions to the tensor graph model.

This is an experiment boundary only. It does not replace
`openassembler.backend.tensor.tensor_schedule`, encode SASS or mutate physical
register assignments. Labels and control-flow instructions are rejected so a
caller cannot accidentally move across a basic-block boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

try:
    from .dependency_scheduler import TensorInstruction, TensorLatency, build_dependency_graph, schedule
except ImportError:
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


def _checked_kind(op: LoweredInstruction, *, shared_visibility_known: bool) -> str:
    """Refuse effects that this adapter cannot describe with complete edges."""
    text = op.text.strip().removesuffix(";")
    if not text or any(not isinstance(reg, int) or not 0 <= reg < 255
                       for reg in op.reads + op.writes):
        raise ValueError("empty instruction or invalid physical register metadata")
    name = _opname(text)
    if text.startswith("<label") or name in {"BRA", "EXIT", "RET", "BSSY", "BSYNC", "CALL", "JMP"}:
        raise ValueError("control flow/labels must remain outside the candidate block")
    if text.startswith("@") or re.search(r"\b(?:P[0-6]|UP[0-6])\b", text):
        raise ValueError("predication or implicit predicate state is not modeled")
    if name in {"LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "WARPSYNC", "MEMBAR"}:
        raise ValueError("async/synchronization visibility tokens are not modeled")
    if name == "HMMA":
        match = re.fullmatch(r"HMMA\.16816\.F32(?:\.BF16)?\s+R(\d+),\s*R(\d+),\s*R(\d+),\s*R(\d+)", text)
        if match is None or op.var_lat:
            raise ValueError("unsupported HMMA form or latency metadata")
        dest, a, b, c = map(int, match.groups())
        if dest % 4 or a % 4 or b % 2 or c % 4 or max(dest + 3, a + 3, b + 1, c + 3) >= 255:
            raise ValueError("unaligned or out-of-range HMMA fragment group")
        expected_reads = set(range(a, a + 4)) | set(range(b, b + 2)) | set(range(c, c + 4))
        if set(op.writes) != set(range(dest, dest + 4)) or not expected_reads <= set(op.reads):
            raise ValueError("incomplete HMMA fragment/accumulator register metadata")
        return "hmma"
    if name == "LDSM":
        if not shared_visibility_known:
            raise ValueError("LDSM shared visibility requires explicit caller context")
        match = re.fullmatch(r"LDSM\.16\.M88\.4\s+R(\d+),\s*\[R(\d+)(?:[+-](?:0x[0-9a-fA-F]+|\d+))?\]", text)
        if match is None or not op.var_lat:
            raise ValueError("unsupported LDSM form or latency metadata")
        dest, address = map(int, match.groups())
        if set(op.writes) != set(range(dest, dest + 4)) or address not in op.reads:
            raise ValueError("incomplete LDSM address/fragment register metadata")
        return "ldsm"
    # Memory aliases, global memory ordering, special registers, uniform
    # registers, carries and predicate effects need a richer adapter first.
    if name not in {"MOV", "IADD3", "IMAD", "LOP3", "SHF", "FADD", "FMUL", "FFMA"}:
        raise ValueError(f"unsupported memory/control/scalar form: {name}")
    mnemonic = text.split()[0]
    if (op.var_lat or re.search(r"\b(?:UR\d+|URZ|SR_\w+)\b|c\[", text)
            or ".X" in mnemonic or ".WIDE" in mnemonic or "[" in text):
        raise ValueError("implicit state or wide/memory operands are not modeled")
    operands = text[len(mnemonic):].strip().split(",")
    if not operands or not re.fullmatch(r"R\d+", operands[0].strip()):
        raise ValueError("scalar destination must be one explicitly modeled register")
    dest = int(operands[0].strip()[1:])
    scalar_atom = r"(?:!?PT|-?R(?:\d+|Z)(?:\.reuse)?|[-+]?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|INF|NAN))"
    if any(not re.fullmatch(scalar_atom, operand.strip()) for operand in operands[1:]):
        raise ValueError("implicit or unsupported scalar operand state")
    sources = {int(reg) for reg in re.findall(r"\bR(\d+)\b", ",".join(operands[1:]))}
    if set(op.writes) != {dest} or not sources <= set(op.reads):
        raise ValueError("incomplete scalar register metadata")
    return "scalar"


def to_tensor_instructions(body: Sequence[LoweredInstruction], *,
                           shared_visibility_known: bool = False) -> list[TensorInstruction]:
    """Convert an explicitly bounded register-only window into model nodes.

    No accepted window is an encoding authorization.  Predicates, async-copy,
    synchronization, general memory and implicit special state fail closed.
    LDSM additionally requires caller-supplied visibility context, and its
    readiness/address-WAR cycle distances remain uncalibrated model inputs.
    """

    rows: list[TensorInstruction] = []
    tensor_results: set[int] = set()
    for index, op in enumerate(body):
        kind = _checked_kind(op, shared_visibility_known=shared_visibility_known)
        if kind == "scalar" and tensor_results.intersection(op.writes):
            raise ValueError("scalar overwrite of an unfinished tensor result is not modeled")
        if kind == "scalar" and tensor_results.intersection(op.reads):
            kind = "scalar_consumer"
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
        tensor_results.difference_update(op.writes)
        if kind == "hmma":
            tensor_results.update(op.writes)
    return rows


def schedule_lowered_block(
    body: Sequence[LoweredInstruction],
    latency: TensorLatency | None = None,
    *, shared_visibility_known: bool = False,
) -> list[int]:
    """Return a candidate source-index order for one lowered basic block."""

    nodes = to_tensor_instructions(body, shared_visibility_known=shared_visibility_known)
    graph = build_dependency_graph(nodes, latency)
    return [item.instruction.source_index for item in schedule(graph, latency)]

