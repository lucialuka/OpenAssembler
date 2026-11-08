"""Experiment-only load/FMA overlap templates for backward row reductions.

The existing FMA experiment contracts local products after all row chunks have
been loaded. This candidate keeps the same mathematical reduction tree and
warp synchronization, but interleaves each chunk's loads with its local FMA
accumulation. It is intended to create eligible arithmetic while later global
loads are still outstanding.

No production backend is changed. The generated PTX is compiled through both
OpenAssembler and ptxas by the usual experiment harness before any timing claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from exp.nn_round2.operator_suite import ROWWISE, rowwise_source


@dataclass(frozen=True)
class OverlapPlan:
    op: str
    width: int
    warps: int

    def __post_init__(self) -> None:
        if self.op not in ("softmax_backward", "rmsnorm_backward"):
            raise ValueError("overlap templates currently cover two backward reductions")
        if self.width not in (512, 513, 1024):
            raise ValueError("held-out reduction widths are 512, 513 and 1024")
        if self.warps not in (4, 8):
            raise ValueError("warps must be 4 or 8")


_LOAD = re.compile(r"(?:@%p1 )?ld\.global\.f32 %f(?P<reg>\d+),\[%d(?P<ptr>\d+)\+(?P<off>\d+)\];")


def _load_groups(lines: list[str], start: int, end: int, nin: int, chunks: int) -> list[list[list[str]]]:
    """Return [chunk][input] source lines, retaining tail neutral initialization."""

    groups: list[list[list[str]]] = [[[] for _ in range(nin)] for _ in range(chunks)]
    pending: list[str] = []
    for line in lines[start:end]:
        match = _LOAD.search(line)
        if not match:
            pending.append(line)
            continue
        reg = int(match["reg"])
        ptr = int(match["ptr"])
        k = 0 if ptr == 8 else 1 if ptr == 9 else None
        if k is None:
            raise ValueError(f"unexpected input pointer in load: {line}")
        j = reg - (32 * k)
        if not 0 <= j < chunks:
            raise ValueError(f"unexpected fragment register in load: {line}")
        groups[j][k].extend(pending)
        groups[j][k].append(line)
        pending = []
    if any(pending):
        raise ValueError(f"unattached load prologue lines: {pending}")
    if any(not entry for chunk in groups for entry in chunk):
        raise ValueError("incomplete per-chunk input load groups")
    return groups


def _local_accumulation(op: str, j: int) -> list[str]:
    if op == "rmsnorm_backward":
        if j == 0:
            # Use an explicit zero addend so the cleanup pass cannot mistake
            # these initializations for the old per-chunk product chain.
            return ["fma.rn.f32 %f64,%f0,%f0,0f00000000;",
                    "fma.rn.f32 %f96,%f0,%f32,0f00000000;"]
        return [
            f"fma.rn.f32 %f64,%f{j},%f{j},%f64;",
            f"fma.rn.f32 %f96,%f{j},%f{32+j},%f96;",
        ]
    if j == 0:
        return ["fma.rn.f32 %f64,%f0,%f32,0f00000000;"]
    return [f"fma.rn.f32 %f64,%f{j},%f{32+j},%f64;"]


def _remove_old_local_products(lines: list[str], op: str, chunks: int) -> list[str]:
    """Remove old per-chunk products and serial scalar accumulation adds."""

    removed = set()
    for j in range(chunks):
        removed.add(f"mul.f32 %f{64+j},%f{j},%f{j};" if op == "rmsnorm_backward"
                    else f"mul.f32 %f{64+j},%f{j},%f{32+j};")
        if op == "rmsnorm_backward":
            removed.add(f"mul.f32 %f{96+j},%f{j},%f{32+j};")
    out = []
    for line in lines:
        if line in removed:
            continue
        if op == "rmsnorm_backward":
            if (line.startswith("add.f32 %f161,%f161,%f")
                    and any(line.endswith(f"%f{64 + j};") for j in range(chunks))):
                continue
            if (line.startswith("add.f32 %f164,%f164,%f")
                    and any(line.endswith(f"%f{96 + j};") for j in range(chunks))):
                continue
        else:
            if (line.startswith("add.f32 %f160,%f160,%f")
                    and any(line.endswith(f"%f{64 + j};") for j in range(chunks))):
                continue
        out.append(line)
    return out


def generate(plan: OverlapPlan) -> tuple[str, dict]:
    """Generate a load/FMA-overlap source and provenance metadata."""

    name, original = rowwise_source(plan.op, plan.width, plan.warps)
    lines = original.splitlines()
    load_indices = [i for i, line in enumerate(lines) if "ld.global.f32" in line]
    if not load_indices:
        raise ValueError("source has no global loads")
    start, end = min(load_indices), max(load_indices) + 1
    chunks = (plan.width + 31) // 32
    groups = _load_groups(lines, start, end, ROWWISE[plan.op][0], chunks)
    replacement: list[str] = []
    for j, chunk in enumerate(groups):
        for source_lines in chunk:
            replacement.extend(source_lines)
        replacement.extend(_local_accumulation(plan.op, j))

    transformed = lines[:start] + replacement + lines[end:]
    transformed = _remove_old_local_products(transformed, plan.op, chunks)
    source = "\n".join(transformed) + "\n"
    metadata = {
        "kernel": name + "_overlap",
        "op": plan.op,
        "width": plan.width,
        "warps": plan.warps,
        "chunks": chunks,
        "candidate": "streamed_load_fma",
        "preserved": ["warp shuffle offsets", "full-lane mask", "tail guards", "stores"],
        "local_accumulators": ["%f64"] + (["%f96"] if plan.op == "rmsnorm_backward" else []),
        "rounding_policy": "explicit FMA for local products",
    }
    return source.replace(".entry " + name + "(", ".entry " + name + "_overlap(", 1), metadata


def all_plans() -> list[OverlapPlan]:
    return [OverlapPlan(op, width, warps)
            for op in ("softmax_backward", "rmsnorm_backward")
            for width in (512, 513, 1024)
            for warps in (4, 8)]


if __name__ == "__main__":
    for plan in all_plans():
        source, metadata = generate(plan)
        print(metadata["kernel"], len(source.splitlines()))

