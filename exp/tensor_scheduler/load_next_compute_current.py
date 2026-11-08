"""Experiment-only CUTLASS-style fragment pipeline for SM80 GEMM.

The production OpenAssembler generator loads all A/B fragments for a K slice,
then emits the HMMA group.  This module keeps the production generator intact
and builds a candidate source that uses two register windows:

* compute the first half of the current K fragment;
* load the next K fragment into the other window;
* finish the current fragment;
* repeat while each accumulator's K order remains unchanged.

This is the ``load-next / compute-current`` pattern used by public CUTLASS
SM80 multistage mainloops.  It is an experiment boundary: callers must still
run ptxas/nvdisasm, full outputs, canaries, poisoned inputs, resource checks
and paired A100 timing before any dispatch rule can be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openassembler.kernels.gemm import GemmPlan, _compute, generate


@dataclass(frozen=True, slots=True)
class FragmentEvent:
    """One event in the candidate fragment schedule."""

    kind: str
    k: int
    slot: int
    mi: int | None = None
    ni: int | None = None


def _load_fragment(stage: str, k: int, slot: int) -> list[str]:
    """Load one 16-wide A/B fragment into an alternating register window."""

    a_base = 400 + slot * 64
    b_base = 500 + slot * 32
    # The address registers are deliberately shared with the production
    # generator.  The candidate changes only fragment register ownership and
    # event order; async-copy, barrier and shared addresses remain identical.
    lines: list[str] = []
    ik = k
    lines += [f"add.u32 %r41,%r{200 + ik * 4},{stage};",
              f"add.u32 %r42,%r{300 + ik * 8},{stage};",
              "add.u32 %r42,%r42,%r47;"]
    for mi in range(4):
        aa = ",".join(f"%r{a_base + mi * 4 + j}" for j in range(4))
        lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{aa}}},[%r41+{mi*2048}];")
    for ni in range(0, 8, 2):
        bb = ",".join(f"%r{b_base + ni * 2 + j}" for j in range(4))
        lines.append(f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bb}}},[%r42+{ni*1024}];")
    return lines


def _hmma_half(dtype: str, k: int, slot: int, nis: Iterable[int]) -> list[str]:
    """Emit half of a K fragment's HMMA work using ``slot`` registers."""

    a_base = 400 + slot * 64
    b_base = 500 + slot * 32
    lines: list[str] = []
    for ni in nis:
        bb = ",".join(f"%r{b_base + ni * 2 + j}" for j in range(2))
        for mi in range(4):
            aa = ",".join(f"%r{a_base + mi * 4 + j}" for j in range(4))
            cc = ",".join(f"%c{(mi*8+ni)*4+j}" for j in range(4))
            lines.append(
                f"mma.sync.aligned.m16n8k16.row.col.f32.{dtype}.{dtype}.f32 "
                f"{{{cc}}},{{{aa}}},{{{bb}}},{{{cc}}};"
            )
    return lines


def fragment_events(k_fragments: int = 4, *, split_n: int = 4) -> tuple[FragmentEvent, ...]:
    """Return the abstract load-next/compute-current event stream.

    ``split_n`` controls where the next fragment load is placed.  The default
    splits the eight N fragments into two groups.  Every accumulator chain is
    still ordered ``k=0,1,...``; only independent fragment loads are moved
    between those true RAW edges.
    """

    if k_fragments <= 0:
        raise ValueError("k_fragments must be positive")
    if split_n <= 0 or split_n >= 8:
        raise ValueError("split_n must be between 1 and 7")
    out: list[FragmentEvent] = [FragmentEvent("load", 0, 0)]
    for k in range(k_fragments):
        slot = k & 1
        for ni in range(split_n):
            for mi in range(4):
                out.append(FragmentEvent("hmma", k, slot, mi, ni))
        if k + 1 < k_fragments:
            out.append(FragmentEvent("load", k + 1, (k + 1) & 1))
        for ni in range(split_n, 8):
            for mi in range(4):
                out.append(FragmentEvent("hmma", k, slot, mi, ni))
    return tuple(out)


def validate_events(events: Iterable[FragmentEvent]) -> None:
    """Fail closed if a schedule violates fragment ownership or K order."""

    loaded: dict[int, int] = {}
    last_k: dict[tuple[int, int], int] = {}
    for event in events:
        if event.kind == "load":
            if event.k in loaded:
                raise ValueError(f"fragment k={event.k} loaded more than once")
            loaded[event.k] = event.slot
            continue
        if event.kind != "hmma" or event.mi is None or event.ni is None:
            raise ValueError("unknown fragment event")
        if event.k not in loaded or loaded[event.k] != event.slot:
            raise ValueError("HMMA consumes an unloaded or overwritten fragment slot")
        key = (event.mi, event.ni)
        previous = last_k.get(key, -1)
        if event.k != previous + 1:
            raise ValueError(f"accumulator {key} breaks K order at {event.k}")
        last_k[key] = event.k


def candidate_compute(dtype: str, stage: str, *, split_n: int = 4) -> list[str]:
    """Render one valid PTX compute block with alternating fragment windows."""

    if dtype not in ("f16", "bf16"):
        raise ValueError("dtype must be f16 or bf16")
    events = fragment_events(4, split_n=split_n)
    validate_events(events)
    lines: list[str] = ["shr.u32 %r47,%r3,4;", "mul.lo.u32 %r47,%r47,1024;"]
    loaded: set[int] = set()
    for event in events:
        if event.kind == "load":
            lines.extend(_load_fragment(stage, event.k, event.slot))
            loaded.add(event.k)
        elif event.mi == 0 and event.ni is not None:
            # The abstract stream records every accumulator chain, while the
            # renderer emits the four M rows together for each N fragment.
            # Register windows alternate by K slice and never overwrite a
            # fragment before its last use.
            lines.extend(_hmma_half(dtype, event.k, event.slot, (event.ni,)))
    return lines


def candidate_source(plan: GemmPlan, *, split_n: int = 4) -> str:
    """Transform a stage-pair GEMM source into the experiment candidate."""

    if not isinstance(plan, GemmPlan):
        raise TypeError("plan must be a GemmPlan")
    if plan.schedule != "stagepair":
        raise ValueError("the experiment currently targets the validated stagepair path")
    source = generate(plan)
    for stage in ("0", "32768"):
        old = "\n".join(_compute(plan.dtype, stage))
        new = "\n".join(candidate_compute(plan.dtype, stage, split_n=split_n))
        if old not in source:
            raise ValueError(f"baseline compute block for stage {stage} not found")
        source = source.replace(old, new)
    return source


def schedule_summary(k_fragments: int = 4, *, split_n: int = 4) -> dict:
    events = fragment_events(k_fragments, split_n=split_n)
    validate_events(events)
    return {
        "candidate": "load_next_compute_current",
        "k_fragments": k_fragments,
        "split_n": split_n,
        "events": len(events),
        "loads": sum(event.kind == "load" for event in events),
        "hmma": sum(event.kind == "hmma" for event in events),
        "accumulator_chains": 32,
        "register_windows": 2,
        "k_order_preserved": True,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(schedule_summary(), indent=2))
