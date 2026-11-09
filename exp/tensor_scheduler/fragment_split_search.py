"""Experiment-only placement search for the next-K fragment load.

The integrated load-next schedule has its load after N fragment 3.  This
module tests other placements without changing tensor control fields, copy
groups, synchronization, shared addresses, or each accumulator's K order.
It is source generation plus a structural certificate, not hardware proof.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
from typing import Iterable

from openassembler.kernels.gemm import GemmPlan

try:
    from .load_next_compute_current import FragmentEvent, candidate_source, fragment_events
except ImportError:
    from load_next_compute_current import FragmentEvent, candidate_source, fragment_events


def certify_events(events: Iterable[FragmentEvent], *, k_fragments: int = 4) -> dict:
    """Check full work, exact chain order, and no live-window overwrite.

    A loaded slot may be replaced only after all 32 accumulator groups have
    consumed its old contents.  Counting loads alone is insufficient: an
    overwritten slot can still appear in a historical ``loaded[k]`` map.
    """
    if k_fragments < 1:
        raise ValueError("k_fragments must be positive")
    expected = {(mi, ni) for mi in range(4) for ni in range(8)}
    slots: dict[int, int] = {}
    seen_loads: set[int] = set()
    consumed: dict[int, set[tuple[int, int]]] = {}
    last_k = {chain: -1 for chain in expected}
    rows = tuple(events)
    for event in rows:
        if event.k not in range(k_fragments) or event.slot not in (0, 1):
            raise ValueError("fragment index or slot outside the bounded contract")
        if event.kind == "load":
            if event.mi is not None or event.ni is not None or event.k in seen_loads:
                raise ValueError("duplicate or malformed load")
            previous = slots.get(event.slot)
            if previous is not None and consumed[previous] != expected:
                raise ValueError("load overwrites a fragment window before its last use")
            slots[event.slot] = event.k
            seen_loads.add(event.k)
            consumed[event.k] = set()
            continue
        if event.kind != "hmma" or (event.mi, event.ni) not in expected:
            raise ValueError("unknown event or accumulator group")
        if slots.get(event.slot) != event.k:
            raise ValueError("HMMA reads an unloaded or overwritten fragment window")
        chain = (event.mi, event.ni)
        if chain in consumed[event.k] or event.k != last_k[chain] + 1:
            raise ValueError("duplicate accumulator update or broken K-order RAW edge")
        consumed[event.k].add(chain)
        last_k[chain] = event.k
    if seen_loads != set(range(k_fragments)) or any(
        consumed.get(k) != expected for k in range(k_fragments)
    ):
        raise ValueError("schedule omits a fragment or accumulator update")
    payload = [asdict(event) for event in rows]
    return {
        "events": len(rows), "loads": len(seen_loads),
        "hmma": k_fragments * len(expected), "accumulator_chains": len(expected),
        "k_order_preserved": True, "live_window_overwrite": False,
        "full_work_preserved": True, "event_stream": payload,
        "hardware_validated": False,
    }


def source_for_split(plan: GemmPlan, *, split_n: int) -> tuple[str, dict]:
    """Return a bounded source candidate and its experiment certificate."""
    if (plan.schedule != "stagepair" or plan.fragment_schedule != "source"
            or plan.cta_order != "m_fast" or plan.pointer_mode != "legacy"
            or plan.pointer_advance != "late" or plan.epilogue != "none"):
        raise ValueError("split search requires baseline stagepair/m-fast/legacy/late/unfused")
    events = fragment_events(4, split_n=split_n)
    certificate = certify_events(events)
    source = candidate_source(plan, split_n=split_n)
    certificate.update({
        "candidate": "fragment_load_placement", "split_n": split_n,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "hmma_control": 8, "epoch_drain": 15,
        "copy_groups_and_barriers_changed": False,
        "production_promoted": False,
    })
    return source, certificate


def static_summary(plan: GemmPlan, *, split_n: int) -> dict:
    """Compile with OA's native encoder, retaining conservative controls.

    No CUDA library or GPU is needed.  These resource/instruction results
    establish source-inclusive compilation only, never execution correctness.
    """
    from openassembler.backend.tensor import generate as encode
    from openassembler.ptx.parse import parse_module
    source, certificate = source_for_split(plan, split_n=split_n)
    kernel = parse_module(source).kernels["gemm"]
    code, labels, words, exits, registers = encode(kernel, "sm_80", "safe")
    counts = Counter(text.split()[0].split(".")[0] for text, _ in code)
    return {
        "plan": plan.metadata, "certificate": certificate,
        "static": {"instructions": len(code), "encoded_words": len(words),
                   "registers": registers, "opcodes": dict(sorted(counts.items())),
                   "labels": len(labels), "exits": len(exits)},
        "evidence_scope": "native source-inclusive compile and structural checks only",
        "requires_a100": ["valid disassembly", "full outputs", "canaries",
                          "input immutability", "poisoned graph replay",
                          "same-source ptxas", "paired timing", "resource/counter checks"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = GemmPlan(args.m, args.n, args.k, dtype=args.dtype)
    rows = [static_summary(plan, split_n=split) for split in (2, 4, 6)]
    payload = json.dumps({"schema": 1, "experiments": rows}, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(args.out)
    else:
        print(payload)
