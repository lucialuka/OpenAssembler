"""Naive PTX -> SASS baseline with fixed conservative scheduling (sm_80).

Instruction selection and output records are shared with backend.compile.
This module keeps its original scheduler for explicit A/B comparisons; it
is intentionally separate from the production measured-latency scheduler.
"""

from __future__ import annotations

from openassembler.backend.compile import (Compiler as Compiler, NotSupported as NotSupported,
                                           Out as Out, PARAM_BASE as PARAM_BASE, SREG as SREG)
from openassembler.isa.word import Control


# ---- scheduling: conservative control bits ---------------------------------------------

def schedule(code: list[Out]) -> list[tuple[str, Control]]:
    """In-order, no reordering. Fixed-latency ALU results: stall so the next reader
    is >= 6 cycles later (above every fixed latency mined from ptxas, max 5-6 for
    IMAD.WIDE/LEA). Variable-latency ops get a write scoreboard; readers wait on it."""
    out = []
    pending_sb: dict[int, int] = {}  # reg -> scoreboard
    next_sb = 0
    for k, o in enumerate(code):
        wait = 0
        for r in o.reads + o.writes:
            if r in pending_sb:
                wait |= 1 << pending_sb.pop(r)
        # clear all regs tied to waited scoreboards
        for r in [r for r, b in pending_sb.items() if wait >> b & 1]:
            pending_sb.pop(r)
        wbar = 7
        if o.var_lat:
            wbar = next_sb
            next_sb = (next_sb + 1) % 6
            for r in o.writes:
                pending_sb[r] = wbar
        stall = 6 if not o.var_lat else 2
        if o.text.startswith(("STG", "EXIT")):
            stall = 5
        out.append((o.text, Control(stall=stall, yield_=False, wbar=wbar, rbar=7, wait=wait, reuse=0)))
    return out
