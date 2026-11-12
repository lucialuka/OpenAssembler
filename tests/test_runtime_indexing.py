"""Runtime integer indexing: exact arithmetic and asynchronous conversion readers.

The arithmetic tests run offline. Real A100 correctness, large-grid repeats,
canaries and same-PTX ptxas controls live in exp/native_indexing/run_modal.py.
An emulator has no operand-collection timing and cannot replace that gate.
"""
from __future__ import annotations

import struct

import pytest

from openassembler.backend.compile import Out, schedule
from openassembler.backend.fragments import library
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def test_f2i_source_overwrite_waits_for_collection_separately_from_result():
    # The historical safe remainder emitted exactly this hazard: R5 became zero
    # while F2I was still collecting it, producing a zero reciprocal estimate.
    code = schedule([Out("F2I.FTZ.U32.TRUNC.NTZ R7, R5", True, [5], [7]),
                     Out("MOV R5, 0x0", False, [], [5]),
                     Out("IMAD R6, R6, R7, RZ", False, [6, 7], [6])])
    convert, overwrite, consume = [ctl for text, ctl in code]
    assert convert.rbar != 7 and convert.wbar != 7
    assert convert.rbar != convert.wbar
    assert overwrite.wait & (1 << convert.rbar)
    assert consume.wait & (1 << convert.wbar)


@pytest.mark.parametrize("level", ["safe", "mined", "sched", "opt"])
@pytest.mark.parametrize("op,signed", [("div", False), ("rem", False),
                                      ("div", True), ("rem", True)])
def test_runtime_integer_fragment_exact_edges_at_every_optimization_level(level, op, signed):
    key = f"{op}.{'s32' if signed else 'u32'}"
    fragment = library()[key]
    source = fragment["ptx"]
    name = fragment["kernel"]
    code, labels, words, exits, regs = generate(parse_module(source).kernels[name], "sm_80", level)
    assert any(text.startswith("F2I") and ctl.rbar != 7 for text, ctl in code)
    edges = [0, 1, 2, 511, 512, 513, 29312, 0x7fffffff,
             0x80000000, 0xfffffffe, 0xffffffff]
    divisors = [1, 2, 3, 7, 22, 127, 512, 513, 1009, 0x7fffffff]
    if signed:
        divisors += [0xffffffff, 0xfffffffd, 0x80000000]
    pairs = [(x, d) for x in edges for d in divisors
             if not (signed and x == 0x80000000 and d == 0xffffffff)]
    data = b"".join(struct.pack("<4I", x, d, 0x13579bdf, 0x2468ace0) for x, d in pairs)
    inp, out = 0x100000, 0x200000
    program = load_listing(listing(code, labels))
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(inp, data)
        params = struct.pack("<QQII", out, inp, 0, 0)
        Machine(program, Launch(block=(len(pairs), 1, 1), params=params, poison=poison), memory).run()
        for i, (x, d) in enumerate(pairs):
            if signed:
                x = x if x < 0x80000000 else x - 0x100000000
                d = d if d < 0x80000000 else d - 0x100000000
            quotient = abs(x) // abs(d)
            if (x < 0) != (d < 0):
                quotient = -quotient
            expected = quotient if op == "div" else x - quotient * d
            assert memory.read(out + i * 16, 4) == expected & 0xffffffff, (key, level, x, d)
        assert bytes(memory.data.get(inp + i, 0) for i in range(len(data))) == data
