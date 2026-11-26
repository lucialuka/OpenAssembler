"""Exact shared state and branch/snapshot regressions; no emulator ordering claim."""
from collections import Counter
import json
from pathlib import Path
import struct

import pytest

from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing
from tests.test_backend_atomics import connected_exchange

FIXTURES = json.loads((Path(__file__).parent / 'data/atomic_ordering.json').read_text())
LEVELS = ['safe', 'mined', 'sched', 'opt']


@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('case', FIXTURES['shared'], ids=lambda case: case['name'])
def test_shared_atomics_barriers_old_values_and_preserved_predicates(case, level):
    name = case['name']
    code, labels, words, _, _ = generate(parse_module(case['source']).kernels[name], 'sm_80', level)
    assert words
    address, size, guard = 0x100000000, 2320, b'\xa5' * 64
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(address - 64, guard + b'\x00' * size + guard)
        for cta in range(4):
            Machine(load_listing(listing(code, labels)),
                    Launch(grid=(4, 1, 1), block=(128, 1, 1), ctaid=(cta, 0, 0),
                           params=struct.pack('<Q', address), smem_static=68, poison=poison), memory).run()
        for cta in range(4):
            for tid in range(1, 128, 2):
                assert memory.read(address + 4 * (cta * 128 + tid), 4) == 0x13579bdf
            for bin_index in range(17):
                lanes = ([i for i in range(0, 128, 2)] if bin_index == 1 else []) if name.endswith('_symbol') else [i for i in range(bin_index, 128, 17) if i % 2 == 0]
                old_values = [memory.read(address + 4 * (cta * 128 + i), 4) for i in lanes]
                final = memory.read(address + 2048 + 4 * (cta * 17 + bin_index), 4)
                if 'add' in name:
                    initial = 0xffffffff if name == 'shared_add_s32' else 0
                    assert final == (initial + len(lanes)) & 0xffffffff
                    assert Counter(old_values) == Counter((initial + i) & 0xffffffff for i in range(len(lanes)))
                else:
                    assert connected_exchange(0, final, [cta * 128 + i + 1 for i in lanes], old_values)
        for offset in (-64, size):
            assert bytes(memory.read(address + offset + i, 1) for i in range(64)) == guard


@pytest.mark.parametrize('level', LEVELS)
@pytest.mark.parametrize('fixture,entry,width', [('pointer', 'pointer_branch', 4), ('snapshot', 'wide_snapshot', 8)])
def test_lazy_wide_values_preserve_control_flow_and_definition_time(fixture, entry, width, level):
    code, labels, _, _, _ = generate(parse_module(FIXTURES[fixture]).kernels[entry], 'sm_80', level)
    out, inp, guard = 0x100000000, 0x100010000, b'\xa5' * 64
    values = [0x12340000 + i for i in range(32)]
    raw_input = struct.pack('<32I', *values)
    expected = values if fixture == 'pointer' else [i * 7 for i in range(32)]
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(out - 64, guard + bytes(width * 32) + guard)
        memory.write_bytes(inp - 64, guard + raw_input + guard)
        params = struct.pack('<QQ', out, inp) if fixture == 'pointer' else struct.pack('<Q', out)
        Machine(load_listing(listing(code, labels)), Launch(block=(32, 1, 1), params=params, poison=poison), memory).run()
        assert [memory.read(out + width * i, width) for i in range(32)] == expected
        assert bytes(memory.read(inp + i, 1) for i in range(len(raw_input))) == raw_input
        for address, size in ((out, width * 32), (inp, len(raw_input))):
            for offset in (-64, size):
                assert bytes(memory.read(address + offset + i, 1) for i in range(64)) == guard


@pytest.mark.parametrize('level', LEVELS)
def test_publication_fences_retain_order_and_wait_for_async_operations(level):
    code, _, words, _, _ = generate(parse_module(FIXTURES['publication']).kernels['publication'], 'sm_80', level)
    assert words
    instructions = [text for text, _ in code]
    release = [i for i, text in enumerate(instructions) if text.startswith('MEMBAR.ALL.GPU')]
    acquire = [i for i, text in enumerate(instructions) if text.startswith('CCTL.IVALL')]
    assert len(release) == len(acquire) == 2
    for i in release:
        ordered = [text for text in instructions[i + 1:]
                   if text.startswith(('MEMBAR', 'ERRBAR', 'CCTL', 'ATOMG', 'LDG', 'STG', 'LDS', 'STS'))]
        assert ordered[0] == 'ERRBAR'
        assert ordered[1].startswith('ATOMG.E.EXCH')
    for i in acquire:
        assert instructions[i - 1].startswith('ATOMG.E.ADD')
        atomic_control = code[i - 1][1]
        assert atomic_control.wbar != 7
        assert code[i][1].wait & (1 << atomic_control.wbar)
