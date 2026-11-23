"""Independent integer references for wide arithmetic, shifts, predicates and casts."""
import json
from pathlib import Path
import struct

import pytest

from openassembler.backend import ra
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

FIXTURE = json.loads((Path(__file__).parent / 'data/scalar_wide_fixtures.json').read_text())
CASES = FIXTURE['cases']
INPUT = bytes.fromhex(FIXTURE['input_hex'])
GUARD = b'\xa5' * 256


def signed(value, width):
    return value - (1 << width) if value & (1 << (width - 1)) else value


def reference(case):
    contract = case['contract']
    operation, width, literal = contract['op'], contract['width'], contract['literal']
    output = bytearray([0x3c] * (FIXTURE['threads'] * 16))
    for index in range(FIXTURE['valid_lanes']):
        x, y, count = struct.unpack_from('<QQI', INPUT, index * 32)
        count = count if literal is None else literal
        mask = (1 << width) - 1
        x, y = x & mask, y & mask
        compare = operation.startswith('setp')
        if operation.startswith('cast_'):
            value = (signed(x & 0xffffffff, 32) & mask if operation == 'cast_s64_s32_snapshot'
                     else x & 0xffffffff if operation == 'cast_u64_u32_snapshot' else x & mask)
        elif index % 2:
            value = 1 if compare else x
        elif operation.startswith('sub'):
            value = (x - y) & mask
        elif operation.startswith('shl'):
            value = (x << count) & mask if count < width else 0
        elif operation.startswith('shr'):
            value = ((signed(x, width) if operation.split('.')[-1].startswith('s') else x) >> min(count, width)) & mask
        else:
            _, comparison, dtype = operation.split('.')
            if dtype.startswith('s'):
                x, y = signed(x, width), signed(y, width)
            comparison = {'lo': 'lt', 'ls': 'le', 'hi': 'gt', 'hs': 'ge'}.get(comparison, comparison)
            value = int({'eq': x == y, 'ne': x != y, 'lt': x < y, 'le': x <= y, 'gt': x > y, 'ge': x >= y}[comparison])
        struct.pack_into('<I' if compare or width == 32 else '<Q', output, index * 16, value)
    return GUARD + output + GUARD


@pytest.mark.parametrize('level', ['safe', 'mined', 'sched', 'opt'])
@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_wide_integer_reference_predication_aliases_and_tails(case, level):
    code, labels, words, _, _ = generate(parse_module(case['source']).kernels['probe'], 'sm_80', level)
    assert words
    program = load_listing(listing(code, labels))
    expected = reference(case)
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(0x100000000, GUARD + INPUT + GUARD)
        memory.write_bytes(0x200000000, GUARD + b'\x3c' * (FIXTURE['threads'] * 16) + GUARD)
        Machine(program, Launch(block=(FIXTURE['threads'], 1, 1), poison=poison,
                                params=struct.pack('<QQI', 0x200000100, 0x100000100, FIXTURE['valid_lanes'])), memory).run()
        assert bytes(memory.read(0x200000000 + i, 1) for i in range(len(expected))) == expected
        assert bytes(memory.read(0x100000000 + i, 1) for i in range(len(INPUT) + 512)) == GUARD + INPUT + GUARD


@pytest.mark.parametrize('instruction', ['shl.b64 %x,%x,-1;', 'shr.s64 %x,%x,-1;',
                                      'setp.lt.f64 %p,%x,%y;', 'sub.sat.s64 %x,%x,%y;'])
def test_unimplemented_or_negative_wide_contracts_reject(instruction):
    text = CASES[0]['source'].replace('@%take sub.s64 %x,%x,%y;', instruction)
    with pytest.raises(ra.NotSupported, match='PTX line .*contract|negative shift count'):
        generate(parse_module(text).kernels['probe'], 'sm_80', 'safe')


@pytest.mark.parametrize('operation', ['shl.b32', 'shr.b32', 'shr.u32', 'shr.s32',
                                     'shl.b64', 'shr.b64', 'shr.u64', 'shr.s64'])
def test_every_literal_shift_through_63(operation):
    case = next(case for case in CASES if case['contract']['op'] == operation and case['contract']['literal'] == 7)
    width = case['contract']['width']
    value = 0x8000000080000001 if width == 64 else 0x80000001
    source_name = '%x' if width == 64 else '%a'
    for count in range(64):
        text = case['source'].replace(f'{operation} {source_name},{source_name},7;',
                                      f'{operation} {source_name},{source_name},{count};')
        code, labels, words, _, _ = generate(parse_module(text).kernels['probe'], 'sm_80', 'opt')
        assert words
        memory = Memory()
        memory.write_bytes(0x100000100, struct.pack('<QQI12x', value, 0, 0))
        Machine(load_listing(listing(code, labels)), Launch(block=(1, 1, 1),
                params=struct.pack('<QQI', 0x200000100, 0x100000100, 1)), memory).run()
        mask = (1 << width) - 1
        if operation.startswith('shl'):
            expected = (value << count) & mask if count < width else 0
        else:
            expected = ((signed(value, width) if operation.endswith('.s' + str(width)) else value)
                        >> min(count, width)) & mask
        assert memory.read(0x200000100, width // 8) == expected, (operation, count)
