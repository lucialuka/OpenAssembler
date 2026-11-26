"""Hardware-gated exact memory bytes and predicate CFG regressions (offline)."""
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

FIXTURE = json.loads((Path(__file__).parent / 'data/backend_memory_widths.json').read_text())
CASES = FIXTURE['cases']
GUARD = b'\xa5' * 256


@pytest.mark.parametrize('level', ['safe', 'mined', 'sched', 'opt'])
@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_exact_memory_bytes_and_predicate_control_flow(case, level):
    kernel = parse_module(case['source']).kernels['probe']
    code, labels, words, _, _ = generate(kernel, 'sm_80', level)
    assert words
    program = load_listing(listing(code, labels))
    payload = bytes.fromhex(case['input_hex'])
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(0x100000000, GUARD + payload + GUARD)
        memory.write_bytes(0x200000000, GUARD + bytes([0x3c]) * len(payload) + GUARD)
        Machine(program, Launch(block=(FIXTURE['threads'], 1, 1), poison=poison,
                                params=struct.pack('<QQI', 0x200000100, 0x100000100, FIXTURE['valid_lanes'])), memory).run()
        # Compare every byte: narrow stores must preserve adjacent bytes, false
        # predicates and inactive tail lanes must preserve their initial values.
        actual = bytes(memory.read(0x200000000 + i, 1) for i in range(len(payload) + 512))
        assert actual == GUARD + bytes.fromhex(case['expected_hex']) + GUARD
        assert bytes(memory.read(0x100000000 + i, 1) for i in range(len(payload) + 512)) == GUARD + payload + GUARD


@pytest.mark.parametrize('instruction', [
    'ld.global.v2.u64 {%a,%b},[%src];',
    'st.global.v2.u64 [%dst],{%wide,%wide};',
    'st.global.v4.u32 [%dst],{%a,%b};',
    'st.global.v4.u16 [%dst],{%a,%b,%c,%d};',
    'st.global.release.u32 [%dst],%a;',
    'st.global.volatile.u32 [%dst],%a;',
    'st.shared.u8 [%a],%a;',
    'ld.global.acquire.u32 %a,[%src];',
    'st.global.v2.f64 [%dst],{%wide,%wide};',
])
def test_unsupported_width_ordering_and_operand_contracts_reject(instruction):
    text = CASES[0]['source'].replace('@%take ld.global.v2.b32 {%a,%b},[%src];', instruction)
    with pytest.raises(ra.NotSupported, match='PTX line .*contract|PTX line .*requires'):
        generate(parse_module(text).kernels['probe'], 'sm_80', 'safe')


@pytest.mark.parametrize('cache', ['wb', 'cg', 'cs', 'wt'])
@pytest.mark.parametrize('name', ['store_v4_u32', 'store_u8', 'store_s16', 'store_u64'])
def test_store_cache_hints_preserve_the_value_contract(cache, name):
    case = dict(next(c for c in CASES if c['name'] == name))
    case['source'] = case['source'].replace('st.global.', 'st.global.' + cache + '.')
    test_exact_memory_bytes_and_predicate_control_flow(case, 'safe')
