"""Bounded relaxed GPU atomics: side effects, old returns, contention and tails."""
from collections import Counter
import json
from pathlib import Path
import random
import struct

import pytest

from openassembler.backend import ra
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch, run_kernel
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES = json.loads((Path(__file__).parent / 'data/atomic_fixtures.json').read_text())['cases']
N, THREADS, GRID, LANES, BINS = 67, 32, (3, 1, 1), 96, 17
GUARD, SENTINEL, CAS_COMPARE = b'\xa5' * 64, 0x13579bdf, 0x76543210


def pack(values):
    return struct.pack('<' + 'I' * len(values), *values)


def connected_exchange(initial, final, values, returns):
    edges = {}
    for old,new in zip(returns,values):edges.setdefault(old,[]).append(new)
    stack,path = [initial],[]
    while stack:
        if edges.get(stack[-1]):stack.append(edges[stack[-1]].pop())
        else:path.append(stack.pop())
    path.reverse()
    return (len(path)==len(values)+1 and path[0]==initial and path[-1]==final
            and Counter(zip(path,path[1:]))==Counter(zip(returns,values)))


@pytest.mark.parametrize('level', ['safe', 'mined', 'sched', 'opt'])
@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_atomic_state_return_values_predication_and_tails(case, mode, level):
    module = parse_module(case['source'])
    code, labels, words, _, _ = generate(module.kernels['probe'], 'sm_80', level)
    assert words
    program = load_listing(listing(code, labels))
    operation, returning = case['contract']['operation'], case['contract']['family'] == 'atom'
    count = BINS if mode else LANES
    edges = [0, 1, 0xffffffff, 0x7fffffff, 0x80000000]
    initial = [0xa5a5a5a5] + ([CAS_COMPARE] * count if operation == 'cas' else [edges[i % len(edges)] for i in range(count)])
    rng = random.Random(14052)
    data = []
    for i in range(LANES):
        value = 1 if mode and operation == 'add' else i + 1 if mode else rng.getrandbits(32)
        compare = CAS_COMPARE if mode else initial[i + 1] if i % 4 == 0 else initial[i + 1] ^ 1
        data.extend((value, compare))
    buffers = {'state': pack(initial), 'input': pack(data), 'output': pack([SENTINEL] * LANES)}
    addresses = {'output': 0x100000000, 'state': 0x100100000, 'input': 0x100200000}
    for engine in ('sass', 'poisoned', 'ptx'):
        memory = Memory()
        for name, raw in buffers.items():memory.write_bytes(addresses[name] - len(GUARD), GUARD + raw + GUARD)
        for cta in range(GRID[0]):
            if engine == 'ptx':
                run_kernel(module, 'probe', PTXLaunch(grid=GRID, block=(THREADS, 1, 1), ctaid=(cta, 0, 0),
                           params={**addresses, 'n': N, 'mode': mode}), memory)
            else:
                params = struct.pack('<QQQII', addresses['output'], addresses['state'], addresses['input'], N, mode)
                Machine(program, Launch(grid=GRID, block=(THREADS, 1, 1), ctaid=(cta, 0, 0), params=params,
                        poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
        assert bytes(memory.read(addresses['input'] + i, 1) for i in range(len(buffers['input']))) == buffers['input']
        assert memory.read(addresses['state'], 4) == initial[0]
        for name, raw in buffers.items():
            for offset in (-len(GUARD), len(raw)):
                assert bytes(memory.read(addresses[name] + offset + i, 1) for i in range(len(GUARD))) == GUARD
        outputs = [memory.read(addresses['output'] + 4 * i, 4) for i in range(LANES)]
        assert all(outputs[i] == SENTINEL for i in range(LANES) if i >= N or i % 2 or not returning)
        for bin_index in range(BINS if mode else N):
            participants = [i for i in range(bin_index, N, BINS) if i % 2 == 0] if mode else ([bin_index] if bin_index % 2 == 0 else [])
            old = initial[bin_index + 1]
            values, returns = [data[2 * i] for i in participants], [outputs[i] for i in participants]
            final = memory.read(addresses['state'] + 4 * (bin_index + 1), 4)
            if operation == 'add':
                assert final == (old + sum(values)) & 0xffffffff
                if returning:
                    expected = [(old + i) & 0xffffffff for i in range(len(values))] if mode else ([old] if participants else [])
                    assert Counter(returns) == Counter(expected)
            elif operation == 'exch':
                assert connected_exchange(old,final,values,returns)
            elif mode and values:
                assert final in values and Counter(returns) == Counter([old] + [final] * (len(values) - 1))
            else:
                assert final == (values[0] if participants and data[2 * participants[0] + 1] == old else old)
                assert returns == ([old] if participants else [])
        if not mode:
            assert [memory.read(addresses['state'] + 4 * (i + 1), 4) for i in range(N, LANES)] == initial[N + 1:]


@pytest.mark.parametrize('operation', ['atom.acquire.gpu.global.add.u32 %old,[%address],%value;',
    'atom.release.gpu.global.add.u32 %old,[%address],%value;',
    'atom.relaxed.sys.global.add.u32 %old,[%address],%value;',
    'atom.relaxed.cta.global.add.u32 %old,[%address],%value;',
    'atom.shared.add.u32 %old,[%address],%value;',
    'atom.global.add.u64 %old,[%address],%value;',
    'atom.global.add.f32 %old,[%address],%value;',
    'atom.global.cas.b32 %old,[%address],%value;',
    'atom.global.add.u32 %old,[%address+1],%value;',
    'red.global.add.u32 [%address],{%value,%compare};'])
def test_unsupported_atomic_scope_ordering_width_and_operands_fail_closed(operation):
    source = CASES[0]['source'].replace('@%take atom.global.add.u32 %old,[%address+4],%value;', operation)
    with pytest.raises(ra.NotSupported, match='PTX line .*contract|expected .*operands|requires'):
        generate(parse_module(source).kernels['probe'], 'sm_80', 'safe')


def test_exchange_linearization_rejects_disconnected_return_cycles():
    assert Counter([0,3,2]+[1]) == Counter([0]+[1,2,3])
    assert not connected_exchange(0,1,[1,2,3],[0,3,2])
    assert connected_exchange(0,3,[1,2,3],[0,1,2])
    assert connected_exchange(1,3,[1,2,3],[1,1,2])


@pytest.mark.parametrize('form',['atom.global.add.u32','red.global.add.u32'])
def test_dead_atomic_result_does_not_remove_the_memory_effect(form):
    operands='%old,[%ptr],1' if form.startswith('atom') else '[%ptr],1'
    text=f'''.version 8.7
.target sm_80
.address_size 64
.entry effect(.param .u64 state) {{
.reg .u64 %ptr; .reg .u32 %old;
ld.param.u64 %ptr,[state]; {form} {operands}; ret; }}'''
    for level in ('safe','mined','sched','opt'):
        code,labels,words,_,_=generate(parse_module(text).kernels['effect'],'sm_80',level)
        assert words and any('ATOMG' in line or 'RED.E' in line for line,_ in code)
        memory=Memory();memory.write(0x100000000,4,0xffffffff)
        Machine(load_listing(listing(code,labels)),Launch(block=(1,1,1),params=struct.pack('<Q',0x100000000)),memory).run()
        assert memory.read(0x100000000,4)==0
