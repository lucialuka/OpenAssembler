"""Independent source-lane arithmetic, rejoined paths and exact preserved state."""
import json
from pathlib import Path
import random
import struct

import pytest

from openassembler.backend import ra
from openassembler.build import generate
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch,run_kernel
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/shuffle_fixtures.json').read_text())


def expected(case,values,delta):
    transformed=[value^0x76543210 if i%32<16 else(value+19)&0xffffffff for i,value in enumerate(values)]
    result=[]
    for i in range(len(values)):
        lane=i%32;b=(33 if case['source_kind']=='literal'else delta if case['source_kind']=='parameter'else transformed[i]if case['source_kind']=='aliased'else lane*13+7)&31
        if case['mode']=='idx':source=b
        elif case['mode']=='up':source=lane-b if lane>=b else lane
        elif case['mode']=='down':source=lane+b if lane+b<=31 else lane
        else:source=lane^b
        result.append(transformed[i-lane+source])
    return result


@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
def test_full_warp_shuffle_independent_indices_after_reconvergence(case,level):
    module=parse_module(case['source']);code,labels,words,_,_=generate(module.kernels[case['name']],'sm_80',level)
    assert words
    for delta in (0,1,17,31,32,33,0xffffffff):
        rng=random.Random(15052+delta);values=[rng.getrandbits(32)for _ in range(128)]
        raw=struct.pack('<128I',*values);reference=expected(case,values,delta);guard=b'\xa5'*64
        for engine in ('sass','poisoned','ptx'):
            memory=Memory();out,inp=0x100000000,0x100010000
            memory.write_bytes(inp-64,guard+raw+guard);memory.write_bytes(out-64,guard+bytes(512)+guard)
            if engine=='ptx':run_kernel(module,case['name'],PTXLaunch(block=(128,1,1),params={'output':out,'input':inp,'delta':delta}),memory)
            else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,delta),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
            assert[memory.read(out+4*i,4)for i in range(128)]==reference
            assert bytes(memory.read(inp+i,1)for i in range(512))==raw
            for address in(out,inp):
                for offset in(-64,512):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard


@pytest.mark.parametrize('instruction',[
    'shfl.sync.idx.b32 %result,%value,%b,31,0xff;',
    'shfl.sync.down.b32 %result,%value,%b,31,0;',
    'shfl.sync.up.b32 %result,%value,%b,31,0xffffffff;',
    'shfl.sync.idx.b32 %result,%value,%b,0x101f,0xffffffff;',
    '@%branch shfl.sync.idx.b32 %result,%value,%b,31,0xffffffff;',
    'shfl.sync.idx.b32 %result|%branch,%value,%b,31,0xffffffff;',
    'shfl.sync.idx.b64 %result,%value,%b,31,0xffffffff;',
    'shfl.sync.unknown.b32 %result,%value,%b,31,0xffffffff;',
])
def test_unadmitted_shuffle_mask_clamp_predicate_and_type_reject(instruction):
    original=CASES[0]['source'];old='shfl.sync.idx.b32 %result,%value,33,31,0xffffffff;'
    assert old in original
    with pytest.raises(ra.NotSupported,match='PTX line|shuffle|shfl'):
        generate(parse_module(original.replace(old,instruction)).kernels[CASES[0]['name']],'sm_80','safe')
