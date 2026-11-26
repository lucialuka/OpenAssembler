"""Scalar spill state, tight limits, automatic pressure and local coexistence."""
import json
from pathlib import Path
import random
import struct
from types import SimpleNamespace

import pytest

from openassembler import build,oracle
from openassembler.backend import ra
from openassembler.backend.spill import allocate_kernel
from openassembler.backend.storage import LocalFrame,local_frame_inventory
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'regression-fixtures.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_spilled_scalar_loop_values_predicates_and_local_coexistence(case,level):
    kernel=parse_module(case['source']).kernels[case['name']];resources={}
    code,labels,words,_,registers=build.generate(kernel,'sm_80',level,resource_info=resources)
    assert words and registers<=case['limit']and resources['spill_slots']>0
    assert resources['local_bytes']%16==0
    rng=random.Random(17052);count=case['count'];data=[rng.getrandbits(32)for _ in range(32*count)]
    expected=[0x13579bdf]*32
    for lane in range(19):
        local=list(data[lane*count:(lane+1)*count]);value=0
        for _ in range(3):
            for i in range(count):
                if lane%2==0:local[i]=(local[i]+i+1)&0xffffffff
                value^=local[i]
        if case.get('local'):value^=data[lane*count]
        expected[lane]=value
    original=struct.pack('<'+str(len(data))+'I',*data);guard=b'\xa5'*64
    inp,out=0x100010000,0x100000000
    for poison in(None,0xdeadbeef):
        memory=Memory();memory.write_bytes(inp-64,guard+original+guard)
        memory.write_bytes(out-64,guard+struct.pack('<32I',*([0x13579bdf]*32))+guard)
        machine=Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),stack_top=0x100000,poison=poison),memory)
        machine.run()
        assert[memory.read(out+4*i,4)for i in range(32)]==expected
        assert bytes(memory.read(inp+i,1)for i in range(len(original)))==original
        for address,size in((inp,len(original)),(out,128)):
            for offset in(-64,size):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard
        for warp in machine.warps:
            for thread in warp.lanes:assert all(0x100000-resources['local_bytes']<=a<0x100000 for a in thread.local.data)


def test_aligned_group_pressure_still_rejects_without_splitting_groups():
    stack=ra.V(1);ops=[ra.Op('MOV {d0}, c[0x0][0x28]',[stack],[],side=True)];groups=[]
    for index in range(12):
        group=tuple(ra.V(2+4*index+i)for i in range(4));groups.append(group)
        ops.append(ra.Op('wide define {d0}',[group],[],var_lat=True))
    for group in groups:ops.append(ra.Op('wide consume {s0}',[],[group],side=True))
    selector=SimpleNamespace(k=SimpleNamespace(directives={'maxnreg':(16,)}),r1=stack,local_frame=LocalFrame({}, {}, 0))
    with pytest.raises(ra.NotSupported,match='wide/predicate'):allocate_kernel(selector,ops)


def test_impossible_low_native_register_limit_rejects():
    text=CASES[0]['source'].replace('.maxnreg 32','.maxnreg 8')
    with pytest.raises(ra.NotSupported,match='at least 16'):
        build.generate(parse_module(text).kernels[CASES[0]['name']],'sm_80','safe')


@pytest.mark.oracle
def test_ordinary_builder_spill_resource_manifest_and_frame_reservation(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=CASES[0];source=tmp_path/'spill.ptx';source.write_text(case['source'])
    target=tmp_path/'build';assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='safe')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0];kernel=entry['kernels'][0]
    assert kernel['registers']<=32 and kernel['register_allocation']['spill_slots']>0
    binary=(target/entry['cubin']).read_bytes()
    assert local_frame_inventory(binary)[case['name']]==kernel['local_bytes']==kernel['register_allocation']['local_bytes']
    from openassembler.native_ownership import validate
    validate(entry,binary)
