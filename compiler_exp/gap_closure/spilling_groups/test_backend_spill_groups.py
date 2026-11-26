"""Independent complete-output checks for bounded tuple/predicate pressure."""
import functools
import json
import operator
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

CASES=json.loads((Path(__file__).parent/'data/spill_group_fixtures.json').read_text())
SENTINEL=0x13579bdf


def reference(data,case,n,lanes):
    result=[SENTINEL]*(lanes*case['output_words'])
    for lane in range(n):
        values=data[lane*case['count']:(lane+1)*case['count']]
        checksum=functools.reduce(operator.xor,values,0)if case['width']else 0
        value=0;even=lane%2==0
        for step in range(3):
            contribution=0
            for index,word in enumerate(values[:case['predicates']]):
                bit=bool(word&1)
                selected=(bit^(even and step%2==0),True,True,not(bit^((not even)and step%2==0)))[index%4]
                contribution+=(index+1)*selected
            value=((value^checksum)+contribution)&0xffffffff
        if case['local']:value^=values[0]
        base=lane*case['output_words']
        if case['width']:
            if even:result[base:base+case['count']]=values
            result[base+case['count']]=value
        else:result[base]=value
    return result


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_full_aligned_and_predicate_spilled_state(case,level):
    resources={};kernel=parse_module(case['source']).kernels[case['name']]
    code,labels,words,_,registers=build.generate(kernel,'sm_80',level,resource_info=resources)
    assert words and registers<=case['limit']and resources['spill_slots']>0
    rng=random.Random(17052);data=[rng.getrandbits(32)for _ in range(32*case['count'])]
    original=struct.pack('<'+str(len(data))+'I',*data);guard=b'\xa5'*64
    inp,out=0x100010000,0x100000000;size=32*case['output_words']*4
    expected=reference(data,case,19,32)
    for poison in(None,0xdeadbeef):
        memory=Memory();memory.write_bytes(inp-64,guard+original+guard)
        memory.write_bytes(out-64,guard+struct.pack('<'+str(size//4)+'I',*([SENTINEL]*(size//4)))+guard)
        machine=Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),stack_top=0x100000,poison=poison),memory)
        machine.run()
        assert[memory.read(out+4*i,4)for i in range(size//4)]==expected
        assert bytes(memory.read(inp+i,1)for i in range(len(original)))==original
        for address,length in((inp,len(original)),(out,size)):
            for offset in(-64,length):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard
        for warp in machine.warps:
            for thread in warp.lanes:assert all(0x100000-resources['local_bytes']<=a<0x100000 for a in thread.local.data)


@pytest.mark.parametrize('name',['group2_tight','group4_tight','groups_predicates_local'])
def test_reloaded_groups_stay_consecutive_and_aligned(name):
    case=next(c for c in CASES if c['name']==name)
    selector=ra.Selector(parse_module(case['source']).kernels[name])
    ops,phys,_=allocate_kernel(selector,ra.coalesce(ra.dce(selector.run())))
    for group in set(ra._groups(ops).values()):
        assert [phys[v]for v in group]==list(range(phys[group[0]],phys[group[0]]+len(group)))
        assert phys[group[0]]%len(group)==0
    assert len(set(ra.allocate_preds(ops).values()))<=7


def test_simultaneous_predicate_operands_fail_closed():
    stack=ra.V(1);predicates=[ra.PV(i)for i in range(1,9)]
    ops=[ra.Op('MOV {d0}, c[0x0][0x28]',[stack],[],side=True)]
    ops+=[ra.Op('define',[predicate])for predicate in predicates]
    ops.append(ra.Op('consume',[],[(predicate,False)for predicate in predicates],side=True))
    selector=SimpleNamespace(k=SimpleNamespace(directives={}),r1=stack,local_frame=LocalFrame({}, {}, 0))
    with pytest.raises(ra.NotSupported,match='predicate pressure exceeds bounded'):allocate_kernel(selector,ops)


@pytest.mark.oracle
def test_group_and_predicate_spill_resources_reach_owned_binary(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']=='groups_predicates_local')
    source=tmp_path/'spill.ptx';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0];kernel=entry['kernels'][0]
    binary=(target/entry['cubin']).read_bytes()
    assert local_frame_inventory(binary)[case['name']]==kernel['local_bytes']==kernel['register_allocation']['local_bytes']
    assert kernel['registers']<=case['limit']and kernel['register_allocation']['spill_slots']>0
    from openassembler.native_ownership import validate
    validate(entry,binary)
