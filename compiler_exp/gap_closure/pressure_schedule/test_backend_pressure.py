"""Independent integer/finite-FP32 outputs and virtual scheduling invariants."""
from functools import reduce
import json
from operator import xor
from pathlib import Path
import random
import struct

import pytest

from openassembler import build,oracle
from openassembler.backend import ra,pressure
from openassembler.backend.storage import local_frame_inventory
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/pressure_fixtures.json').read_text())
SENTINEL=0x13579bdf


def values(case):
    rng=random.Random(17052)
    if case.get('dtype')=='f32':return[struct.unpack('<I',struct.pack('<f',rng.randrange(-1024,1025)))[0]for _ in range(32*case['count'])]
    return[rng.getrandbits(32)for _ in range(32*case['count'])]


def reference(data,case,n):
    result=[SENTINEL]*32
    for lane in range(n):
        words=data[lane*case['count']:(lane+1)*case['count']]
        if case.get('dtype')=='f32':
            total=sum(int(struct.unpack('<f',struct.pack('<I',word))[0])for word in words)
            result[lane]=struct.unpack('<I',struct.pack('<f',3*total/2))[0];continue
        repetitions=3 if case['loop']else 1
        if case['predicates']:value=repetitions*sum(i+1 for i,word in enumerate(words)if(word&1)^(i%2))
        else:
            transformed=[word^(i+1)for i,word in enumerate(words)]
            value=repetitions*sum(transformed)if lane%2==0 else(reduce(xor,transformed,0)if repetitions%2 else 0)
        value&=0xffffffff
        if case['local']:value^=words[0]
        result[lane]=value
    return result


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_pressure_schedule_eliminates_spills_with_complete_outputs(case,level):
    resources={};kernel=parse_module(case['source']).kernels[case['name']]
    code,labels,_,_,registers=build.generate(kernel,'sm_80',level,resource_info=resources)
    assert registers<=case['limit']and resources['spill_slots']==0
    assert resources['local_bytes']==(case.get('frame_bytes',32)if case['local']else 0)
    data=values(case);original=struct.pack('<'+str(len(data))+'I',*data)
    inp,out=0x100010000,0x100000000;guard=b'\xa5'*64
    for poison in(None,0xdeadbeef):
        memory=Memory();memory.write_bytes(inp-64,guard+original+guard)
        memory.write_bytes(out-64,guard+struct.pack('<32I',*([SENTINEL]*32))+guard)
        machine=Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),stack_top=0x100000,poison=poison),memory)
        machine.run()
        assert[memory.read(out+4*i,4)for i in range(32)]==reference(data,case,19)
        assert bytes(memory.read(inp+i,1)for i in range(len(original)))==original
        for address,size in((inp,len(original)),(out,128)):
            for offset in(-64,size):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard
        for warp in machine.warps:
            for thread in warp.lanes:assert all(0x100000-resources['local_bytes']<=a<0x100000 for a in thread.local.data)


@pytest.mark.parametrize('name',['quad96','pred24','local_loop48'])
def test_order_preserves_register_predicate_and_memory_dependencies(name):
    case=next(case for case in CASES if case['name']==name)
    selector=ra.Selector(parse_module(case['source']).kernels[name]);ops=ra.coalesce(ra.dce(selector.run()))
    ordered=pressure.schedule(ops);positions={id(op):i for i,op in enumerate(ordered)}
    assert len(ops)==len(ordered)and{id(op)for op in ops}==set(positions)
    reads=[ra._all_uses(op)|(ra._defs(op)if op.guard is not None else set())for op in ops]
    writes=[ra._defs(op)for op in ops];memory=[]
    for i,op in enumerate(ops):
        opcode=op.fmt.split()[0].split('.')[0]if op.fmt else ''
        if op.side or op.label or op.target or opcode in('S2R','ULDC'):
            assert positions[id(op)]==i
        if opcode in('LDG','LDS','LDL','LDC'):memory.append(positions[id(op)])
        for j in range(i+1,len(ops)):
            if writes[i]&reads[j]or reads[i]&writes[j]or writes[i]&writes[j]:assert positions[id(op)]<positions[id(ops[j])]
    assert memory==sorted(memory)


def test_no_pressure_keeps_the_existing_allocation_path(monkeypatch):
    text='.version 8.7\n.target sm_80\n.address_size 64\n.entry simple(.param .u64 out){.reg .u64 %p;ld.param.u64 %p,[out];st.global.u32 [%p],1;ret;}'
    def unexpected(_):raise AssertionError('unnecessary pressure scheduling')
    monkeypatch.setattr(pressure,'schedule',unexpected)
    resources={};build.generate(parse_module(text).kernels['simple'],'sm_80','opt',resource_info=resources)
    assert resources['spill_slots']==0


def test_equal_resource_proposal_keeps_prior_loop_allocation():
    prior=json.loads((Path(__file__).parent/'data/spill_group_fixtures.json').read_text())
    case=next(case for case in prior if case['name']=='predicates')
    selector=ra.Selector(parse_module(case['source']).kernels[case['name']]);ops=ra.coalesce(ra.dce(selector.run()))
    from openassembler.backend.spill import allocate_kernel,_allocate_kernel
    baseline=_allocate_kernel(selector,ops);current=allocate_kernel(selector,ops)
    assert baseline[2]['spill_slots']>0 and current==baseline


def test_admission_rejects_extra_loads_before_a_guarded_exit(monkeypatch):
    from openassembler.backend import spill
    stack,value,predicate=ra.V(1),ra.V(2),ra.PV(1)
    load=lambda:ra.Op('LDL {d0}, [{s0}]',[value],[stack],var_lat=True)
    exit_op=ra.Op('EXIT',[],[],side=True,guard=(predicate,False))
    baseline=([load(),exit_op,load(),load()],{},dict(spill_slots=8,local_bytes=32))
    proposal=([load(),load(),exit_op,load()],{},dict(spill_slots=4,local_bytes=16))
    results=iter([baseline,proposal]);monkeypatch.setattr(spill,'_allocate_kernel',lambda *_:next(results))
    monkeypatch.setattr(pressure,'schedule',lambda ops:[ra.Op('replacement')])
    assert spill.allocate_kernel(None,[ra.Op('original')])is baseline


@pytest.mark.oracle
def test_recovered_local_loop_build_has_only_its_declared_frame(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(case for case in CASES if case['name']=='local_loop48')
    source=tmp_path/'pressure.ptx';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0];kernel=entry['kernels'][0]
    binary=(target/entry['cubin']).read_bytes()
    assert local_frame_inventory(binary)[case['name']]==kernel['local_bytes']==32
    assert kernel.get('register_allocation',{}).get('spill_slots',0)==0
    from openassembler.native_ownership import validate
    validate(entry,binary)
