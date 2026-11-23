"""Atomic effects, old values, typed bounds and barrier convergence from Python."""
import json,struct
from pathlib import Path
import pytest
from openassembler import build
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
ROOT=Path(__file__).resolve().parent
CASES=json.loads((ROOT/'data/pyfront_primitives.json').read_text())

@pytest.mark.parametrize('case',CASES,ids=[c['name']for c in CASES])
@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
def test_primitive_old_values_effects_bounds_and_evaluation_order(case,level):
    from openassembler.ptx.parse import parse_module
    ptx=compile_kernel(find_kernel(ROOT/'data/pyfront_primitives.py',case['name']));kernel=parse_module(ptx).kernels[case['name']]
    addresses={name:0x100000000+i*8192+256 for i,name in enumerate(case['arrays'])};blob=b''
    for kind,name in kernel.params:blob+=struct.pack('<Q'if kind=='u64'else'<I',addresses.get(name,case['scalar'].get(name)))
    code,labels,*_=build.generate(kernel,'sm_80',level)
    for poison in(None,0xdeadbeef):
        memory=Memory()
        for name,values in case['arrays'].items():memory.write_bytes(addresses[name]-256,b'\xa5'*256+struct.pack('<'+str(len(values))+'I',*(x&0xffffffff for x in values))+b'\xa5'*256)
        Machine(load_listing(listing(code,labels)),Launch(block=(case['threads'],1,1),params=blob,param_base=0x160,poison=poison),memory).run()
        for name,expected in case['expected'].items():
            assert[memory.read(addresses[name]+4*i,4)for i in range(len(expected))]==[x&0xffffffff for x in expected]
            for offset in(-256,len(expected)*4):assert bytes(memory.read(addresses[name]+offset+i,1)for i in range(256))==b'\xa5'*256

@pytest.mark.parametrize('body,diagnostic',[
 ('atomic_add(a,0,1.0)','value type'),('atomic_exchange(a,0,1)','require u32'),
 ('atomic_add(a,0)','expects 3'),('atomic_add(a,i64(0),1)','int32'),
 ('atomic_add(unknown,0,1)','target'),('atomic_add(a,0,1,scope=1)','keyword'),
])
def test_primitive_unsupported_contracts_reject(tmp_path,body,diagnostic):
    p=tmp_path/'bad.py';p.write_text(f'@kernel\ndef k(a:Array(i32)):\n    {body}\n')
    with pytest.raises(NotSupported,match=diagnostic):compile_file(p)

@pytest.mark.parametrize('annotation', ['Buffer(i32,(4,),access="read")','Array(f32)','Array(i64)'])
def test_primitive_wrong_storage_rejects(tmp_path,annotation):
    p=tmp_path/'bad.py';p.write_text(f'@kernel\ndef k(a:{annotation}):\n    atomic_add(a,0,1)\n')
    with pytest.raises(NotSupported):compile_file(p)


def test_helper_effect_and_uniform_argument_atomic_barrier_reject(tmp_path):
    p=tmp_path/'helper.py';p.write_text('@device\ndef h(x:i32)->i32:\n    return atomic_add(x,0,1)\n@kernel\ndef k(a:Array(i32)):\n    a[0] = h(1)\n')
    with pytest.raises(NotSupported,match='pure scalar'):compile_file(p)
    p.write_text('@kernel\ndef k(a:Array(i32)):\n    winner = atomic_add(a,0,1)\n    if winner == 0:\n        syncthreads()\n')
    with pytest.raises(NotSupported,match='thread-dependent'):compile_file(p)
