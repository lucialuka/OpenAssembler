"""Typed Python shared-wide exchange, complete bytes and storage/convergence limits."""
import json
from pathlib import Path
import struct
import pytest
from openassembler import build,oracle
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler.backend.ra import Selector
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PTXLaunch
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

HERE=Path(__file__).resolve().parent
CASES=json.loads((HERE/'data/pyfront_shared_wide.json').read_text())

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_shared_wide_source_complete_bytes_and_natural_layout(tmp_path,case,level):
    path=tmp_path/(case['name']+'.py');path.write_text(case['source']);ptx,names=compile_file(path)
    assert names==[case['name']]and compile_kernel(find_kernel(path,case['name']))==ptx
    module=parse_module(ptx);kernel=module.kernels[case['name']]
    assert Selector(kernel).smem_bytes==case['shared_bytes']
    code,labels,words,_,_=build.generate(kernel,'sm_80',level);assert words
    before=bytes.fromhex(case['input_hex']);wanted=bytes.fromhex(case['expected_hex']);guard=b'\xa5'*64
    inp,out=0x100010000,0x100000000
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,guard+before+guard)
        memory.write_bytes(out-64,guard+struct.pack('<Q',0x13579bdf2468ace0)*(128*4)+guard)
        if engine=='ptx':Interp(module,case['name'],PTXLaunch(block=(128,1,1),params={'a':inp,'out':out,'delta':case['delta'],'n':91}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',inp,out,case['delta']&((1<<64)-1),91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(len(before)+128))==guard+before+guard

@pytest.mark.parametrize('dtype',['i64','u64'])
@pytest.mark.parametrize('body,diagnostic',[
    ('s = shared_array(TYPE, 6145)','49160'),
    ('p = shared_array(i32, 1)\ns = shared_array(TYPE, 6144)','49160'),
    ('s = shared_array(TYPE, True)','positive integer literal'),
    ('s = shared_array(TYPE, 0)','positive integer literal'),
    ('s = shared_array(TYPE, -1)','positive integer literal'),
    ('s = shared_array(TYPE, 2.0)','positive integer literal'),
    ('s = shared_array(TYPE, n)','positive integer literal'),
    ('s = shared_array(TYPE, 2)\ns[-1] = TYPE(3)','negative index'),
    ('s = shared_array(TYPE, 2)\ns[2] = TYPE(3)','out of range'),
    ('s = shared_array(TYPE, 2)\no[0] = s[0]','no possible writer'),
    ('s = shared_array(TYPE, 2)\ns[0] = TYPE(1)\no[0] = s[1]','no possible writer'),
    ('s = shared_array(TYPE, 2)\nif tid_x() == 0:\n    s[0] = TYPE(1)\n    syncthreads()','thread-dependent control flow'),
    ('s = shared_array(TYPE, 2)\nif tid_x() != 0:\n    return\ns[0] = TYPE(1)\nsyncthreads()','thread-dependent control flow'),
])
def test_wide_shared_resource_index_initialization_and_barrier_limits(tmp_path,dtype,body,diagnostic):
    body=body.replace('TYPE',dtype)
    source='from openassembler.pyfront import *\n@kernel\ndef bad(o: Array('+dtype+'), n: i32):\n'+''.join('    '+line+'\n'for line in body.splitlines())
    path=tmp_path/'bad.py';path.write_text(source)
    with pytest.raises(NotSupported,match=diagnostic)as error:compile_file(path)
    assert error.value.path==str(path)and error.value.lineno is not None

@pytest.mark.parametrize('dtype',['f32','i32','u32'])
def test_shared_boolean_count_rejects_before_ptx_generation(tmp_path,dtype):
    path=tmp_path/'count.py';path.write_text('@kernel\ndef bad():\n    s = shared_array('+dtype+', True)\n')
    with pytest.raises(NotSupported,match='positive integer literal'):compile_file(path)

@pytest.mark.oracle
def test_python_source_builder_owns_wide_shared_binary_and_reservation(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']=='i64_mixed')
    source=tmp_path/'mixed.py';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0]
    assert entry['language']=='python'
    from openassembler.native_ownership import validate
    from openassembler.container.elf import Elf
    binary=(target/entry['cubin']).read_bytes();validate(entry,binary)
    assert Elf.parse(binary).section('.nv.shared.'+case['name']).size==case['shared_bytes']


def test_cta_convergence_phase_has_one_token_and_remaps_labels():
    from openassembler.backend.compile import Out,_converge_cta_phases
    code=[Out('@P0 BRA `(other)',False,[],[]),Out('MOV R2, 0x1',False,[],[]),Out('BRA `(join)',False,[],[]),Out('MOV R2, 0x2',False,[],[]),Out('STS [R3], R2',False,[2,3],[]),Out('BAR.SYNC.DEFER_BLOCKING 0x0',False,[],[]),Out('EXIT',False,[],[])]
    labels={'other':3,'join':4};updated=_converge_cta_phases(code,labels)
    assert len(updated)==9 and updated[0].text.startswith('BSSY B0')and updated[6].text=='BSYNC B0'
    assert labels['other']==4 and labels['join']==5 and labels['__oa_cta_join_5']==6
    assert updated[7].text.startswith('BAR.')


@pytest.mark.parametrize('texts,labels',[
    (['MOV R2, 0x1','BAR.SYNC.DEFER_BLOCKING 0x0','EXIT'],{}),
    (['@P0 BRA `(leave)','BAR.SYNC.DEFER_BLOCKING 0x0','EXIT'],{'leave':2}),
    (['@P0 BRA `(cycle)','BAR.SYNC.DEFER_BLOCKING 0x0','EXIT'],{'cycle':0}),
    (['@P0 BRA `(unresolved)','BAR.SYNC.DEFER_BLOCKING 0x0','EXIT'],{}),
])
def test_ineligible_phase_keeps_code_and_labels(texts,labels):
    from openassembler.backend.compile import Out,_converge_cta_phases
    code=[Out(text,False,[],[])for text in texts];before=dict(labels)
    assert _converge_cta_phases(code,labels) is code and labels==before


def test_long_phase_avoids_recursion_and_allocates_distinct_join_name():
    from openassembler.backend.compile import Out,_converge_cta_phases
    code=[Out('@P0 BRA `(join)',False,[],[])]+[Out('MOV R2, R3',False,[3],[2])for _ in range(4096)]+[Out('BAR.SYNC.DEFER_BLOCKING 0x0',False,[],[]),Out('EXIT',False,[],[])]
    labels={'join':4097,'__oa_cta_join_4097':4098};updated=_converge_cta_phases(code,labels)
    assert len(updated)==len(code)+2 and '__oa_cta_join_4097_'in labels
    assert updated[labels['join']].text=='BSYNC B0'


def test_token_reuse_occurs_after_each_completed_barrier_phase():
    from openassembler.backend.compile import Out,_converge_cta_phases
    texts=['@P0 BRA `(a)','MOV R2, 1','BAR.SYNC.DEFER_BLOCKING 0x0','@P0 BRA `(b)','MOV R2, 2','BAR.SYNC.DEFER_BLOCKING 0x0','EXIT']
    labels={'a':2,'b':5};updated=_converge_cta_phases([Out(t,False,[],[])for t in texts],labels)
    tokens=[o.text.split()[0]for o in updated if o.text.startswith(('BSSY','BSYNC'))]
    assert tokens==['BSSY','BSYNC','BSSY','BSYNC']
