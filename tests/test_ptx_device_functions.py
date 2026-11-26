"""Direct function expansion, scalar slot ABI, independent bytes and rejection gates."""
import json,struct
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PTXLaunch
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
HERE=Path(__file__).resolve().parent
CASES=json.loads((HERE/'data/device_function_fixtures.json').read_text())
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
HELPER='.func (.param .b32 r) f(.param .b32 a){.reg .u32 %x;ld.param.u32 %x,[a];st.param.b32 [r],%x;ret;}\n'
CALLER='.entry k(){.reg .u32 %x;{.param .b32 a;.param .b32 r;st.param.b32 [a],7;call.uni (r),f,(a);ld.param.b32 %x,[r];}ret;}'

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_direct_function_complete_bytes_and_call_inventory(case,level):
    module=parse_module(case['source']);kernel=module.kernels[case['entry']]
    assert module.inlined_calls[case['entry']]and all(i.op!='call'for i in kernel.body)
    code,labels,words,_,_=build.generate(kernel,'sm_80',level);assert words
    before=bytes.fromhex(case['input_hex']);wanted=bytes.fromhex(case['expected_hex']);guard=b'\xa5'*64;inp,out=0x100010000,0x100000000
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,guard+before+guard);memory.write_bytes(out-64,guard+struct.pack('<Q',0x13579bdf2468ace0)*128+guard)
        if engine=='ptx':Interp(module,case['entry'],PTXLaunch(block=(128,1,1),params={kernel.params[0][1]:out,kernel.params[1][1]:inp,kernel.params[2][1]:91}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(len(before)+128))==guard+before+guard

@pytest.mark.parametrize('helper,caller,reason',[
    (HELPER,CALLER.replace('f,(a)','missing,(a)'),'unresolved device function'),
    (HELPER.replace('ret;','call.uni (r),f,(a);ret;'),CALLER,'recursive device call'),
    (HELPER,CALLER.replace('f,(a)','f,()'),'arity mismatch'),
    (HELPER,CALLER.replace('.param .b32 a;','.param .b64 a;'),'matching.*width'),
    (HELPER,CALLER.replace('st.param.b32 [a],7;',''),'before definite initialization'),
    (HELPER.replace('st.param.b32 [r],%x;',''),CALLER,'not definitely initialized'),
    (HELPER.replace('ret;',''),CALLER,'falls through'),
    (HELPER.replace('st.param.b32 [r],%x;','st.param.b32 [a],%x;'),CALLER,'read-only'),
    (HELPER.replace('[a]','[a+4]'),CALLER,'unknown device parameter'),
    (HELPER,CALLER.replace('ld.param.b32 %x,[r]','ld.param.b32 %x,[r+4]'),'parameter slot used outside'),
    (HELPER.replace('.param .b32 r','.param .align 16 .b8 r[16]'),CALLER,'parameter ABI'),
    (HELPER.replace('.param .b32 r)', '.param .b32 r,.param .b32 s)'),CALLER,'at most one'),
    (HELPER.replace('ret;','exit;'),CALLER,'must return'),
    (HELPER,CALLER.replace('f,(a)','%x,(a)'),'unresolved device function'),
    (HELPER,CALLER.replace('call.uni','call.uni.tail'),'direct scalar'),
])
def test_unresolved_recursive_abi_slot_return_and_escape_domains_reject(helper,caller,reason):
    with pytest.raises(SyntaxError,match=reason):
        parse_module(HEADER+helper+caller)


def test_false_guard_preserves_initialized_caller_return_slot():
    module=parse_module(HEADER+HELPER+CALLER.replace('.reg .u32 %x;','.reg .u32 %x;.reg .pred %p;').replace('call.uni','st.param.b32 [r],9;setp.eq.u32 %p,%x,0;@%p call.uni'))
    assert any(i.op=='bra'and i.guard for i in module.kernels['k'].body)


def test_unused_external_declaration_is_recorded_but_not_called():
    m=parse_module(HEADER+'.extern .func (.param .b32 r) ext(.param .b32 a);.entry k(){ret;}')
    assert not m.functions['ext']['defined']and not m.inlined_calls


@pytest.mark.oracle
def test_normal_builder_owns_inlined_function_inventory_and_frame(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']=='local');path=tmp_path/'calls.ptx';path.write_text(case['source']);target=tmp_path/'out'
    assert build.compile_paths(path,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0]
    from openassembler.native_ownership import validate
    binary=(target/entry['cubin']).read_bytes();validate(entry,binary)
    contract=entry['kernels'][0]['inlined_device_calls'];assert contract['functions']['helper']['call_sites']==2
    assert entry['kernels'][0]['local_bytes']==64
    contract['functions']['helper']['call_sites']=3
    with pytest.raises(ValueError,match='inventory'):validate(entry,binary)


def test_function_body_source_lines_survive_expansion():
    text=HEADER+HELPER.replace('add.u32','add.u32')+CALLER
    module=parse_module(text);kernel=module.kernels['k']
    store=next(i for i in kernel.body if i.op=='mov'and i.mods==('b32',)and len(i.args)==2 and getattr(i.args[1],'name','').endswith('x'))
    assert store.line==text[:text.index('st.param.b32 [r],%x')].count('\n')+1


def test_forward_prototype_uses_signature_types_and_definition_bindings():
    prototype='.func (.param .b32 other_result) f(.param .b32 other_input);\n'
    module=parse_module(HEADER+prototype+HELPER+CALLER)
    assert module.functions['f']['parameters']==[('b32','a')]and module.inlined_calls['k']=={'f':1}


def test_external_declaration_and_local_definition_conflict_rejects():
    with pytest.raises(SyntaxError,match='external declaration'):
        parse_module(HEADER+'.extern .func (.param .b32 r) f(.param .b32 a);'+HELPER+CALLER)


def test_call_return_must_be_defined_on_false_guard_path():
    source=CALLER.replace('.reg .u32 %x;','.reg .u32 %x;.reg .pred %p;setp.eq.u32 %p,0,1;').replace('call.uni','@%p call.uni')
    with pytest.raises(SyntaxError,match='before definite initialization'):parse_module(HEADER+HELPER+source)


def test_nonexponential_expansion_depth_and_register_range_limits():
    source=HELPER.replace('f(', 'f0(')
    for i in range(1,34):source+=HELPER.replace('f(',f'f{i}(').replace('st.param.b32 [r],%x;',f'call.uni (r),f{i-1},(a);')
    with pytest.raises(SyntaxError,match='depth 32'):parse_module(HEADER+source+CALLER.replace('f,(a)','f33,(a)'))
    with pytest.raises(SyntaxError,match='register declaration too large'):
        parse_module(HEADER+HELPER.replace('%x;','%x<999999999999>;',1)+CALLER)
