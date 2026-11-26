"""Typed literals, double rounding, raw payloads and complete native execution."""
import json
from pathlib import Path
import struct

import pytest

from openassembler import build,oracle
from openassembler.ptx.parse import ImmF,parse_operand,parse_instr,parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/float_literal_fixtures.json').read_text())
SENTINEL=0x13579bdf


@pytest.mark.parametrize('text,bits64,bits32',[
    ('0.0',0,0),('-0.0',0x8000000000000000,0x80000000),
    ('+1.5',0x3ff8000000000000,0x3fc00000),('.5',0x3fe0000000000000,0x3f000000),
    ('2.',0x4000000000000000,0x40000000),('1E+3',0x408f400000000000,0x447a0000),
    ('1.0000000596046448',0x3ff0000010000000,0x3f800000),
    ('3.4028235677973366e38',0x47effffff0000000,0x7f800000),
    ('1.401298464324817e-45',0x36a0000000000000,1),
])
def test_decimal_is_binary64_before_use_conversion(text,bits64,bits32):
    assert parse_operand(text)==ImmF(bits64,64)
    assert parse_instr('mov.f32 %value,'+text,17).args[1]==ImmF(bits32,32)


@pytest.mark.parametrize('bits64,bits32',[
    (0x7ff0000000000000,0x7f800000),(0xfff0000000000000,0xff800000),
    (0x7ff8000000000000,0x7fc00000),(0xfff8000000000000,0xffc00000),
    (0x7ff0000000000001,0x7fc00000),(0x7ffabcdef1234567,0x7fd5e6f7),
    (0xfffabcdef1234567,0xffd5e6f7),(1,0),(0x8000000000000000,0x80000000),
])
def test_raw_double_use_conversion_preserves_observed_nan_sign_payload(bits64,bits32):
    assert parse_instr(f'mov.f32 %value,0d{bits64:016x}',1).args[1]==ImmF(bits32,32)
    assert parse_instr(f'mov.f64 %value,0d{bits64:016x}',1).args[1]==ImmF(bits64,64)


@pytest.mark.parametrize('bits',[0,0x3fc00000,0x80000000,0x7fc12345,0x7f800001])
def test_raw_single_exact_bits_and_wide_move_store_zero_extension(bits):
    assert parse_instr(f'mov.f32 %v,0f{bits:08x}',1).args[1]==ImmF(bits,32)
    assert parse_instr(f'mov.f64 %v,0f{bits:08x}',1).args[1]==ImmF(bits,64)
    assert parse_instr(f'st.global.f64 [%out],0f{bits:08x}',1).args[1]==ImmF(bits,64)


@pytest.mark.parametrize('text',[
    'mov.b32 %v,1.5','mov.u32 %v,1.5','mov.u64 %v,1.5','mov.b64 %v,0f3fc00000',
    'mov.f32 %v,0','mov.f64 %v,0x3f800000','mov.f32 %v,1e5000','mov.f32 %v,1e-5000',
    'mov.f64 %v,1e-308','selp.f32 %v,%a,%b,1.5','setp.lt.f32 1.5,%a,%b',
    'mov.b64 %v,{1.5,%r}',
    'st.global.b64 [%out],0f3fc00000',
])
def test_invalid_literal_contexts_and_binary64_range_reject(text):
    with pytest.raises(SyntaxError):parse_instr(text,19)


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_literal_sources_have_complete_independent_outputs(case,level):
    module=parse_module(case['source']);code,labels,_,_,_=build.generate(module.kernels[case['name']],'sm_80',level)
    data=case['input_words'];wanted=case['expected_words'];guard=b'\xa5'*64
    inp,out=0x100010000,0x100000000
    original=guard+struct.pack('<32I',*data)+guard
    reference=guard+struct.pack('<'+str(len(wanted))+'I',*wanted)+guard
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,original)
        memory.write_bytes(out-64,guard+struct.pack('<'+str(len(wanted))+'I',*([SENTINEL]*len(wanted)))+guard)
        if engine=='ptx':Interp(module,case['name'],PLaunch(block=(32,1,1),params={'output':out,'input':inp,'n':19}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(reference)))==reference
        assert bytes(memory.read(inp-64+i,1)for i in range(len(original)))==original


@pytest.mark.oracle
def test_normal_builder_decimal_literal_ownership_and_invalid_context(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(case for case in CASES if case['name']=='decimal_math')
    source=tmp_path/'literal.ptx';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0];binary=(target/entry['cubin']).read_bytes()
    from openassembler.native_ownership import validate
    validate(entry,binary)
    source.write_text(case['source'].replace('1.5','1e5000'))
    assert build.compile_paths(source,tmp_path/'invalid',build.Options(no_sass=True,opt_level='opt')).rc!=0
