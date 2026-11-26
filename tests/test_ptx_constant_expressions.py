"""Typed expression semantics, explicit unresolved cases and complete outputs."""
import json
from pathlib import Path
import struct

import pytest

from openassembler import build,oracle
from openassembler.ptx.parse import ImmI,ImmF,parse_operand,parse_instr,parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/constant_expression_fixtures.json').read_text())


@pytest.mark.parametrize('text,value',[
    ('1+2*3',7),('(1+2)*3',9),('8-3-2',3),('16/4/2',2),('-8/3',-2),
    ('1<<64',1),('1<<65',2),('1<<-1',-(1<<63)),('8>>65',4),('((.u64)-1)>>63',1),
    ('-8 % 3',2),('8 % -3',8),('((.s64)(-8 % 3))*-1',-2),
    ('0xffffffffffffffff+1',0),('18446744073709551616+1',1),
    ('(-2&-1)>0',0),('(-2&-1U)>0',1),('~0>0',1),
    ('(.s64)0xffffffffffffffff/2',0),('(.u64)-1/2',(1<<63)-1),
    ('(1.5<2.0)?3:4',3),('0?3:4',4),('1||0&&0',1),
    ('(0123+1)',84),('(0b1010+1)',11),
])
def test_integer_expression_precedence_signedness_and_casts(text,value):
    assert parse_operand(text)==ImmI(value)


@pytest.mark.parametrize('text,bits',[
    ('((1.0e20+1.0)-1.0e20)',0),('1.0/3.0',0x3fd5555555555555),
    ('1.5+0.5*4.0',0x400c000000000000),('(-0.0)+(-0.0)',0x8000000000000000),
    ('1.0e308*2.0',0x7ff0000000000000),('1.0e-307/1.0e308',0),
    ('0d7ff0000000000001+1.0',0x7ff8000000000001),
    ('0dfff8000000000001+1.0',0xfff8000000000001),
])
def test_floating_expression_rounds_each_operation_in_binary64(text,bits):
    assert parse_operand(text)==ImmF(bits,64)


@pytest.mark.parametrize('text',[
    '(0?1U:-1)>0','(1?-1:1U)/2','1 % 0','-(8 % 3)',
    '1.0+1','0f3f800000+1.0','(.s64)1.5','1?1.5:2.5',
    '1.0/0.0','0d7ff8000000000000/0.0','0d7ff0000000000000-0d7ff0000000000000',
    '0d7ff0000000000000*0.0','1**2','(1+2','1+','(42u+1)','(08+1)',
])
def test_unresolved_types_invalid_arithmetic_and_malformed_expressions_reject(text):
    with pytest.raises(SyntaxError):parse_operand(text)


@pytest.mark.parametrize('text',[
    'add.rn.f32 %f,(1+2),%a','st.global.f32 [%out],(1+2)',
    'selp.f32 %f,(1+2),%a,%p','setp.lt.f32 %p,%a,(1+2)',
])
def test_integer_expression_does_not_become_float_bits_implicitly(text):
    with pytest.raises(SyntaxError,match='requires a floating literal'):parse_instr(text,19)


def test_nested_and_oversized_expression_diagnostics_are_explicit():
    for text,diagnostic in(('('*140+'1.0'+')'*140,'128 nested'),('1?1:'*140+'1','128 nested'),('1?1:'*1000+'1','128 nested'),('1.0+'*2100+'1.0','4096 tokens')):
        with pytest.raises(SyntaxError,match=diagnostic):parse_operand(text)


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_expression_kernels_have_complete_golden_outputs(case,level):
    module=parse_module(case['source']);code,labels,_,_,_=build.generate(module.kernels[case['name']],'sm_80',level)
    inp,out=0x100010000,0x100000000;guard=b'\xa5'*64;data=case['input_words'];wanted=case['expected_words']
    original=guard+struct.pack('<32I',*data)+guard;reference=guard+struct.pack('<'+str(len(wanted))+'I',*wanted)+guard
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,original);memory.write_bytes(out-64,guard+struct.pack('<'+str(len(wanted))+'I',*([0x13579bdf]*len(wanted)))+guard)
        if engine=='ptx':Interp(module,case['name'],PLaunch(block=(32,1,1),params={'output':out,'input':inp,'n':19}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(32,1,1),params=struct.pack('<QQI',out,inp,19),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(reference)))==reference
        assert bytes(memory.read(inp-64+i,1)for i in range(len(original)))==original


@pytest.mark.oracle
def test_normal_builder_expression_sources_and_ownership(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(case for case in CASES if case['name']=='expressions_guards')
    source=tmp_path/'expression.ptx';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0]
    from openassembler.native_ownership import validate
    validate(entry,(target/entry['cubin']).read_bytes())
