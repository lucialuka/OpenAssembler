"""Complete low64 product bits from typed Python/PTX/CUDA and explicit support limits."""
import json
from pathlib import Path
import struct
import pytest
from openassembler import build,oracle
from openassembler.backend import ra
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file,NotSupported
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/wide_multiply_fixtures.json').read_text())

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_complete_products_overflow_carries_guards_snapshots_and_python_sources(case,level,tmp_path):
    text=case['source']
    if case['mode']=='python':
        source=tmp_path/(case['name']+'.py');source.write_text(case['python_source'])
        text,names=compile_file(source)
        assert names==[case['entry']]and text==case['source']
    module=parse_module(text);kernel=module.kernels[case['entry']]
    code,labels,words,_,_=build.generate(kernel,'sm_80',level);assert words
    inp,out=0x100010000,0x100000000;guard=b'\xa5'*64
    before,wanted=bytes.fromhex(case['input_hex']),bytes.fromhex(case['expected_hex'])
    if case['mode']=='cuda':params=[out,inp,inp+128*8,91];raw=struct.pack('<QQQI',*params)
    else:params=[out,inp,91,0x8123456776543210];raw=struct.pack('<QQI4xQ',*params)
    mapping=dict(zip((name for _,name in kernel.params),params))
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,guard+before+guard);memory.write_bytes(out-64,guard+struct.pack('<Q',0x13579bdf2468ace0)*128+guard)
        if engine=='ptx':Interp(module,case['entry'],PLaunch(block=(128,1,1),params=mapping),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=raw,poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(len(before)+128))==guard+before+guard

@pytest.mark.parametrize('instruction',[
    'mul.hi.u64 %r,%a,%b;','mul.hi.s64 %r,%a,%b;','mul.wide.u64 %r,%a,%b;',
    'mul.lo.b64 %r,%a,%b;','mul.lo.sat.u64 %r,%a,%b;',
    'mul.lo.u64 %r,%a;','mul.lo.u64 %r,{%a,%b},%b;',
    'mul.lo.u64 %r,%missing,%b;','mul.lo.u64 17,%a,%b;','mul.rn.f64 %r,%a,%b;',
])
def test_unvalidated_high_wide_modifiers_types_and_operand_shapes_reject(instruction):
    case=next(c for c in CASES if c['name']=='u64_register')
    source=case['source'].replace('mul.lo.u64 %r,%a,%b;',instruction)
    with pytest.raises(ra.NotSupported):build.generate(parse_module(source).kernels['probe'],'sm_80','safe')


def test_integer_product_rejects_raw_floating_literal_in_parser():
    case=next(c for c in CASES if c['name']=='u64_register')
    source=case['source'].replace('mul.lo.u64 %r,%a,%b;','mul.lo.u64 %r,0f3f800000,%b;')
    with pytest.raises(SyntaxError,match='floating literal width 32'):parse_module(source)


@pytest.mark.parametrize('expression',[
    'x * f32(2.0)','x * i32(2)','x * u64(2)','x / x','x // x','x % x',
    'x ** i64(2)','float(x)','x * 18446744073709551616','x * -9223372036854775809',
    'x * True','x * (x == x)',
])
def test_python_mixed_width_float_division_unbounded_and_predicate_domains_reject(tmp_path,expression):
    source=tmp_path/'bad.py';source.write_text('from openassembler.pyfront import *\n@kernel\ndef bad(x:i64,out:Array(i64)):\n    out[0] = '+expression+'\n')
    with pytest.raises(NotSupported)as error:compile_file(source)
    assert error.value.path==str(source)and error.value.lineno==4


@pytest.mark.oracle
@pytest.mark.parametrize('name',['u64_false_guard','cuda_multiply','python_i64_loop'])
def test_normal_ptx_cuda_and_python_builds_own_complete_product_binary(tmp_path,name):
    if not oracle.available('nvcc'if name=='cuda_multiply'else'ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']==name);kind=case['mode']
    suffix='.py'if kind=='python'else'.cu'if kind=='cuda'else'.ptx'
    source,target=tmp_path/('product'+suffix),tmp_path/'out'
    source.write_text(case.get('python_source',case.get('cuda_source',case['source'])))
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry,=json.loads((target/'manifest.json').read_text())['entries']
    from openassembler.native_ownership import validate
    validate(entry,(target/entry['cubin']).read_bytes())
    assert entry['kernels'][0]['name']==case['entry']
