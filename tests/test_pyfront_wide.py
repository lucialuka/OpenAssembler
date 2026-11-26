"""Typed wide integers match independent fixed-width references at every native level."""
import json
from pathlib import Path
import struct
import pytest
from openassembler import build
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch,run_kernel
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import NotSupported,compile_file,compile_kernel,find_kernel
from tests.backend_harness import listing

ROOT=Path(__file__).resolve().parent
CASES=json.loads((ROOT/'data/pyfront_wide.json').read_text())
FORMAT={'s64':'q','u64':'Q','s32':'i','u32':'I'}


@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
@pytest.mark.parametrize('case',CASES,ids=[c['name'] for c in CASES])
def test_full_width_values_arrays_parameters_helpers_and_casts(case,level):
    text=compile_kernel(find_kernel(ROOT/'data/pyfront_wide.py',case['name']))
    module=parse_module(text);kernel=module.kernels[case['name']]
    addresses={name:0x100000000+0x2000*i+256 for i,name in enumerate(case['arrays'])}
    def fresh():
        memory=Memory()
        for name,(kind,values) in case['arrays'].items():
            payload=struct.pack('<'+str(len(values))+FORMAT[kind],*values)
            memory.write_bytes(addresses[name]-256,b'\xa5'*256+payload+b'\xa5'*256)
        return memory
    def read(memory):
        result={}
        for name,(kind,values) in case['arrays'].items():
            size=struct.calcsize(FORMAT[kind])*len(values)
            raw=bytes(memory.read(addresses[name]+i,1) for i in range(size))
            result[name]=list(struct.unpack('<'+str(len(values))+FORMAT[kind],raw))
            for offset in (-256,size):
                assert bytes(memory.read(addresses[name]+offset+i,1) for i in range(256))==b'\xa5'*256
        return result
    memory=fresh();run_kernel(module,case['name'],PTXLaunch(block=(16,1,1),params={**addresses,**case['scalars']}),memory)
    assert read(memory)==case['expected']
    blob=b''
    for kind,name in kernel.params:
        blob+=b'\0'*((-len(blob))%(8 if kind.endswith('64') else 4))
        blob+=struct.pack('<'+FORMAT[kind],addresses.get(name,case['scalars'].get(name)))
    code,labels,words,_,_=build.generate(kernel,'sm_80',level);assert words
    memory=fresh();Machine(load_listing(listing(code,labels)),Launch(block=(16,1,1),params=blob,param_base=0x160),memory).run()
    assert read(memory)==case['expected']


@pytest.mark.parametrize('expression,diagnostic',[
    ('x ** i64(2)','operator Pow'),('x / x','convert s64 to f32'),
    ('x // x','64-bit floor division'),('x % x','64-bit floor division'),
    ('x + float(1)','mixed-type'),('float(x)','convert s64 to f32'),
    ('i64(1.25)','integer values'),('u64(1.25)','integer values'),
    ('x << -1','constant count'),('x >> 64','constant count'),('x >> i32(2)','constant count'),
    ('x + 18446744073709551616','does not fit'),('abs(x)','abs'),
    ('min(x,x)','min'),('int(x)','explicit i32/u32/i64/u64'),
])
def test_wide_unsupported_domains_reject_with_source(tmp_path,expression,diagnostic):
    path=tmp_path/'unsupported.py';path.write_text('from openassembler.pyfront import *\n@kernel\ndef test(x:i64,o:Array(i64)):\n    o[0] = '+expression+'\n')
    with pytest.raises(NotSupported,match=diagnostic) as error:compile_file(path)
    assert error.value.path==str(path) and error.value.lineno==4


def test_oversize_shared_wide_array_rejects_and_metadata_disambiguates_scalar_u64(tmp_path):
    path=tmp_path/'bad.py';path.write_text('@kernel\ndef test():\n    a = shared_array(i64,6145)\n')
    with pytest.raises(NotSupported,match='49152-byte static shared-memory limit'):compile_file(path)
    ptx=compile_kernel(find_kernel(ROOT/'data/pyfront_wide.py','unsigned'))
    marker='// openassembler-pyfront-types-v1 '
    contract=json.loads(next(line[len(marker):] for line in ptx.splitlines() if line.startswith(marker)))
    assert contract['schema']==1 and contract['kernel']=='unsigned'
    assert next(p for p in contract['parameters'] if p['name']=='offset')=={'name':'offset','kind':'scalar','dtype':'u64'}
    assert next(p for p in contract['parameters'] if p['name']=='a')=={'name':'a','kind':'array','dtype':'u64'}
    assert 'cvta.to.global.u64' in ptx
