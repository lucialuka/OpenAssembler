"""Narrow/64-bit shared exchange, adjacent bytes and explicit support limits."""
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
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/shared_width_fixtures.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_complete_shared_snapshots_sign_extension_and_wide_bits(case,level):
    module=parse_module(case['source']);code,labels,words,_,_=build.generate(module.kernels[case['name']],'sm_80',level)
    assert words
    original=bytes.fromhex(case['input_hex']);wanted=bytes.fromhex(case['expected_hex']);guard=b'\xa5'*64
    inp,out=0x100010000,0x100000000
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,guard+original+guard)
        memory.write_bytes(out-64,guard+struct.pack('<I',0x13579bdf)*(128*16)+guard)
        if engine=='ptx':Interp(module,case['name'],PLaunch(block=(128,1,1),params={'output':out,'input':inp,'n':91}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(len(original)+128))==guard+original+guard


@pytest.mark.parametrize('instruction',[
    'st.shared.u16 [%addr+1],%lo;','ld.shared.u64 %loadedwide,[%other+4];',
    'st.shared.v2.u64 [%addr],{%wide,%wide};','ld.shared.v2.u8 {%loaded,%lo},[%other];',
    'st.shared.volatile.u8 [%addr],%lo;','ld.shared.acquire.u8 %loaded,[%other];',
    'st.shared.u64 [arena+2048],%wide;','ld.shared.s16 %loaded,[arena+-2];',
])
def test_unaligned_literal_vector_ordering_and_symbol_bounds_reject(instruction):
    case=next(c for c in CASES if c['name']=='exchange_u8')
    source=case['source'].replace('@%writer st.shared.u8 [%addr+6],%lo;',instruction)
    with pytest.raises(ra.NotSupported):build.generate(parse_module(source).kernels[case['name']],'sm_80','safe')


def test_insufficient_symbol_alignment_rejects_wide_literal_access():
    case=next(c for c in CASES if c['name']=='symbol_u64')
    source=case['source'].replace('.shared .align 16 .b8 arena[2048];','.shared .align 4 .b8 arena[2048];')
    with pytest.raises(ra.NotSupported,match='declared size/alignment'):build.generate(parse_module(source).kernels[case['name']],'sm_80','safe')


@pytest.mark.parametrize('level',['mined','sched','opt'])
@pytest.mark.parametrize('name',['exchange_b64','existing_vector64','guarded_alias_b64'])
def test_shared64_barriers_retain_validated_issue_gap(name,level):
    case=next(c for c in CASES if c['name']==name)
    code,*_=build.generate(parse_module(case['source']).kernels[name],'sm_80',level)
    barriers=[ctl for text,ctl in code if text.startswith('BAR.')]
    assert len(barriers)==3 and all(ctl.stall>=6 for ctl in barriers)


@pytest.mark.oracle
def test_normal_builder_installs_owned_shared_width_binary(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']=='exchange_s16')
    source=tmp_path/'shared.ptx';source.write_text(case['source']);target=tmp_path/'build'
    assert build.compile_paths(source,target,build.Options(no_sass=True,opt_level='opt')).rc==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0]
    from openassembler.native_ownership import validate
    validate(entry,(target/entry['cubin']).read_bytes())
