"""Static local frames, private state, retained predicates and metadata custody."""
import copy
import json
from pathlib import Path
import random
import struct

import pytest

from openassembler import build,oracle
from openassembler.backend import ra
from openassembler.backend.storage import local_frame,local_frame_inventory,set_local_frames
from openassembler.container.elf import Elf
from openassembler.container import nvinfo
from openassembler.emu.machine import Launch,Machine,Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES=json.loads((Path(__file__).parent/'data/local_storage_fixtures.json').read_text())


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['name'])
@pytest.mark.parametrize('level',['safe','mined','sched','opt'])
def test_local_private_state_symbols_predicates_and_loop_updates(case,level):
    rng=random.Random(13052);values=[rng.getrandbits(32)for _ in range(384)]
    code,labels,words,_,_=build.generate(parse_module(case['source']).kernels[case['name']],'sm_80',level)
    assert words
    out,inp=0x100000000,0x100010000;guard=b'\xa5'*64
    reference=[(((values[i]+i%8)&0xffffffff)^2)^(values[i]if i%2==0 else 0x89abcdef)if i<259 else 0x13579bdf for i in range(384)]
    for poison in(None,0xdeadbeef):
        memory=Memory();memory.write_bytes(inp-64,guard+struct.pack('<384I',*values)+guard)
        memory.write_bytes(out-64,guard+struct.pack('<384I',*([0x13579bdf]*384))+guard)
        for cta in range(3):
            machine=Machine(load_listing(listing(code,labels)),Launch(grid=(3,1,1),block=(128,1,1),ctaid=(cta,0,0),
                params=struct.pack('<QQI',out,inp,259),poison=poison,stack_top=0x100000),memory)
            machine.run()
            frame=local_frame(parse_module(case['source']).kernels[case['name']])
            for warp in machine.warps:
                for thread in warp.lanes:
                    assert all(0x100000-frame.bytes<=address<0x100000 for address in thread.local.data)
        assert[memory.read(out+4*i,4)for i in range(384)]==reference
        assert[memory.read(inp+4*i,4)for i in range(384)]==values
        for address in(out,inp):
            for offset in(-64,1536):assert bytes(memory.read(address+offset+i,1)for i in range(64))==guard


@pytest.mark.parametrize('declaration,diagnostic',[
    ('.local .align 32 .b8 x[32];','alignment'),
    ('.local .align 4 .b8 x[65537];','65536'),
    ('.local .align 4 .b8 x[0];','positive'),
    ('.local .align 4 .u32 x;.local .align 4 .u32 x;','duplicate'),
])
def test_invalid_local_layout_rejects(declaration,diagnostic):
    kernel=parse_module('.version 8.7\n.target sm_80\n.address_size 64\n.entry bad(){'+declaration+'ret;}').kernels['bad']
    with pytest.raises(ra.NotSupported,match=diagnostic):build.generate(kernel,'sm_80','safe')


@pytest.mark.parametrize('instruction',[
    'ld.local.u8 %r,[x];','st.local.u64 [x],%r;',
    'ld.local.v2.u32 {%r,%s},[x];','ld.volatile.local.u32 %r,[x];',
    'st.local.u32 [x+1],%r;','ld.local.u32 %r,[x+8];',
    'ld.local.u32 %r,[x-4];','mov.u64 %wide,x;',
])
def test_unsupported_local_access_width_qualifier_and_bounds_reject(instruction):
    source='.version 8.7\n.target sm_80\n.address_size 64\n.entry bad(){.local .align 4 .b8 x[8];.reg .b32 %r,%s;.reg .b64 %wide;mov.u32 %r,1;'+instruction+'ret;}'
    with pytest.raises(ra.NotSupported):build.generate(parse_module(source).kernels['bad'],'sm_80','safe')


def test_local_layout_uses_explicit_alignment_and_padding():
    kernel=parse_module(CASES[0]['source']).kernels[CASES[0]['name']]
    frame=local_frame(kernel)
    assert frame.offsets=={'prefix':0,'slots':16,'suffix':48}and frame.bytes==64
    assert local_frame(parse_module(CASES[-1]['source']).kernels[CASES[-1]['name']]).bytes==65536


@pytest.mark.oracle
def test_local_donor_frame_replacement_manifest_and_fail_closed_metadata(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    source=tmp_path/'local.ptx';source.write_text(CASES[0]['source'])
    output=tmp_path/'build';result=build.compile_paths(source,output,build.Options(no_sass=True))
    assert result.rc==0
    entry=json.loads((output/'manifest.json').read_text())['entries'][0]
    binary=(output/entry['cubin']).read_bytes();reference=(output/entry['reference']).read_bytes()
    name=CASES[0]['name']
    assert local_frame_inventory(reference)[name]==0
    assert local_frame_inventory(binary)[name]==entry['kernels'][0]['local_bytes']==64
    from openassembler.native_ownership import validate
    assert validate(entry,binary)==binary
    broken=copy.deepcopy(entry);broken['kernels'][0]['local_bytes']=32
    with pytest.raises(ValueError,match='local frame'):validate(broken,binary)
    elf=Elf.parse(binary);records=nvinfo.parse(elf.section('.nv.info').data)
    records=[record for record in records if record.attr!=0x11]
    elf.replace_data('.nv.info',nvinfo.serialize(records));missing=elf.serialize()
    with pytest.raises(ValueError,match='record pair'):set_local_frames(missing,{name:64})
    with pytest.raises(ValueError,match='record pair'):local_frame_inventory(missing)
    for record in records:
        if record.attr==0x12:record.data=record.data[:4]+struct.pack('<I',32)
    elf.replace_data('.nv.info',nvinfo.serialize(records))
    with pytest.raises(ValueError):set_local_frames(elf.serialize(),{name:17})
