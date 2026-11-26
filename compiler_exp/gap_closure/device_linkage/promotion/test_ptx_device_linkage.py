"""Scalar source-library linkage, independent effects and normal build/transport custody."""
import json,struct
from pathlib import Path
import pytest
from openassembler import build,oracle
from openassembler.ptx.link import resolve,LinkError
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
HERE=Path(__file__).resolve().parent
CASES=json.loads((HERE/'data/device_linkage_fixtures.json').read_text())
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'
DEF='.visible .func (.param .b32 r) helper(.param .b32 a){.reg .u32 %x;ld.param.u32 %x,[a];st.param.b32 [r],%x;ret;}\n'
ROOT=HEADER+'.extern .func (.param .b32 r) helper(.param .b32 a);.entry probe(){.reg .u32 %x;{.param .b32 a;.param .b32 r;st.param.b32 [a],7;call.uni (r),helper,(a);ld.param.u32 %x,[r];}ret;}'

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['name'])
@pytest.mark.parametrize('level',['safe','opt'])
def test_linked_complete_buffers_private_names_and_selected_contract(case,level):
    source,contract=resolve([('entry',case['root'])]+[(f'lib{i}',s)for i,s in enumerate(case['libraries'])])
    assert contract['strategy']=='scalar-ptx-source-link'and contract['retained_functions']
    module=parse_module(source);code,labels,words,_,_=build.generate(module.kernels['probe'],'sm_80',level);assert words
    original=bytes.fromhex(case['input_hex']);wanted=bytes.fromhex(case['expected_hex']);guard=b'\xa5'*64;inp,out=0x100010000,0x100000000
    for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,guard+original+guard);memory.write_bytes(out-64,guard+struct.pack('<Q',0x13579bdf2468ace0)*128+guard)
        if engine=='ptx':Interp(module,'probe',PLaunch(block=(128,1,1),params={'out':out,'input':inp,'n':91}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)+128))==guard+wanted+guard
        assert bytes(memory.read(inp-64+i,1)for i in range(len(original)+128))==guard+original+guard

@pytest.mark.parametrize('libraries,root,reason',[
    ([HEADER+DEF,HEADER+DEF],ROOT,'duplicate strong'),
    ([HEADER+DEF.replace('b32 a','b64 a')],ROOT,'ABI mismatch'),
    ([HEADER+DEF.replace('.visible ', '')],ROOT,'unresolved external'),
    ([HEADER+DEF.replace('ret;','call.uni (r),helper,(a);ret;')],ROOT,'recursive'),
    ([HEADER+DEF.replace('.visible ','.weak '),HEADER+DEF.replace('.visible ','.weak ').replace('ld.param.u32','ld.param.b32')],ROOT,'nonidentical weak'),
    ([HEADER+'.entry library_entry(){ret;}'],ROOT,'kernel entries'),
    ([HEADER+'.global .u32 value;'+DEF],ROOT,'module data'),
    ([HEADER.replace('sm_80','sm_90')+DEF],ROOT,'target'),
    ([HEADER.replace('.address_size 64','.address_size 32')+DEF],ROOT,'address_size'),
    ([HEADER+DEF+DEF],ROOT,'duplicate device'),
    ([HEADER+DEF.replace('helper','missing')],ROOT,'unresolved external'),
    ([HEADER+'.visible .func (.param .b64 r) helper(.param .b32 a);'+DEF],ROOT,'signatures'),
    ([HEADER+DEF.replace('st.param.b32',' .loc 77 1 1\nst.param.b32')],ROOT,'debug file'),
])
def test_duplicate_abi_visibility_recursion_weak_storage_target_and_debug_rejections(libraries,root,reason):
    with pytest.raises(LinkError,match=reason):resolve([('entry',root)]+[(str(i),s)for i,s in enumerate(libraries)])


def test_link_determinism_and_unused_unresolved_declaration():
    text=HEADER+DEF+'.extern .func unused(.param .b32 x);'
    first=resolve([('entry',ROOT),('library',text)]);assert first==resolve([('entry',ROOT),('library',text)])
    assert 'unused'not in first[0]


def test_library_count_bound_and_relocatable_conflict():
    from openassembler.ptx.link import relocatable_flags
    with pytest.raises(LinkError,match='64 libraries'):resolve([('unit',ROOT)]*66)
    with pytest.raises(LinkError,match='conflicting'):relocatable_flags(['-rdc=false'])
    assert relocatable_flags(['-rdc=true'])==['-rdc=true']


@pytest.mark.oracle
def test_ordinary_cli_and_library_manifest_ownership(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case=next(c for c in CASES if c['name']=='private_names');source=tmp_path/'entry.ptx';source.write_text(case['root']);libs=[]
    for i,text in enumerate(case['libraries']):p=tmp_path/f'lib{i}.ptx';p.write_text(text);libs.append(p)
    from openassembler.cli import main
    target=tmp_path/'build';args=['compile',str(source),'--out-dir',str(target),'--no-sass']
    for path in libs:args.extend(['--device-library',str(path)])
    assert main(args)==0
    entry=json.loads((target/'manifest.json').read_text())['entries'][0]
    from openassembler.native_ownership import validate
    binary=(target/entry['cubin']).read_bytes();validate(entry,binary)
    link=entry['kernels'][0]['device_link'];assert len(link['libraries'])==2
    link['libraries'][0]['source_sha256']='0'*64
    with pytest.raises(ValueError,match='inventory'):validate(entry,binary)


@pytest.mark.oracle
def test_cuda_library_compilation_database_dependency_and_changed_header(tmp_path):
    if not oracle.available('nvcc'):pytest.skip('needs CUDA toolkit')
    import tempfile
    temporary=tempfile.TemporaryDirectory(prefix='oa-link-db-test-',dir=oracle.SCRATCH)
    tmp_path=Path(temporary.name)
    entry=tmp_path/'entry.cu';lib=tmp_path/'library.cu';header=tmp_path/'value.cuh';header.write_text('#define OFFSET 9u\n')
    entry.write_text('extern "C" __device__ unsigned helper(unsigned);\nextern "C" __global__ void probe(unsigned*out){out[0]=helper(3u);}\n')
    lib.write_text('#include "value.cuh"\nextern "C" __device__ __noinline__ unsigned helper(unsigned x){return x*MUL+OFFSET;}\n')
    db=tmp_path/'compile_commands.json';db.write_text(json.dumps([dict(directory=str(tmp_path),file=str(path),arguments=['nvcc','-c','-DMUL=17',str(path)])for path in(entry,lib)]))
    target=tmp_path/'build';options=build.Options(no_sass=True,device_libraries=[lib],compile_commands=db)
    assert build.compile_paths(entry,target,options).rc==0
    manifest=json.loads((target/'manifest.json').read_text())['entries'][0];link=manifest['kernels'][0]['device_link'];first=link['libraries'][0]['dependencies'][str(header)]
    assert '-DMUL=17'in link['libraries'][0]['compile_contract']['device_flags']
    header.write_text('#define OFFSET 11u\n');assert build.compile_paths(entry,target,options).rc==0
    latest=json.loads((target/'manifest.json').read_text())['entries'][0]['kernels'][0]['device_link']
    assert latest['libraries'][0]['dependencies'][str(header)]!=first
    temporary.cleanup()


def test_external_library_remote_bundle_and_serializable_options(tmp_path):
    from openassembler.remote_build import _sources
    root=tmp_path/'main';external=tmp_path/'external';root.mkdir();external.mkdir();src=root/'entry.ptx';lib=external/'helper.ptx';src.write_text(ROOT);lib.write_text(HEADER+DEF)
    options=build.Options(device_libraries=[lib]);_,payload,_=_sources(src,root/'output',options)
    record,=json.loads(payload['.oa-device-libraries.json']);assert payload[record['path']]==lib.read_bytes()
    assert record['source_origin']==str(lib.resolve())and not record['compile_commands']
    assert all(not key.startswith('source/helper')for key in payload)


def test_loc_directive_does_not_delete_following_inlined_instruction():
    helper=DEF.replace('st.param.b32 [r],%x;','.loc 1 7 1\nadd.u32 %x,%x,1;st.param.b32 [r],%x;')
    source=HEADER+'.file 1 "source.cu"\n'+helper+ROOT[len(HEADER):].replace('.extern .func (.param .b32 r) helper(.param .b32 a);','')
    kernel=parse_module(source).kernels['probe'];assert any(i.op=='add'for i in kernel.body)


def test_private_dotted_and_underscored_names_are_distinct():
    private=DEF.replace('.visible ','').replace('helper','internal.x')+DEF.replace('.visible ','').replace('helper','internal_x')
    exported=DEF.replace('st.param.b32 [r],%x;', '{.param .b32 arg;.param .b32 value;st.param.b32 [arg],%x;call.uni (value),internal.x,(arg);ld.param.b32 %x,[value];st.param.b32 [arg],%x;call.uni (value),internal_x,(arg);ld.param.b32 %x,[value];}st.param.b32 [r],%x;')
    source,contract=resolve([('entry',ROOT),('library',HEADER+private+exported)])
    module=parse_module(source)
    assert len(module.inlined_calls['probe'])==3
    assert len({row['name']for row in contract['retained_functions']})==3


def test_exported_signature_conflict_rejects_even_if_weak():
    with pytest.raises(LinkError,match='exported function signatures'):
        resolve([('entry',ROOT),('a',HEADER+DEF),('b',HEADER+DEF.replace('.visible ','.weak ').replace('.b32 a','.b64 a'))])


def test_empty_library_and_unreachable_external_have_no_executable_text():
    source,contract=resolve([('entry',ROOT),('empty',HEADER),('provider',HEADER+DEF)])
    assert len(contract['retained_functions'])==1 and set(parse_module(source).kernels)=={'probe'}


def test_debug_file_ids_remap_across_root_and_two_libraries():
    case=next(c for c in CASES if c['name']=='transitive')
    units=[('entry',case['root'].replace(HEADER,HEADER+'.file 1 "entry.cu"\n',1))]
    units +=[(str(i),s.replace(HEADER,HEADER+f'.file 1 "library{i}.cu"\n',1))for i,s in enumerate(case['libraries'])]
    source,contract=resolve(units)
    assert '.file 1 "entry.cu"'in source and '.file 2 "library0.cu"'in source and '.file 3 "library1.cu"'in source


def test_device_library_path_and_format_errors(tmp_path):
    from openassembler.ptx.link import link_paths
    missing=tmp_path/'missing.ptx'
    with pytest.raises(LinkError,match='does not exist'):link_paths(ROOT,[missing],arch='sm_80')
    bad=tmp_path/'lib.o';bad.write_bytes(b'not a source')
    with pytest.raises(LinkError,match='PTX or CUDA source'):link_paths(ROOT,[bad],arch='sm_80')
    with pytest.raises(LinkError,match='unique'):link_paths(ROOT,[bad,bad],arch='sm_80')


def test_unsupported_function_pointer_annotations_reject():
    with pytest.raises(LinkError,match='pointer annotations'):
        resolve([('entry',ROOT),('lib',HEADER+DEF.replace('.b32 a','.u64 .ptr .global a'))])
