"""Pointer annotations preserve scalar parameter offsets and launch promises."""
import json
import struct
import tempfile
from pathlib import Path
import pytest
from openassembler.ptx.parse import parse_module
from openassembler import build,oracle
HEADER='.version 8.7\n.target sm_80\n.address_size 64\n'


def source(annotation):
    return HEADER+f'''.entry copy(.param .u32 n,.param .u64 {annotation} output,.param .u64 .ptr .global .align 8 input) {{
.reg .u32 %i,%n,%v; .reg .u64 %out,%in,%off,%p; .reg .pred %end;
ld.param.u32 %n,[n];ld.param.u64 %out,[output];ld.param.u64 %in,[input];
mov.u32 %i,%tid.x;setp.ge.u32 %end,%i,%n;@%end bra done;
mul.wide.u32 %off,%i,4;add.u64 %p,%in,%off;ld.global.u32 %v,[%p];
add.u64 %p,%out,%off;st.global.u32 [%p],%v;done:ret;}}
'''


@pytest.mark.parametrize('annotation,space,alignment',[('.ptr .global .align 16','global',16),('.ptr .global','global',4),('.ptr .align 8','generic',8),('.ptr','generic',4)])
def test_pointer_metadata_does_not_change_parameter_slot_layout(annotation,space,alignment):
    kernel=parse_module(source(annotation)).kernels['copy']
    assert kernel.params==[('u32','n'),('u64','output'),('u64','input')]
    assert kernel.parameter_attributes['output']=={'pointer':True,'space':space,'alignment':alignment}
    from openassembler.backend.ra import Selector
    selector=Selector(kernel)
    assert selector.param_off=={'n':0x160,'output':0x168,'input':0x170}


@pytest.mark.parametrize('declaration',['.param .u32 .ptr .global output','.param .u64 .ptr .shared output',
    '.param .u64 .ptr .global .align 3 output','.param .u64 .ptr .global .align 0 output',
    '.param .align 16 .b8 output[16]','.param .u64 .ptr .const output'])
def test_unimplemented_parameter_layout_or_address_space_fails(declaration):
    with pytest.raises(SyntaxError):parse_module(HEADER+'.entry k('+declaration+') {ret;}')


def test_annotated_and_scalar_words_remain_equal_with_real_packaged_ownership():
    if not oracle.available('ptxas'):pytest.skip('requires CUDA tools')
    oracle.SCRATCH.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='oa-pointer-abi-',dir=oracle.SCRATCH) as temporary:
        root=Path(temporary).resolve();path=root/'input.ptx';path.write_text(source('.ptr .global .align 16'))
        result=build.build_source(path,'ptx','input',build.Options(no_sass=True,verify=True),single=True)
        assert result.rc==0,result.error
        kernel=result.kernels[0];assert kernel.verify=='ok'
        plain=source('').replace('.param .u64 .ptr .global .align 8 input','.param .u64 input')
        before=build.generate(parse_module(plain).kernels['copy'],'sm_80','opt')
        after=build.generate(parse_module(path.read_text()).kernels['copy'],'sm_80','opt')
        assert before==after
        entry=result.entry(True)
        assert entry['kernels'][0]['parameter_attributes']['output']['alignment']==16
        binary=result.files[entry['cubin']]
        from openassembler.native_ownership import validate
        validate(entry,binary)
        entry['kernels'][0]['parameter_attributes']['output']['alignment']=8
        with pytest.raises(ValueError,match='inventory'):validate(entry,binary)


def test_explicit_runner_rejects_misaligned_pointer_offset(tmp_path):
    from tests.native_fixture import save_native_manifest
    from openassembler.runner import RunOptions,RunError,build_plan
    kernel={'name':'k','params':[['u64','output']],'compiled_by':'openassembler',
            'parameter_attributes':{'output':{'pointer':True,'space':'global','alignment':16}}}
    save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[{'source':'k.ptx','cubin':'k.cubin','kernels':[kernel]}])
    spec={'buffers':{'output':{'count':32}},'steps':[{'kernel':'k','args':[{'buf':'output','offset':4}]}]}
    path=tmp_path/'launch.json';path.write_text(json.dumps(spec))
    with pytest.raises(RunError,match='alignment'):build_plan(RunOptions(out_dir=tmp_path,spec=path,reps=0))


def test_pointer_modifier_spacing_and_default_pointed_alignment():
    k=parse_module(source('.ptr.global.align 16')).kernels['copy']
    assert k.parameter_attributes['output']['alignment']==16
    k=parse_module(source('.ptr.global')).kernels['copy']
    assert k.parameter_attributes['output']['alignment']==4


def test_native_ownership_contract_is_a_value_snapshot(tmp_path):
    from tests.native_fixture import attach_native_fixture
    from openassembler.native_ownership import validate
    from copy import deepcopy
    entry={'cubin':'k.cubin','kernels':[{'name':'k','params':[['u64','ptr']],
        'compiled_by':'openassembler','ptx_directives':{'maxntid':[32,1,1]}}]}
    attach_native_fixture(tmp_path,entry);binary=(tmp_path/'k.cubin').read_bytes()
    for key,value in [('params',[['u32','ptr']]),('ptx_directives',{'maxntid':[64,1,1]})]:
        changed=deepcopy(entry);changed['kernels'][0][key]=value
        with pytest.raises(ValueError,match='inventory'):validate(changed,binary)
    old=deepcopy(entry['native_ownership'])
    entry['kernels'][0]['params'][0][0]='u32'
    entry['kernels'][0]['ptx_directives']['maxntid'][0]=64
    assert entry['native_ownership']==old
