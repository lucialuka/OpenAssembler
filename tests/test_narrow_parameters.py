"""Independent scalar byte layout, extraction, extension, effects and launch contracts."""
import importlib
import json
import struct
from pathlib import Path
import pytest
from openassembler import build, oracle
from openassembler.backend import ra
from openassembler.ptx.parse import parse_module, parameter_layout
from openassembler.ptx.interp import Interp, Launch as PLaunch
from openassembler.emu.machine import Memory, Machine, Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

DATA = json.loads((Path(__file__).parent/'data/narrow_parameter_fixtures.json').read_text())


@pytest.mark.parametrize('case', DATA['cases'], ids=lambda row: row['name'])
@pytest.mark.parametrize('level', ['safe', 'opt'])
def test_original_narrow_slots_extensions_guard_loop_alias_and_subword_snapshot(case, level):
    module = parse_module(case['source'])
    kernel = module.kernels[case['name']]
    code, labels, words, _, registers = build.generate(kernel, 'sm_80', level)
    assert words and registers >= 16
    pointer = case['values']['out']
    guard = b'\xa5'*64
    wanted = guard+bytes.fromhex(case['expected_hex'])+guard
    for engine in ('ptx', 'normal', 'poisoned'):
        memory = Memory()
        memory.write_bytes(pointer-64, guard+bytes.fromhex(case['initial_hex'])+guard)
        if engine == 'ptx':
            Interp(module, case['name'], PLaunch(block=(64, 1, 1), params=case['values']), memory).run()
        else:
            Machine(load_listing(listing(code, labels)), Launch(block=(64, 1, 1), params=bytes.fromhex(case['packed_hex']), poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
        assert bytes(memory.read(pointer-64+i, 1) for i in range(len(wanted))) == wanted


@pytest.mark.parametrize('case', DATA['negatives'], ids=lambda row: row['name'])
def test_unimplemented_aggregate_width_offset_space_or_device_parameter_rejects(case):
    with pytest.raises((SyntaxError, ra.NotSupported, ValueError)):
        module = parse_module(case['source'])
        for kernel in module.kernels.values():build.generate(kernel, 'sm_80', 'safe')


def test_adjacent_scalar_layout_has_actual_byte_slots_and_preserves_wide_alignment():
    params = [('u8','a'),('u8','b'),('u16','c'),('u32','d'),('u8','e'),('u64','ptr'),('u16','last')]
    expected = dict(a=(0,1),b=(1,1),c=(2,2),d=(4,4),e=(8,1),ptr=(16,8),last=(24,2))
    assert parameter_layout(params) == expected
    from openassembler.backend.compile import Compiler
    kernel = parse_module('.entry k('+','.join('.param .'+kind+' '+name for kind,name in params)+') {ret;}').kernels['k']
    wanted = {name:0x160+offset for name,(offset,size) in expected.items()}
    assert ra.Selector(kernel).param_off == Compiler(kernel).param_off == wanted


@pytest.mark.parametrize('dtype', ['b8','u8','s8','b16','u16','s16'])
@pytest.mark.parametrize('backend_name', ['compile','naive'])
def test_legacy_selector_narrow_signed_and_unsigned_snapshots(dtype, backend_name):
    backend = importlib.import_module('openassembler.backend.'+backend_name)
    from openassembler.isa.encode import Encoder
    width = int(dtype[1:])//8
    params = '.param .u8 prefix,.param .'+dtype+' v,.param .u8 neighbor,.param .u64 out'
    text = '.entry k('+params+') {.reg .u32 %v;.reg .u64 %ptr;ld.param.'+dtype+' %v,[v];ld.param.u64 %ptr,[out];st.global.u32 [%ptr],%v;ret;}'
    kernel = parse_module(text).kernels['k']
    scheduled = backend.schedule(backend.Compiler(kernel).compile())
    for instruction, control in scheduled:Encoder('sm_80').encode(instruction, control)
    raw = (1 << (width*8))-1
    packed = bytearray([0xa5])
    packed += bytes((-len(packed)) % width)+raw.to_bytes(width, 'little')+b'\xc3'
    packed += bytes((-len(packed)) % 8)+(0x100000000).to_bytes(8,'little')
    wanted = 0xffffffff if dtype.startswith('s') else raw
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        Machine(load_listing(listing(scheduled, {})), Launch(block=(1,1,1), params=bytes(packed), poison=poison), memory).run()
        assert memory.read(0x100000000,4) == wanted


@pytest.mark.parametrize('dtype', ['b8','u8','s8','b16','u16','s16'])
def test_generic_verifier_uses_matching_narrow_packing_and_extension(dtype):
    from openassembler.verify import run_and_diff
    source = '.entry k(.param .u8 neighbor,.param .'+dtype+' v,.param .u64 out){.reg .u32 %v;.reg .u64 %p;ld.param.'+dtype+' %v,[v];ld.param.u64 %p,[out];st.global.u32 [%p],%v;ret;}'
    module = parse_module(source)
    code,labels,words,_,regs = build.generate(module.kernels['k'],'sm_80','opt')
    for seed in range(8):assert run_and_diff(module,'k',code,labels,n_elems=8,seed=seed) == []


def test_generic_verifier_rejects_unrepresentable_narrow_structural_dimension():
    from openassembler.verify import run_and_diff
    module = parse_module('.entry k(.param .s8 n){ret;}')
    code,labels,words,_,regs = build.generate(module.kernels['k'],'sm_80','safe')
    with pytest.raises(ValueError,match='does not fit narrow'):run_and_diff(module,'k',code,labels,n_elems=256)


def runner_plan(tmp_path, dtype, argument):
    from tests.native_fixture import save_native_manifest
    from openassembler.runner import RunOptions, build_plan
    kernel = dict(name='k',params=[[dtype,'value']],compiled_by='openassembler')
    save_native_manifest(tmp_path,tmp_path,'sm_80','opt',[dict(source='k.ptx',cubin='k.cubin',kernels=[kernel])])
    spec = dict(buffers={'out':dict(count=1,init='zeros')},steps=[dict(kernel='k',args=[argument])],compare=['out'],reps=0)
    path = tmp_path/'launch.json'
    path.write_text(json.dumps(spec))
    return build_plan(RunOptions(out_dir=tmp_path,spec=path,reps=0))[0]


@pytest.mark.parametrize('dtype,argument,expected',[
    ('s8',{'i8':-128},('u8',128)),('u8',{'u8':255},('u8',255)),('b8',{'i8':-1},('u8',255)),
    ('s16',{'i16':-32768},('u16',32768)),('u16',{'u16':65535},('u16',65535)),('b16',{'u16':49152},('u16',49152)),
])
def test_explicit_run_spec_preserves_exact_scalar_launch_width(tmp_path,dtype,argument,expected):
    plan = runner_plan(tmp_path,dtype,argument)
    assert plan.jobs[0]['steps'][0][-1] == [expected]


@pytest.mark.parametrize('dtype,argument',[
    ('s8',{'i8':-129}),('u8',{'u8':256}),('u8',{'u8':True}),('s16',{'i16':-32769}),
    ('u16',{'u16':65536}),('s8',{'u32':1}),('u16',{'u8':1}),('u32',{'u16':1}),
    ('u8',{'buf':'out'}),('u8',{'u8':1,'offset':0}),
])
def test_explicit_run_spec_rejects_narrow_range_and_width_mismatch(tmp_path,dtype,argument):
    from openassembler.runner import RunError
    with pytest.raises(RunError):runner_plan(tmp_path,dtype,argument)


@pytest.mark.oracle
def test_normal_build_verification_and_native_ownership_bind_narrow_signature(tmp_path):
    if not oracle.available('ptxas'):pytest.skip('needs CUDA toolkit')
    case = DATA['cases'][0]
    source = tmp_path/'narrow.ptx'
    source.write_text(case['source'])
    output = tmp_path/'output'
    assert build.compile_paths(source,output,build.Options(no_sass=True,verify=True)).rc == 0
    entry, = json.loads((output/'manifest.json').read_text())['entries']
    assert entry['kernels'][0]['verify'] == 'ok'
    from openassembler.native_ownership import validate
    binary = (output/entry['cubin']).read_bytes()
    validate(entry,binary)
    entry['kernels'][0]['params'][0][0] = 'u32'
    with pytest.raises(ValueError,match='inventory'):validate(entry,binary)
