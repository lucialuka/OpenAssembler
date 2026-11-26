"""Independent bit snapshots for wide integer/FP64 selection and support limits."""
import json
from pathlib import Path
import struct
import pytest
from openassembler import build, oracle
from openassembler.backend import ra
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Interp, Launch as PLaunch
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing

CASES = json.loads((Path(__file__).parent/'data/wide_select_fixtures.json').read_text())

@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
@pytest.mark.parametrize('level', ['safe', 'opt'])
def test_complete_wide_selection_bits_alias_guard_and_source_snapshots(case, level):
    module = parse_module(case['source'])
    kernel = module.kernels[case['entry']]
    code, labels, words, _, _ = build.generate(kernel, 'sm_80', level)
    assert words
    inp, out, choice = 0x100010000, 0x100000000, 0x8123456776543210
    guard = b'\xa5'*64
    before, wanted = bytes.fromhex(case['input_hex']), bytes.fromhex(case['expected_hex'])
    params = [out, inp, 91] + ([] if case['mode'] == 'cuda' else [choice])
    raw = struct.pack('<QQI', *params) if len(params) == 3 else struct.pack('<QQI4xQ', *params)
    for engine in ('ptx', 'normal', 'poisoned'):
        memory = Memory()
        memory.write_bytes(inp-64, guard+before+guard)
        memory.write_bytes(out-64, guard+struct.pack('<Q', 0x13579bdf2468ace0)*128+guard)
        if engine == 'ptx':
            mapping = dict(zip((name for _, name in kernel.params), params))
            Interp(module, case['entry'], PLaunch(block=(128, 1, 1), params=mapping), memory).run()
        else:
            Machine(load_listing(listing(code, labels)), Launch(block=(128, 1, 1), params=raw, poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
        assert bytes(memory.read(out-64+i, 1) for i in range(len(wanted)+128)) == guard+wanted+guard
        assert bytes(memory.read(inp-64+i, 1) for i in range(len(before)+128)) == guard+before+guard

@pytest.mark.parametrize('instruction', [
    'selp.ftz.b64 %r,%a,%b,%take;',
    'selp.rn.f64 %r,%a,%b,%take;',
    'selp.b16 %r,%a,%b,%take;',
    'selp.v2.b64 %r,%a,%b,%take;',
    'selp.b64 %r,%a,%b;',
    'selp.b64 %r,{%a,%b},%b,%take;',
    'selp.b64 %r,%missing,%b,%take;',
    'selp.b64 %r,%a,%b,%missing;',
    'selp.b64 %r,%a,%b,%a;',
    'add.f64 %r,%a,%b;',
])
def test_unvalidated_modifiers_types_operand_shapes_and_predicates_reject(instruction):
    case = next(c for c in CASES if c['name'] == 'b64_register')
    source = case['source'].replace('selp.b64 %r,%a,%b,%take;', instruction)
    with pytest.raises(ra.NotSupported):
        build.generate(parse_module(source).kernels['probe'], 'sm_80', 'safe')

def test_wrong_float_literal_width_rejects_in_parser():
    case = next(c for c in CASES if c['name'] == 'b64_register')
    source = case['source'].replace('selp.b64 %r,%a,%b,%take;', 'selp.b64 %r,0f3f800000,%b,%take;')
    with pytest.raises(SyntaxError, match='floating literal width 32 is invalid for selp.b64'):
        parse_module(source)

@pytest.mark.oracle
def test_normal_builder_owns_wide_selection_binary(tmp_path):
    if not oracle.available('ptxas'):
        pytest.skip('needs CUDA toolkit')
    case = next(c for c in CASES if c['name'] == 'f64_false_guard')
    source, target = tmp_path/'select.ptx', tmp_path/'out'
    source.write_text(case['source'])
    assert build.compile_paths(source, target, build.Options(no_sass=True, opt_level='opt')).rc == 0
    entry, = json.loads((target/'manifest.json').read_text())['entries']
    from openassembler.native_ownership import validate
    validate(entry, (target/entry['cubin']).read_bytes())

@pytest.mark.oracle
def test_original_cuda_noinline_source_builds_through_normal_frontend(tmp_path):
    if not oracle.available('nvcc'):
        pytest.skip('needs CUDA toolkit')
    case = next(c for c in CASES if c['name'] == 'cuda_noinline')
    source, target = tmp_path/'select.cu', tmp_path/'out'
    source.write_text(case['cuda_source'])
    assert build.compile_paths(source, target, build.Options(no_sass=True, opt_level='opt')).rc == 0
    entry, = json.loads((target/'manifest.json').read_text())['entries']
    from openassembler.native_ownership import validate
    validate(entry, (target/entry['cubin']).read_bytes())
    assert set(entry['kernels'][0]['inlined_device_calls']['functions']) == {'mix', 'next'}
