"""Entry completeness, ABI and launch metadata through actual native builds."""
import json
import tempfile
from pathlib import Path

import pytest

from openassembler import build, oracle, project
from openassembler.cppfront import _kernel_names
from openassembler.native_ownership import validate
from openassembler.ptx.parse import entry_names, parse_module
from openassembler.runner import RunError, validate_launch

HEADER = '.version 8.7\n.target sm_80\n.address_size 64\n'


@pytest.mark.parametrize('prefix', ['', '.visible ', '.weak '])
def test_visibility_dollar_symbols_and_launch_headers(prefix):
    text = HEADER + prefix + '.entry $kernel$1(.param .u64 out) .reqntid 32, 2, 1 .maxnreg 64 { ret; }'
    kernel = parse_module(text).kernels['$kernel$1']
    assert kernel.params == [('u64', 'out')]
    assert kernel.directives == {'reqntid': (32, 2, 1), 'maxnreg': (64,)}
    assert [i.op for i in kernel.body] == ['ret']


def test_inventory_ignores_comments_strings_and_prototypes():
    text = HEADER + '''
// .entry comment() { ret; }
.file 1 "// source .entry fake() { ret; }"
.extern .entry prototype();
.entry bare() { ret; }
.weak .entry weak() .maxntid 128 { ret; }
.visible .entry visible() { ret; }
'''
    assert entry_names(text) == _kernel_names(text) == list(parse_module(text).kernels) == ['bare', 'weak', 'visible']
    assert parse_module(text).kernels['weak'].directives == {'maxntid': (128, 1, 1)}


@pytest.mark.parametrize('declaration,reason', [
    ('.entry k(.param .align 8 .b8 arg[16]) { ret; }', 'parameter ABI'),
    ('.entry k(.param .u32 x, .param .u32 x) { ret; }', 'duplicate parameter'),
    ('.entry k() .unknown 4 { ret; }', 'unsupported entry directive'),
    ('.entry k() .maxntid 0 { ret; }', 'invalid'),
    ('.entry k() .reqntid 2,2,2,2 { ret; }', 'dimensions'),
    ('.entry k() .maxntid 128 .reqntid 64 { ret; }', 'cannot be combined'),
    ('.entry k() .maxntid 128 .maxntid 64 { ret; }', 'duplicate'),
    ('.entry k() { ret; } .entry k() { ret; }', 'duplicate entry'),
    ('.entry k(.param .u64 out { ret; }', 'unbalanced parameter'),
    ('.entry k()', 'missing body'),
])
def test_unsupported_headers_fail_instead_of_losing_entry_or_abi(declaration, reason):
    with pytest.raises(SyntaxError, match=reason):
        parse_module(HEADER + declaration)


def test_launch_bound_product_and_exact_required_dimensions():
    validate_launch({'name': 'k', 'ptx_directives': {'maxntid': [128, 1, 1]}}, (16, 8, 1))
    with pytest.raises(RunError, match='exceeds .maxntid'):
        validate_launch({'name': 'k', 'ptx_directives': {'maxntid': [128, 1, 1]}}, (256, 1, 1))
    validate_launch({'name': 'k', 'ptx_directives': {'reqntid': [32, 2, 1]}}, (32, 2, 1))
    with pytest.raises(RunError, match='differs from .reqntid'):
        validate_launch({'name': 'k', 'ptx_directives': {'reqntid': [32, 2, 1]}}, (64, 1, 1))


def test_frontend_parser_inventory_mismatch_stops_before_codegen(tmp_path, monkeypatch):
    import openassembler.frontend as frontend
    source = tmp_path / 'input.ptx'
    source.write_text(HEADER + '.entry k() { ret; }')
    monkeypatch.setattr(frontend, 'to_ptx_module', lambda *a, **kw: (source.read_text(), ['k', 'lost'], []))
    monkeypatch.setattr(build, 'generate', lambda *a, **kw: pytest.fail('incomplete inventory reached codegen'))
    result = build.build_source(source, 'ptx', 'input', build.Options(), single=True)
    assert result.rc == 1 and 'entry inventory mismatch' in result.error
    assert not result.files


def test_register_limit_rejects_oa_output_before_donor_build(tmp_path, monkeypatch):
    source = tmp_path / 'limited.ptx'
    source.write_text(HEADER + '.entry k() .maxnreg 64 { ret; }')
    monkeypatch.setattr(build, 'generate', lambda *a, **kw: ([], {}, [], [], 65))
    monkeypatch.setattr(build, 'ptxas_build', lambda *a, **kw: pytest.fail('over-limit output reached donor build'))
    result = build.build_source(source, 'ptx', 'limited', build.Options(), single=True)
    assert result.rc == 1 and not result.files
    assert 'exceeds .maxnreg 64' in result.failed[0].error


@pytest.mark.oracle
def test_mixed_entries_native_build_preserves_metadata_and_integrity(tmp_path):
    if not oracle.available('ptxas'):
        pytest.skip('needs CUDA toolkit')
    source = tmp_path / 'mixed.ptx'
    source.write_text(HEADER + '''
.entry bare() { ret; }
.weak .entry weak() .maxntid 128, 1, 1 { ret; }
.visible .entry required() .reqntid 64, 1, 1 .maxnreg 64 { ret; }
''')
    output = tmp_path / 'out'
    result = build.compile_paths(source, output, build.Options(verify=True, no_sass=False))
    assert result.rc == 0
    entry = json.loads((output / 'manifest.json').read_text())['entries'][0]
    assert [k['name'] for k in entry['kernels']] == ['bare', 'weak', 'required']
    assert entry['kernels'][1]['ptx_directives'] == {'maxntid': [128, 1, 1]}
    assert entry['kernels'][2]['ptx_directives'] == {'reqntid': [64, 1, 1], 'maxnreg': [64]}
    binary = (output / entry['cubin']).read_bytes()
    validate(entry, binary)
    entry['kernels'][1]['ptx_directives']['maxntid'] = [256, 1, 1]
    with pytest.raises(ValueError, match='inventory'):
        validate(entry, binary)


def test_cuda_macro_local_headers_and_cycles_are_candidates(tmp_path):
    (tmp_path / 'inc').mkdir()
    (tmp_path / 'inc' / 'qualifier.cuh').write_text('#include "cycle.h"\n#define OA_GLOBAL __global__\n')
    (tmp_path / 'inc' / 'cycle.h').write_text('#include "qualifier.cuh"\n')
    source = tmp_path / 'kernel.cu'
    source.write_text('#include "inc/qualifier.cuh"\nOA_GLOBAL void k(float *p) { p[0] = 1; }\n')
    assert project.classify(source, source.name).language == 'cuda'
    assert project.classify(tmp_path / 'inc' / 'qualifier.cuh', 'inc/qualifier.cuh').language is None
    assert not project.has_cuda_kernel('#define OA_GLOBAL __global__\n')


def test_cuda_header_closure_stays_inside_source_tree(tmp_path):
    outside = tmp_path / 'outside.h'
    outside.write_text('#define OA_GLOBAL __global__\n')
    root = tmp_path / 'repo'
    root.mkdir()
    source = root / 'kernel.cu'
    source.write_text('#include "../outside.h"\nOA_GLOBAL void k() {}\n')
    assert project.classify(source, source.name).language is None


@pytest.mark.oracle
def test_macro_repository_compiles_through_ordinary_cli(tmp_path):
    from openassembler.cli import main
    if not oracle.available('nvcc'):
        pytest.skip('needs CUDA toolkit')
    # Colima cannot see pytest's macOS /private temporary directories.
    oracle.SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=oracle.SCRATCH) as temporary:
        root = Path(temporary)
        (root / 'qualifier.cuh').write_text('#define OA_GLOBAL __global__\n')
        (root / 'kernel.cu').write_text('#include "qualifier.cuh"\nextern "C" OA_GLOBAL void k(float *p) { p[0] = 1; }\n')
        assert main(['compile', str(root), '--verify']) == 0
        entry = json.loads((root / 'output' / 'manifest.json').read_text())['entries'][0]
        assert entry['source'] == 'kernel.cu'
        assert entry['kernels'][0]['compiled_by'] == 'openassembler'


@pytest.mark.parametrize('directives,expected', [({'reqntid': [32, 1, 1]}, 32),
                                                ({'maxntid': [16, 1, 1]}, 16)])
def test_automatic_runner_obeys_bound_before_submission(tmp_path, directives, expected):
    from tests.native_fixture import save_native_manifest
    from openassembler.runner import RunOptions, build_plan
    kernel = {'name': 'k', 'params': [['u64', 'out'], ['u32', 'n']],
              'compiled_by': 'openassembler', 'ptx_directives': directives}
    save_native_manifest(tmp_path, tmp_path, 'sm_80', 'opt', [
        {'source': 'k.ptx', 'cubin': 'k.cubin', 'kernels': [kernel]}])
    plan, _ = build_plan(RunOptions(out_dir=tmp_path, n=64, reps=0))
    assert plan.jobs[0]['steps'][0][3] == (expected, 1, 1)


def test_explicit_spec_cannot_bypass_required_dimensions(tmp_path):
    from tests.native_fixture import save_native_manifest
    from openassembler.runner import RunOptions, build_plan
    kernel = {'name': 'k', 'params': [], 'compiled_by': 'openassembler',
              'ptx_directives': {'reqntid': [32, 2, 1]}}
    save_native_manifest(tmp_path, tmp_path, 'sm_80', 'opt', [
        {'source': 'k.ptx', 'cubin': 'k.cubin', 'kernels': [kernel]}])
    spec = tmp_path / 'launch.json'
    spec.write_text(json.dumps({'buffers': {}, 'steps': [{'kernel': 'k', 'block': 64}]}))
    with pytest.raises(RunError, match='differs from .reqntid'):
        build_plan(RunOptions(out_dir=tmp_path, spec=spec, reps=0))
