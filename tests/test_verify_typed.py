"""Typed checker contracts: full-width data/scalars, upper halves and bounded guards."""
import copy
import json

import pytest

from openassembler.backend import ra
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx, run_and_diff


def source(dtype, *, offset_store=False):
    width = 8 if dtype.endswith('64') else 4
    arithmetic = 'add.rn.f32' if dtype == 'f32' else 'add.' + dtype
    contract = {'schema': 1, 'kernel': 'typed', 'parameters': [
        {'name': 'input', 'kind': 'array', 'dtype': dtype},
        {'name': 'output', 'kind': 'array', 'dtype': dtype},
        {'name': 'value', 'kind': 'scalar', 'dtype': dtype},
        {'name': 'n', 'kind': 'scalar', 'dtype': 'u32'}]}
    text = '// openassembler-pyfront-types-v1 ' + json.dumps(contract) + '\n' + f'''.version 8.7
.target sm_80
.address_size 64
.visible .entry typed(.param .u64 input,.param .u64 output,.param .{dtype} value,.param .u32 n) {{
.reg .u64 %input,%output,%offset,%src,%dst;
.reg .{dtype} %value,%x; .reg .u32 %t,%n; .reg .pred %end;
ld.param.u64 %input,[input]; ld.param.u64 %output,[output]; ld.param.{dtype} %value,[value];
ld.param.u32 %n,[n]; mov.u32 %t,%tid.x; setp.ge.u32 %end,%t,%n; @%end bra DONE;
mul.wide.u32 %offset,%t,{width}; add.u64 %src,%input,%offset; add.u64 %dst,%output,%offset;
ld.global.{dtype} %x,[%src]; {arithmetic} %x,%x,%value;
st.global.{dtype} [%dst{('+'+str(width)) if offset_store else ''}],%x;
DONE: ret; }}
'''
    return text, contract


def compiled(dtype, **kwargs):
    text, contract = source(dtype, **kwargs)
    module = parse_module(text)
    code, stats = ra.compile_kernel(module.kernels['typed'])
    return text, contract, module, code, stats['labels']


@pytest.mark.parametrize('dtype', ['f32', 's32', 'u32', 's64', 'u64'])
@pytest.mark.parametrize('seed', [0, 3, 4, 6])
def test_explicit_roles_full_width_scalars_and_arrays(dtype, seed):
    text, contract, module, code, labels = compiled(dtype)
    assert parameter_contract_from_ptx(text, module.kernels['typed']) == contract
    assert run_and_diff(module, 'typed', code, labels, n_elems=19, seed=seed,
                        parameter_contract=contract) == []


def test_upper_word_only_corruption_is_observed():
    _, contract, module, code, labels = compiled('u64')
    # Read only the low word; the incorrect upper word must be compared too.
    broken = [(text.replace('LDG.E.64', 'LDG.E'), control) for text, control in code]
    assert broken != code
    mismatches = run_and_diff(module, 'typed', broken, labels, n_elems=19, parameter_contract=contract)
    assert any(m.param.startswith('output') and m.ours & 0xffffffff == m.expected & 0xffffffff
               and m.ours >> 32 != m.expected >> 32 for m in mismatches)


def test_same_out_of_bounds_write_in_both_engines_is_rejected():
    _, contract, module, code, labels = compiled('s64', offset_store=True)
    mismatches = run_and_diff(module, 'typed', code, labels, n_elems=8, parameter_contract=contract)
    assert any(m.param.endswith(':sass-guard') for m in mismatches)
    assert any(m.param.endswith(':ptx-guard') for m in mismatches)


@pytest.mark.parametrize('mutation', ['schema', 'kernel', 'order', 'dtype', 'array_abi', 'count', 'kind', 'extra'])
def test_malformed_metadata_cannot_fall_back_to_pointer_guessing(mutation):
    text, contract, module, _, _ = compiled('u64')
    damaged = copy.deepcopy(contract)
    if mutation == 'schema': damaged['schema'] = True
    if mutation == 'kernel': damaged['kernel'] = 'other'
    if mutation == 'order': damaged['parameters'][0:2] = reversed(damaged['parameters'][0:2])
    if mutation == 'dtype': damaged['parameters'][2]['dtype'] = 'f64'
    if mutation == 'array_abi': damaged['parameters'][2]['kind'] = 'array'; damaged['parameters'][2]['dtype'] = 's64'
    if mutation == 'count': damaged['parameters'].pop()
    if mutation == 'kind': damaged['parameters'][0]['kind'] = 'pointer'
    if mutation == 'extra': damaged['parameters'][0]['ignored'] = True
    if mutation == 'array_abi':
        # A scalar s64 ABI cannot be relabeled as a pointer array.
        text, _, module, _, _ = compiled('s64')
    # A tag naming a distinct kernel is absent for this entry; direct contracts
    # must still reject a mismatched kernel identity.
    if mutation == 'kernel':
        with pytest.raises(ValueError, match='contract'):
            run_and_diff(module, 'typed', [], {}, parameter_contract=damaged)
        malformed = '// openassembler-pyfront-types-v1 ' + json.dumps(damaged) + '\n' + text.split('\n', 1)[1]
        with pytest.raises(ValueError, match='contract'):
            parameter_contract_from_ptx(malformed, module.kernels['typed'])
    else:
        malformed = '// openassembler-pyfront-types-v1 ' + json.dumps(damaged) + '\n' + text.split('\n', 1)[1]
        with pytest.raises(ValueError, match='contract'):
            parameter_contract_from_ptx(malformed, module.kernels['typed'])


def test_duplicate_and_invalid_json_tags_reject():
    text, _, module, _, _ = compiled('u64')
    with pytest.raises(ValueError, match='duplicate'):
        parameter_contract_from_ptx(text.split('\n', 1)[0] + '\n' + text, module.kernels['typed'])
    with pytest.raises(ValueError, match='metadata'):
        parameter_contract_from_ptx('// openassembler-pyfront-types-v1 {broken}\n' + text, module.kernels['typed'])
    assert parameter_contract_from_ptx(text.split('\n', 1)[1], module.kernels['typed']) is None


def test_shaped_buffer_metadata_cannot_fall_back_to_generic_allocations():
    text, _, module, _, _ = compiled('u64')
    for metadata in ('{}', '{broken}'):
        with pytest.raises(ValueError, match='shaped buffer verification'):
            parameter_contract_from_ptx('// openassembler-pyfront-buffer-v1 ' + metadata + '\n' + text,
                                        module.kernels['typed'])
