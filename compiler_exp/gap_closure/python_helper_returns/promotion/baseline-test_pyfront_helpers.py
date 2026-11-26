"""Independent numeric and rejection contracts for statically inlined device helpers."""
from __future__ import annotations

import pytest

from openassembler.pyfront import NotSupported, compile_file, compile_kernel, device, find_kernel
from pyfront_threeway import Pending, run_ptx_and_sass


HEADER = 'from openassembler.pyfront import Array, i32, u32, f32, device, kernel, tid_x\n'


def source_file(tmp_path, source):
    path = tmp_path / 'helpers.py'
    path.write_text(HEADER + source)
    return path


def test_nested_helpers_match_independent_python_and_encoded_emulation(tmp_path):
    path = source_file(tmp_path, '''
raise RuntimeError('compilation must never execute this')
@device
def twice(x: i32) -> i32:
    x += 3
    return x * 2
@device
def combine(x: i32) -> i32:
    z = twice(x)
    return z + twice(x)
@kernel
def evaluate(a: Array(i32), out: Array(i32), n: i32):
    i = tid_x()
    if i >= n:
        return
    x = a[i]
    z = 99
    out[i] = combine(x) + twice(x) + x + z
''')
    values = [-7, -1, 0, 1, 5, 7, 12, 30]
    arrays = {'a': values, 'out': [-999] * 8}
    reference = {'a': values, 'out': [7 * x + 117 for x in values[:6]] + [-999] * 2}
    fn = find_kernel(path, 'evaluate')
    ptx, names = compile_file(path)
    assert names == ['evaluate']
    assert compile_kernel(fn) == ptx
    assert '.func' not in ptx and ' call' not in ptx
    interpreted, encoded = run_ptx_and_sass(fn, arrays, {'n': 6}, 8)
    assert interpreted == reference
    assert not isinstance(encoded, Pending), str(encoded)
    assert encoded == reference


def test_float_helpers_use_annotated_literals_and_preserve_caller(tmp_path):
    path = source_file(tmp_path, '''
@device
def affine(x: f32, gain: f32, bias: f32) -> f32:
    x *= gain
    return x + bias
@kernel
def evaluate(a: Array(f32), out: Array(f32)):
    i = tid_x()
    x = a[i]
    out[i] = affine(x, 2, 0.5) + affine(x, 0.5, -1.0) + x
''')
    values = [-4.0, -1.5, -0.0, 0.0, 0.25, 1.0, 4.0, 16.0]
    arrays = {'a': values, 'out': [0.0] * 8}
    reference = {'a': values, 'out': [x * 3.5 - 0.5 for x in values]}
    interpreted, encoded = run_ptx_and_sass(find_kernel(path, None), arrays, {}, 8)
    assert interpreted == reference
    assert not isinstance(encoded, Pending), str(encoded)
    assert encoded == reference


@pytest.mark.parametrize('imports,decorator', [
    ('from openassembler.pyfront import device as scalar', '@scalar'),
    ('import openassembler.pyfront as pf', '@pf.device'),
    ('import openassembler.pyfront', '@openassembler.pyfront.device'),
])
def test_device_import_aliases_and_late_definitions(tmp_path, imports, decorator):
    path = source_file(tmp_path, f'''
{imports}
@kernel
def evaluate(out: Array(i32)):
    out[0] = add(2)
{decorator}
def add(x: i32) -> i32:
    return x + 1
''')
    ptx, names = compile_file(path)
    assert names == ['evaluate'] and 'add.s32' in ptx


@pytest.mark.parametrize('helper,call,diagnostic', [
    ('def h(x) -> i32:\n    return x', 'h(1)', 'scalar annotation'),
    ('def h(x: i32):\n    return x', 'h(1)', 'return annotation'),
    ('def h(x: Array(i32)) -> i32:\n    return 0', 'h(1)', 'scalar parameters'),
    ('def h(x: i32) -> Array(i32):\n    return x', 'h(1)', 'return scalars'),
    ('def h(x: i32 = 1) -> i32:\n    return x', 'h(1)', 'without defaults'),
    ('def h(*, x: i32) -> i32:\n    return x', 'h(1)', 'plain positional'),
    ('def h(x: i32) -> i32:\n    x += 1', 'h(1)', 'terminal return'),
    ('def h(x: i32) -> i32:\n    if x:\n        return 1\n    return 0', 'h(1)', 'scalar assignments'),
    ('def h(x: i32) -> i32:\n    return h(x)', 'h(1)', 'recursive device helper'),
    ('def h(x: i32) -> i32:\n    return g(x)\n@device\ndef g(x: i32) -> i32:\n    return h(x)', 'h(1)', 'h -> g -> h'),
    ('def h(x: i32) -> i32:\n    return x', 'h(1.0)', 'expected s32, got f32'),
    ('def h(x: i32) -> f32:\n    return x', 'h(1)', 'returns s32, annotated f32'),
    ('def h(x: i32) -> i32:\n    return x', 'h(1, 2)', 'expects 1 arguments'),
    ('def h(x: i32) -> i32:\n    return x', 'h(x=1)', 'keyword arguments'),
    ('def h(x: i32) -> i32:\n    return x + hidden', 'h(1)', 'uninitialized or captured'),
    ('def h(x: i32) -> i32:\n    return source[0]', 'h(1)', 'cannot access memory'),
    ('def h(x: i32) -> i32:\n    return tid_x() + x', 'h(1)', 'pure scalar calls'),
    ('def h(x: i32) -> i32:\n    return arbitrary(x)', 'h(1)', 'not supported'),
    ('def h(x: i32) -> i32:\n    abs = 1\n    return abs(x)', 'h(1)', 'local value'),
    ('async def h(x: i32) -> i32:\n    return x', 'h(1)', 'async device helpers'),
])
def test_helpers_fail_closed_with_source_diagnostics(tmp_path, helper, call, diagnostic):
    path = source_file(tmp_path, f'''
@device
{helper}
@kernel
def evaluate(out: Array(i32)):
    out[0] = {call}
''')
    with pytest.raises(NotSupported, match=diagnostic) as exc:
        compile_file(path)
    assert str(path) in str(exc.value)
    assert exc.value.lineno is not None


@pytest.mark.parametrize('decoration', ['@device()', '@kernel\n@device'])
def test_malformed_device_decorators_reject(tmp_path, decoration):
    path = source_file(tmp_path, f'''
{decoration}
def h(x: i32) -> i32:
    return x
@kernel
def evaluate(out: Array(i32)):
    out[0] = h(1)
''')
    with pytest.raises(NotSupported, match='@device takes no arguments or additional decorators'):
        compile_file(path)


def test_rebound_helper_and_local_shadow_reject(tmp_path):
    helper = '@device\ndef h(x: i32) -> i32:\n    return x\n'
    path = source_file(tmp_path, helper + 'h = 1\n@kernel\ndef evaluate():\n    pass\n')
    with pytest.raises(NotSupported, match='rebound at module scope'):
        compile_file(path)
    path = source_file(tmp_path, helper + '''
@kernel
def evaluate(out: Array(i32)):
    out[0] = h(1)
    h = 1
''')
    with pytest.raises(NotSupported, match='local value'):
        compile_file(path)
    path = source_file(tmp_path, '''
import math
@device
def h(x: i32) -> i32:
    math = 1
    return int(math.sqrt(float(x)))
@kernel
def evaluate(out: Array(i32)):
    out[0] = h(1)
''')
    with pytest.raises(NotSupported, match='local value'):
        compile_file(path)
    path = source_file(tmp_path, helper + 'if False:\n    h = 1\n')
    with pytest.raises(NotSupported, match='rebound at module scope') as exc:
        compile_file(path)
    assert str(path) in str(exc.value)


def test_device_runtime_calls_reject():
    @device
    def scalar(x):
        return x
    with pytest.raises(NotSupported, match='inside a compiled @kernel'):
        scalar(1)


def test_helper_specialization_has_an_expansion_budget(tmp_path):
    source = '@device\ndef h0(x: i32) -> i32:\n    return x\n'
    for i in range(1, 12):
        source += f'@device\ndef h{i}(x: i32) -> i32:\n    return h{i-1}(x) + h{i-1}(x)\n'
    source += '@kernel\ndef evaluate(out: Array(i32)):\n    out[0] = h11(1)\n'
    with pytest.raises(NotSupported, match='expansion exceeds 1024 call sites'):
        compile_file(source_file(tmp_path, source))
