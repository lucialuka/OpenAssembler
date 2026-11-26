"""Structured helper control flow must preserve values and reject missing definitions."""
from __future__ import annotations

import pytest
from pyfront_threeway import Pending, run_ptx_and_sass
from openassembler.pyfront import NotSupported, compile_file, find_kernel

HEADER='from openassembler.pyfront import Array, i32, f32, kernel, device, tid_x, syncthreads\n'


def file(tmp_path, helper):
    path=tmp_path/'control.py'
    path.write_text(HEADER+helper)
    return path


def test_helper_branch_loop_values_match_python(tmp_path):
    source='''
@device
def value(x: i32, n: i32) -> i32:
    total = 0
    for j in range(n, 0, -1):
        if j == 3:
            continue
        if j == 1:
            break
        total += j
    left = n
    while left > 0:
        left -= 1
        if left == 2:
            continue
        if left == 0:
            break
        total += left
    if x < 0:
        y = -x
    elif x == 0:
        y = 7
    else:
        y = x + 2
    return y + total
@kernel
def evaluate(a: Array(i32), out: Array(i32)):
    i = tid_x()
    total = 99
    out[i] = value(a[i], i % 6) + a[i] + total
'''
    path=file(tmp_path,source)
    # Independently execute the identical helper body as ordinary Python.
    namespace={};exec(source[source.index('def value'):source.index('@kernel')],namespace)
    values=[-20,-7,-1,0,1,5,12,30]
    expected={'a':values,'out':[namespace['value'](x,i%6)+x+99 for i,x in enumerate(values)]}
    got,encoded=run_ptx_and_sass(find_kernel(path,None),{'a':values,'out':[-999]*8},{},8)
    assert got==expected
    assert not isinstance(encoded,Pending),str(encoded)
    assert encoded==expected


@pytest.mark.parametrize('body,diagnostic',[
    ('if x:\n    y = 1\nreturn y','may be uninitialized'),
    ('for j in range(x):\n    y = 1\nreturn y','may be uninitialized'),
    ('for j in range(x):\n    pass\nreturn j','may be uninitialized'),
    ('while x > 0:\n    y = x\n    x -= 1\nreturn y','may be uninitialized'),
    ('if x:\n    y = 1\nelse:\n    y += 1\nreturn y','may be uninitialized'),
    ('y = 1\nif x:\n    return y\nreturn x','one terminal return'),
    ('y = 1\nfor j in range(x):\n    pass\nelse:\n    y += 1\nreturn y','loop else'),
    ('y = 1\nwhile x:\n    x -= 1\nelse:\n    y += 1\nreturn y','loop else'),
    ('break\nreturn x','outside a loop'),
    ('continue\nreturn x','outside a loop'),
    ('y = x\nif x:\n    y = 1.0\nreturn y','different type'),
    ('while x:\n    syncthreads()\n    x -= 1\nreturn x','pure scalar calls'),
    ('range = 2\nfor j in range(x):\n    x += j\nreturn x','local value'),
])
def test_control_flow_rejects_unsafe_or_undeclared_forms(tmp_path,body,diagnostic):
    body='\n'.join('    '+line for line in body.splitlines())
    path=file(tmp_path,f'@device\ndef helper(x: i32) -> i32:\n{body}\n@kernel\ndef evaluate(out: Array(i32)):\n    out[0] = helper(2)\n')
    with pytest.raises(NotSupported,match=diagnostic) as error:compile_file(path)
    assert error.value.path==str(path) and error.value.lineno is not None


def test_helper_loop_cannot_hide_divergent_barrier(tmp_path):
    path=file(tmp_path,'''@device
def helper(x: i32) -> i32:
    while x > 0:
        x -= 1
    return x
@kernel
def evaluate():
    if helper(tid_x()):
        syncthreads()
''')
    with pytest.raises(NotSupported,match='thread-dependent control flow'):compile_file(path)


def test_initialized_loop_target_preserves_empty_loop_value(tmp_path):
    path=file(tmp_path,'''@device
def helper(x: i32) -> i32:
    j = 19
    total = 0
    for j in range(x):
        total += j
        j = 100
    return j + total
@kernel
def evaluate(out: Array(i32)):
    out[tid_x()] = helper(tid_x())
''')
    expected={'out':[19]+[100+sum(range(i)) for i in range(1,8)]}
    interpreted,encoded=run_ptx_and_sass(find_kernel(path,None),{'out':[-999]*8},{},8)
    assert interpreted==expected
    assert not isinstance(encoded,Pending),str(encoded)
    assert encoded==expected
