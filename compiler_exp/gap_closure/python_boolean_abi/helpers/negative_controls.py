"""Logical predicates do not implicitly widen helper effects or physical ABIs."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file, NotSupported

CASES = [
    ('integer_argument', '@device\ndef h(x:bool)->bool:\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(1))\n', 'expected pred, got s32'),
    ('numeric_argument', '@device\ndef h(x:bool)->bool:\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(a[0]))\n', 'expected pred, got s32'),
    ('float_argument', '@device\ndef h(x:bool)->bool:\n    return x\n@kernel\ndef k(a:Array(f32)):\n    a[0]=float(i32(h(a[0])))\n', 'expected pred, got f32'),
    ('numeric_return', '@device\ndef h(x:bool)->bool:\n    return 1\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'returns s32, annotated pred'),
    ('mixed_return', '@device\ndef h(x:bool)->bool:\n    if x:\n        return True\n    return 0\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'returns s32, annotated pred'),
    ('missing_return', '@device\ndef h(x:bool)->bool:\n    if x:\n        return True\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'return on every path'),
    ('bool_to_integer', '@device\ndef h(x:i32)->i32:\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(True)\n', 'expected s32, got pred'),
    ('bool_array_argument', '@device\ndef h(x:Array(bool))->bool:\n    return True\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'unsupported type annotation'),
    ('bool_array_return', '@device\ndef h(x:bool)->Array(bool):\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'unsupported type annotation'),
    ('kernel_parameter', '@kernel\ndef k(a:Array(i32),flag:bool):\n    a[0]=i32(flag)\n', 'unsupported type annotation'),
    ('array_parameter', '@kernel\ndef k(a:Array(bool)):\n    a[0]=True\n', 'unsupported type annotation'),
    ('buffer_parameter', '@kernel\ndef k(a:Buffer(bool,(4,))):\n    a[0]=True\n', 'unsupported type annotation'),
    ('shared_storage', '@kernel\ndef k(a:Array(i32)):\n    data=shared_array(4,bool)\n    a[0]=1\n', 'unsupported type annotation'),
    ('memory_helper', '@device\ndef h(x:bool)->bool:\n    return bool(a[0])\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'cannot access memory'),
    ('atomic_helper', '@device\ndef h(x:bool)->bool:\n    return bool(atomic_add(a,0,1))\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'pure scalar calls'),
    ('barrier_helper', '@device\ndef h(x:bool)->bool:\n    syncthreads()\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'pure scalar calls'),
    ('capture', '@device\ndef h(x:bool)->bool:\n    return flag\n@kernel\ndef k(a:Array(i32)):\n    flag=True\n    a[0]=i32(h(flag))\n', 'uninitialized or captured'),
    ('uninitialized', '@device\ndef h(x:bool)->bool:\n    if x:\n        flag=True\n    return flag\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'uninitialized'),
    ('recursion', '@device\ndef h(x:bool)->bool:\n    return h(x)\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'recursive'),
    ('keyword_call', '@device\ndef h(x:bool)->bool:\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(x=True))\n', 'keyword'),
    ('boolean_union', '@device\ndef h(x:bool)->bool:\n    return x or 7\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'one fixed scalar kind'),
    ('shadow_annotation', 'bool=i32\n@device\ndef h(x:bool)->bool:\n    return x\n@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(h(True))\n', 'unsupported type annotation'),
]

if __name__ == '__main__':
    rows = []
    with tempfile.TemporaryDirectory(prefix='oa-boolean-helper-negative-') as temporary:
        for name, body, reason in CASES:
            source = 'from openassembler.pyfront import *\n'+body
            path = Path(temporary)/(name+'.py')
            path.write_text(source)
            try:
                compile_file(path)
            except NotSupported as error:
                assert reason in str(error), (name, reason, str(error))
                assert error.path and error.lineno
                rows.append(dict(name=name, source=source, reason=reason, rejected=True, diagnostic=str(error)))
            else:
                raise AssertionError(name)
    paths = [HERE/n for n in ('candidate-pyfront.py', 'candidate-helpers.py', 'candidate-modules.py', 'negative_controls.py')]
    report = dict(schema=1, passed=True, results=rows,
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope='22 original helper type/effect/source/initialization and physical Boolean ABI rejection controls; no GPU claim')
    (HERE/'negative-controls.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Passed 22 original logical helper boundaries')
