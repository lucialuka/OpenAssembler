"""Isolated Boolean value type/effect/ABI boundaries; no production change."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
CASES=[
 ('mixed_value','@kernel\ndef k(a:Array(i32),x:f32):\n    value=a[0]or x\n    a[0]=i32(value)\n','one fixed scalar kind'),
 ('bool_numeric_union','@kernel\ndef k(a:Array(i32)):\n    value=True or a[0]\n    a[0]=i32(value)\n','one fixed scalar kind'),
 ('bool_arity','@kernel\ndef k(a:Array(i32)):\n    value=bool()\n    a[0]=i32(value)\n','exactly one scalar argument'),
 ('bool_keywords','@kernel\ndef k(a:Array(i32)):\n    value=bool(x=a[0])\n    a[0]=i32(value)\n','keyword arguments'),
 ('bool_arrays','@kernel\ndef k(a:Array(i32)):\n    value=bool(a)\n    a[0]=i32(value)\n','used as a value'),
 ('bool_parameter','@kernel\ndef k(a:Array(i32),value:bool):\n    a[0]=i32(value)\n','annotation'),
 ('helper_memory','@device\ndef h(x:i32)->i32:\n    return i32(bool(a[0]))\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(3)\n','cannot access memory'),
 ('uninitialized','@device\ndef h(x:i32)->i32:\n    if x:\n        flag=True\n    return i32(flag)\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(3)\n','uninitialized'),
 ('varying_barrier','@kernel\ndef k(a:Array(i32)):\n    flag=bool(tid_x())\n    if flag:\n        syncthreads()\n','thread-dependent'),
 ('bool_arithmetic','@kernel\ndef k(a:Array(i32)):\n    value=True+False\n    a[0]=i32(value)\n','arithmetic on a comparison result'),
 ('bool_store','@kernel\ndef k(a:Array(i32)):\n    value=not a[0]\n    a[0]=value\n','storing'),
 ('wide_union','@kernel\ndef k(a:Array(i32)):\n    value=a[0] or i64(7)\n    a[0]=i32(value)\n','one fixed scalar kind'),
 ('inexact_context','@kernel\ndef k(a:Array(f32)):\n    a[0]=16777217 or a[0]\n','exactly representable'),
 ('unsigned_context','@kernel\ndef k(a:Array(u64)):\n    a[0]=-1 or a[0]\n','fit in u64'),
 ('bool_named_shadow','@kernel\ndef k(a:Array(i32)):\n    bool=7\n    a[0]=i32(bool(a[0]))\n','local value'),
 ('float_integer_cast','@kernel\ndef k(a:Array(i32)):\n    a[0]=i32(1.5)\n','integer values'),
]
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-boolean-negative-')as temporary:
  for name,source,reason in CASES:
   path=Path(temporary)/(name+'.py');path.write_text('from openassembler.pyfront import *\n'+source)
   try:compile_file(path)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error));rows.append(dict(name=name,source=source,rejected=True,diagnostic=str(error)))
   else:raise AssertionError(name)
 report=dict(schema=1,passed=True,results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-pyfront.py',HERE/'candidate-backend.py',HERE/'candidate-modules.py',Path(__file__))},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='16 original fixed-kind/literal/Boolean ABI/array/arity/keyword/helper-memory/initialization/convergence/arithmetic/conversion/shadowing boundaries with explicit type/effect limits; hardware separate')
 (HERE/'negative-controls.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed sixteen Boolean type/effect/ABI boundaries')
