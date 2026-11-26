"""Isolated Boolean value type/effect/ABI boundaries; no production change."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
CASES=[
 ('mixed_value','@kernel\ndef k(a:Array(i32),x:f32):\n    value=a[0]or x\n    a[0]=i32(value)\n','one fixed scalar kind'),
 ('bool_numeric_union','@kernel\ndef k(a:Array(i32)):\n    value=True or a[0]\n    a[0]=i32(value)\n','one fixed scalar kind'),
 ('literal_context','@kernel\ndef k(a:Array(i64)):\n    value=0 or a[0]\n    a[0]=value\n','one fixed scalar kind'),
 ('bool_arity','@kernel\ndef k(a:Array(i32)):\n    value=bool()\n    a[0]=i32(value)\n','exactly one scalar argument'),
 ('bool_keywords','@kernel\ndef k(a:Array(i32)):\n    value=bool(x=a[0])\n    a[0]=i32(value)\n','keyword arguments'),
 ('bool_arrays','@kernel\ndef k(a:Array(i32)):\n    value=bool(a)\n    a[0]=i32(value)\n','used as a value'),
 ('bool_parameter','@kernel\ndef k(a:Array(i32),value:bool):\n    a[0]=i32(value)\n','annotation'),
 ('helper_memory','@device\ndef h(x:i32)->i32:\n    return i32(bool(a[0]))\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(3)\n','cannot access memory'),
 ('uninitialized','@device\ndef h(x:i32)->i32:\n    if x:\n        flag=True\n    return i32(flag)\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(3)\n','uninitialized'),
 ('varying_barrier','@kernel\ndef k(a:Array(i32)):\n    flag=bool(tid_x())\n    if flag:\n        syncthreads()\n','thread-dependent'),
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
 report=dict(schema=1,passed=True,results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-pyfront.py',Path(__file__))},scope='Ten isolated mixed-kind/literal-context/Boolean ABI/array/arity/keyword/helper-memory/initialization/convergence boundaries; original diagnostic shape and promotion/hardware remain separate work')
 (HERE/'negative-controls.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed ten isolated Boolean type/effect/ABI boundaries')
