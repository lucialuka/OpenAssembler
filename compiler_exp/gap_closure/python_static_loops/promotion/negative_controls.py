"""Tuple iteration rejection and convergence/resource contracts before promotion."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
from openassembler.ptx.parse import parse_module
from openassembler import build
HEADER='from typing import Final\nfrom openassembler.pyfront import *\n'
CASES=[
 ('runtime_iter','@kernel\ndef k(a:Array(i32)):\n    for c in a:\n        a[0]=c\n','compile-time'),
 ('list','@kernel\ndef k(a:Array(i32)):\n    for c in [1,2]:\n        a[0]=c\n','compile-time'),
 ('dict','@kernel\ndef k(a:Array(i32)):\n    for c in {1:2}:\n        a[0]=c\n','compile-time'),
 ('unpack','@kernel\ndef k(a:Array(i32)):\n    for x,y in ((1,2),(3,4)):\n        a[0]=x+y\n','scalar name'),
 ('nested_values','@kernel\ndef k(a:Array(i32)):\n    for c in ((1,2),(3,4)):\n        a[0]=c\n','numeric scalar'),
 ('bool_values','@kernel\ndef k(a:Array(i32)):\n    for c in (True,False):\n        a[0]=c\n','numeric scalar'),
 ('nonfinite','VALUES:Final=(1.0,1e309)\n@kernel\ndef k(a:Array(f32)):\n    for c in VALUES:\n        a[0]=c\n','nonfinite'),
 ('integer_range','@kernel\ndef k(a:Array(i64)):\n    for c in (1<<80,2):\n        a[0]=c\n','single device integer'),
 ('signed_unsigned_mix','@kernel\ndef k(a:Array(i64)):\n    for c in (-1,18446744073709551615):\n        a[0]=c\n','single device integer'),
 ('mixed_inexact','@kernel\ndef k(a:Array(f32)):\n    for c in (0.5,16777217):\n        a[0]=c\n','exactly representable'),
 ('fixed_target_type','@kernel\ndef k(a:Array(i32)):\n    c=1\n    for c in (0.5,1.0):\n        a[0]=c\n','does not fit'),
 ('local_tuple','@kernel\ndef k(a:Array(i32)):\n    values=17\n    for c in values:\n        a[0]=c\n','compile-time'),
 ('tuple_len','@kernel\ndef k(a:Array(i32)):\n    for c in ('+','.join('1'for _ in range(257))+'):\n        a[0]=c\n','256'),
 ('empty_target_read','@device\ndef h(x:i64)->i64:\n    for c in ():\n        x+=i64(c)\n    return i64(c)\n@kernel\ndef k(a:Array(i64)):\n    a[0]=h(a[0])\n','uninitialized'),
 ('early_helper_return','@device\ndef h(x:i64)->i64:\n    for c in (1,2):\n        if x<i64(c):\n            return x\n    return x\n@kernel\ndef k(a:Array(i64)):\n    a[0]=h(a[0])\n','terminal return'),
 ('dynamic_loop_else','@kernel\ndef k(a:Array(i32)):\n    for c in range(3):\n        a[0]=c\n    else:\n        a[0]=7\n','else'),
 ('barrier_break','@kernel\ndef k(a:Array(i32)):\n    for c in (1,2,3):\n        if tid_x()<c:\n            break\n        syncthreads()\n    a[0]=7\n','thread-dependent'),
 ('barrier_else','@kernel\ndef k(a:Array(i32)):\n    for c in (1,2,3):\n        if tid_x()<c:\n            break\n    else:\n        syncthreads()\n    a[0]=7\n','thread-dependent'),
 ('barrier_continue','@kernel\ndef k(a:Array(i32)):\n    for c in (1,2,3):\n        if tid_x()<c:\n            continue\n        syncthreads()\n    a[0]=7\n','thread-dependent'),
 ('selection_budget','@kernel\ndef k(a:Array(i32)):\n'+''.join('    for c'+str(i)+' in ('+','.join('1'for _ in range(256))+'):\n        a[0]=c'+str(i)+'\n'for i in range(5)),'1024'),
]
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-tuples-negative-')as temp:
  for name,source,reason in CASES:
   path=Path(temp)/(name+'.py');path.write_text(HEADER+source)
   try:
    text,_=compile_file(path)
    for kernel in parse_module(text).kernels.values():build.generate(kernel,'sm_80','safe')
   except (NotSupported,Exception)as error:
    assert isinstance(error,NotSupported)or type(error).__name__=='NotSupported',(name,type(error).__name__,str(error))
    assert reason in str(error),(name,reason,str(error));rows.append(dict(name=name,rejected=True,message=str(error)))
   else:raise AssertionError('invalid tuple iteration admitted: '+name)
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()for n in('candidate-static_values.py','candidate-pyfront.py','candidate-helpers.py','candidate-modules.py')},scope='Original tuple control/type/iteration/binding/size/helper-return/convergence and source diagnostics rejection controls; independent from hardware correctness')
 (HERE/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'negative controls')
