"""Condition-chain type/effect/initialization/convergence rejection domains."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
HEADER='from openassembler.pyfront import *\n'
CASES=[
 ('mixed_wide','@kernel\ndef k(a:Array(i32),x:i64):\n    if 0<a[0]<x:\n        a[0]=7\n','mismatched'),
 ('mixed_float','@kernel\ndef k(a:Array(i32),x:f32):\n    if 0<a[0]<x:\n        a[0]=7\n','mismatched'),
 ('unsigned_negative','@kernel\ndef k(a:Array(u32)):\n    if -1<a[0]<7:\n        a[0]=7\n','fit in u32'),
 ('float_inexact','@kernel\ndef k(a:Array(f32)):\n    if 16777217<a[0]<16777219:\n        a[0]=7.0\n','exactly representable'),
 ('wide_literal','@kernel\ndef k(a:Array(i64)):\n    if i64(0)<a[0]<18446744073709551615:\n        a[0]=i64(7)\n','fit in s64'),
 ('wide_unsigned_float','@kernel\ndef k(a:Array(u64)):\n    if u64(0)<a[0]<1.5:\n        a[0]=u64(7)\n','fit in u64'),
 ('membership','@kernel\ndef k(a:Array(i32)):\n    if 0<a[0]in(1,2):\n        a[0]=7\n','unsupported expression: Tuple'),
 ('identity','@kernel\ndef k(a:Array(i32)):\n    if 0<a[0]is 1:\n        a[0]=7\n','comparison Is'),
 ('bool_operand','@kernel\ndef k(a:Array(i32)):\n    if False<a[0]<7:\n        a[0]=7\n','bool constants'),
 ('chain_value','@kernel\ndef k(a:Array(i32)):\n    value=0<a[0]<7\n    a[0]=value\n','chained comparisons'),
 ('bool_value','@kernel\ndef k(a:Array(i32)):\n    value=a[0]and 7\n    a[0]=value\n','only supported directly'),
 ('compare_comparison','@kernel\ndef k(a:Array(i32)):\n    if 0<(a[0]<3)<7:\n        a[0]=7\n','comparison'),
 ('uninitialized','@device\ndef h(x:i32)->i32:\n    if 0<x<7:\n        y=3\n    return y\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(a[0])\n','uninitialized'),
 ('helper_effect','@device\ndef h(x:i32)->i32:\n    if 0<x<atomic_add(x,0,1):\n        return 3\n    return 7\n@kernel\ndef k(a:Array(i32)):\n    a[0]=h(a[0])\n','pure scalar calls'),
 ('varying_barrier','@kernel\ndef k(a:Array(i32)):\n    if 0<tid_x()<7:\n        syncthreads()\n','thread-dependent'),
 ('varying_loop','@kernel\ndef k(a:Array(i32)):\n    x=tid_x()\n    while 0<x<7:\n        syncthreads()\n        x+=1\n','thread-dependent'),
]
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-chain-negative-')as temporary:
  for name,source,reason in CASES:
   path=Path(temporary)/(name+'.py');path.write_text(HEADER+source)
   try:compile_file(path)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error));assert Path(error.path).resolve()==path.resolve()and error.lineno is not None
    rows.append(dict(name=name,source=HEADER+source,reason=reason,rejected=True,error=str(error)))
   else:raise AssertionError(name)
 report=dict(schema=1,passed=True,results=rows,source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()for n in('candidate-pyfront.py','candidate-modules.py','negative_controls.py')},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='16 condition-chain mixed-kind/literal/value/membership/identity/initialization/pure-helper/convergence rejections with original diagnostics; no hardware claim')
 (HERE/'negative-controls.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 16 original chain rejection controls')
