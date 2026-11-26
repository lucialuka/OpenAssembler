"""Unsupported range domains and loop-else convergence/initialization failures."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
HEADER='from openassembler.pyfront import *\n'
CASES=[
 ('zero_step','@kernel\ndef k(out:Array(i32)):\n    for i in range(0,3,0):\n        out[0]=i\n','must not be 0'),
 ('wide_step','@kernel\ndef k(out:Array(i32)):\n    for i in range(0,3,2147483648):\n        out[0]=i\n','fit in i32'),
 ('runtime_step','@kernel\ndef k(out:Array(i32),step:i32):\n    for i in range(0,3,step):\n        out[0]=i\n','constant int'),
 ('float_step','@kernel\ndef k(out:Array(i32)):\n    for i in range(0,3,0.5):\n        out[0]=i\n','constant int'),
 ('wide_bounds','@kernel\ndef k(out:Array(i32),n:i64):\n    for i in range(n):\n        out[0]=i\n','int32'),
 ('float_bounds','@kernel\ndef k(out:Array(i32),n:f32):\n    for i in range(n):\n        out[0]=i\n','int32'),
 ('range_else_missing','@device\ndef h(x:i32)->i32:\n    for i in range(x):\n        if x:\n            break\n    else:\n        return x\n@kernel\ndef k(out:Array(i32)):\n    out[0]=h(3)\n','every path'),
 ('while_else_missing','@device\ndef h(x:i32)->i32:\n    while x:\n        break\n    else:\n        return x\n@kernel\ndef k(out:Array(i32)):\n    out[0]=h(3)\n','every path'),
 ('else_bypassed_init','@device\ndef h(x:i32)->i32:\n    for i in range(x):\n        break\n    else:\n        value=x\n    return value\n@kernel\ndef k(out:Array(i32)):\n    out[0]=h(3)\n','uninitialized'),
 ('body_only_init','@device\ndef h(x:i32)->i32:\n    while x:\n        value=x\n        x-=1\n    else:\n        return value\n@kernel\ndef k(out:Array(i32)):\n    out[0]=h(3)\n','uninitialized'),
 ('else_break_no_outer','@kernel\ndef k(out:Array(i32)):\n    for i in range(3):\n        pass\n    else:\n        break\n','outside a loop'),
 ('else_continue_no_outer','@kernel\ndef k(out:Array(i32)):\n    while False:\n        pass\n    else:\n        continue\n','outside a loop'),
 ('for_break_barrier','@kernel\ndef k(out:Array(i32)):\n    for i in range(3):\n        if tid_x()==i:\n            break\n    else:\n        syncthreads()\n','thread-dependent'),
 ('while_break_barrier','@kernel\ndef k(out:Array(i32)):\n    i=0\n    while i<3:\n        if tid_x()==i:\n            break\n        i+=1\n    else:\n        syncthreads()\n','thread-dependent'),
 ('for_return_barrier','@kernel\ndef k(out:Array(i32)):\n    for i in range(3):\n        if tid_x()==i:\n            return\n    else:\n        syncthreads()\n','thread-dependent'),
 ('varying_while_else','@kernel\ndef k(out:Array(i32)):\n    i=tid_x()\n    while i<3:\n        i+=1\n    else:\n        syncthreads()\n','thread-dependent'),
]
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-loop-control-negative-')as temporary:
  for name,source,reason in CASES:
   path=Path(temporary)/(name+'.py');path.write_text(HEADER+source)
   try:compile_file(path)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error));assert Path(error.path).resolve()==path.resolve()and error.lineno is not None
    rows.append(dict(name=name,source=HEADER+source,reason=reason,rejected=True,error=str(error)))
   else:raise AssertionError(name)
 result=dict(schema=1,passed=True,results=rows,source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()for n in('candidate-pyfront.py','candidate-return_flow.py')},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='16 original unsupported range/else flow/initialization/convergence rejection controls with original diagnostics; no hardware claim')
 (HERE/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 16 loop domain/flow/convergence rejection controls')
