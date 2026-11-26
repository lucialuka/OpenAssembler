"""Original return completeness, definite initialization, pure effects and ABI limits."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
HEADER='from openassembler.pyfront import *\n'
CASES=[
 ('missing','    x+=i64(1)\n','every path'),
 ('one_branch','    if x<i64(0):\n        return x\n','every path'),
 ('bare','    if x<i64(0):\n        return\n    return x\n','requires a value'),
 ('wrong_type','    if x<i64(0):\n        return 1.0\n    return x\n','annotated'),
 ('wrong_width','    if x<i64(0):\n        return 1\n    return x\n','annotated'),
 ('zero_trip','    for i in range(0):\n        return x\n','every path'),
 ('empty','    for i in ():\n        return x\n','every path'),
 ('while_only','    while x<i64(0):\n        return x\n','every path'),
 ('break_fallthrough','    for i in (1,2):\n        if x<i64(0):\n            break\n        return x\n','every path'),
 ('continue_fallthrough','    for i in (1,2):\n        if x<i64(0):\n            continue\n        return x\n','every path'),
 ('capture','    if x<i64(0):\n        return hidden\n    return x\n','uninitialized or captured'),
 ('read_memory','    if x<i64(0):\n        return source[0]\n    return x\n','cannot access memory'),
 ('barrier','    if x<i64(0):\n        syncthreads()\n        return x\n    return x\n','pure scalar calls'),
 ('atomic','    if x<i64(0):\n        return atomic_add(x,0,1)\n    return x\n','pure scalar calls'),
 ('thread_id','    if x<i64(0):\n        return i64(tid_x())\n    return x\n','pure scalar calls'),
 ('recursion','    if x<i64(0):\n        return h(x)\n    return x\n','recursive device helper'),
 ('uninitialized_branch','    if x<i64(0):\n        value=x\n    else:\n        pass\n    return value\n','uninitialized'),
 ('uninitialized_loop','    for i in range(3):\n        value=x\n        if x<i64(0):\n            return x\n    return value\n','uninitialized'),
 ('uninitialized_empty_target','    for i in ():\n        return x\n    return i64(i)\n','uninitialized'),
 ('uninitialized_break','    for i in (1,2):\n        if x<i64(0):\n            break\n        value=x\n    else:\n        value=x\n    return value\n','uninitialized'),
 ('range_else','    for i in range(3):\n        if x<i64(0):\n            return x\n    else:\n        return x\n','loop else'),
 ('while_else','    while x<i64(0):\n        return x\n    else:\n        return x\n','loop else'),
 ('break_outside','    break\n    return x\n','outside a loop'),
 ('continue_outside','    continue\n    return x\n','outside a loop'),
]
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-return-negative-')as temp:
  for name,body,reason in CASES:
   text=HEADER+'@device\ndef h(x:i64)->i64:\n'+body+'@kernel\ndef k(a:Array(i64)):\n    a[0]=h(a[0])\n';path=Path(temp)/(name+'.py');path.write_text(text)
   try:compile_file(path)
   except NotSupported as error:
    assert reason in str(error),(name,reason,str(error))
    assert Path(error.path).resolve()==path.resolve()and error.lineno is not None,(name,error.path,error.lineno)
    rows.append(dict(name=name,source=text,reason=reason,rejected=True,error=str(error)))
   else:raise AssertionError(name)
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()for n in('return_flow.py','candidate-pyfront.py','candidate-helpers.py','candidate-modules.py')},scope='24 original missing/type/empty/zero-trip/break/continue/initialization/capture/memory/convergence/recursion rejections, with original file and line diagnostics; no hardware claim')
 (HERE/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'original rejection controls')
