"""Original path completeness, initialization and scalar return-type boundaries."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file,NotSupported
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory
ACCEPTED=[
 ('both_branches','    if x<i64(0):\n        return -x\n    else:\n        return x+i64(3)\n',lambda x:-x if x<0 else x+3),
 ('initialized_live_path','    if x<i64(0):\n        return -x\n    else:\n        value=x+i64(7)\n    return value\n',lambda x:-x if x<0 else x+7),
 ('finite_first_return','    for c in (3,7):\n        return x+i64(c)\n',lambda x:x+3),
 ('empty_else_return','    for c in ():\n        return x\n    else:\n        return x+i64(11)\n',lambda x:x+11),
 ('break_return','    value=x\n    for c in (3,7):\n        if x<i64(c):\n            break\n        value+=i64(c)\n    else:\n        return value+i64(17)\n    return value\n',lambda x:x if x<3 else x+3 if x<7 else x+27),
]
REJECTED=[
 ('missing','    if x<i64(0):\n        return -x\n','every path'),
 ('bare','    return\n','requires a value'),
 ('zero_trip_range','    for i in range(0):\n        return x\n','every path'),
 ('empty_tuple','    for i in ():\n        return x\n','every path'),
 ('uninitialized','    if x<i64(0):\n        value=x\n    return value\n','uninitialized'),
 ('bad_return_type','    if x<i64(0):\n        return 1.0\n    return x\n','annotated'),
 ('break_missing','    for i in (1,2):\n        if x<i64(0):\n            break\n        return x\n','every path'),
]
if __name__=='__main__':
 rows=[];domain=[-19,-1,0,1,3,7,19];mask=(1<<64)-1
 with tempfile.TemporaryDirectory(prefix='oa-return-paths-')as temp:
  for name,body,expect in ACCEPTED+REJECTED:
   source='from openassembler.pyfront import *\n@device\ndef h(x:i64)->i64:\n'+body+'@kernel\ndef k(a:Array(i64),out:Array(i64)):\n    i=tid_x()\n    out[i]=h(a[i])\n'
   path=Path(temp)/(name+'.py');path.write_text(source)
   if (name,body,expect)in REJECTED:
    try:compile_file(path)
    except NotSupported as error:
     assert expect in str(error),(name,str(error));rows.append(dict(name=name,rejected=True,error=str(error)))
    else:raise AssertionError(name)
   else:
    text,_=compile_file(path);module=parse_module(text);mem=Memory();ap,out=0x100010000,0x100000000;mem.write_bytes(ap,struct.pack('<7Q',*[x&mask for x in domain]));mem.write_bytes(out,bytes(56))
    Interp(module,'k',PLaunch(block=(7,1,1),params={'a':ap,'out':out}),mem).run();expected=struct.pack('<7Q',*[expect(x)&mask for x in domain]);assert bytes(mem.read(out+i,1)for i in range(56))==expected
    rows.append(dict(name=name,accepted=True,expected_hex=expected.hex(),source=source))
 result=dict(schema=1,passed=True,results=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),return_flow_sha256=hashlib.sha256((HERE/'return_flow.py').read_bytes()).hexdigest(),scope='Five original all-path return/live-path initialization/exhaustion controls and seven missing/bare/type/empty/zero-trip/break return rejections; finite PTX interpretation only, no native/toolkit/build/ownership/hardware claim')
 (HERE/'paths-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed five original all-path controls and seven rejection boundaries')
