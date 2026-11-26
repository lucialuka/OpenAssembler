"""Deterministic tuple/exit/target state combinations against an independent loop reference."""
import hashlib,json,random,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
rng=random.Random(53711);MASK=(1<<64)-1

def reference(values,x,skip,change):
 current=11;value=x
 for current in values:
  if current==skip:continue
  if x<current:break
  value=(value*3+current)&MASK
  current+=change
 else:value=(value+17)&MASK
 return(value+current)&MASK
if __name__=='__main__':
 rows=[];data=[-(1<<63),-19,-5,-1,0,1,3,5,7,19,(1<<63)-1]+[rng.randint(-31,31)for _ in range(5)];ap,out=0x100010000,0x100000000;guard=b'\xa5'*64;before=guard+struct.pack('<16Q',*[x&MASK for x in data])+guard
 with tempfile.TemporaryDirectory(prefix='oa-tuple-exit-properties-')as temp:
  for index in range(128):
   values=tuple(rng.randint(-7,7)for _ in range(rng.randrange(7)));skip=rng.randint(-7,7);change=rng.randint(-9,9)
   source=f'''from typing import Final
from openassembler.pyfront import *
VALUES:Final={values!r}
@device
def helper(x:i64)->i64:
    current=11
    value=x
    for current in VALUES:
        if current=={skip}:
            continue
        if x<i64(current):
            break
        value=value*i64(3)+i64(current)
        current+={change}
    else:
        value+=i64(17)
    return value+i64(current)
@kernel
def k(a:Array(i64),out:Array(i64)):
    i=tid_x()
    out[i]=helper(a[i])
'''
   path=Path(temp)/'property.py';path.write_text(source);text,_=compile_file(path);module=parse_module(text)
   wanted=guard+struct.pack('<16Q',*[reference(values,x,skip,change)for x in data])+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level);assert words
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(128)+guard)
     if engine=='ptx':Interp(module,'k',PLaunch(block=(16,1,1),params={'a':ap,'out':out}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(16,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(index,values,skip,change,level,engine)
     assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
   rows.append(dict(index=index,values=values,skip=skip,mutation=change,expected_modulo64=[str(reference(values,x,skip,change))for x in data],source=source,executions=12,passed=True))
 result=dict(schema=1,passed=True,seed=53711,inputs=[str(x)for x in data],results=rows,executions=1536,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in('openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/pyfront/static_values.py','openassembler/pyfront/modules.py')},scope='128 deterministic original tuple/skip/break/target mutation/exhaustion combinations, independent Python loop reference with explicit modulo64 device arithmetic, full guarded inputs/outputs and four PTX/normal/poisoned levels; no extra GPU execution or universal semantic proof')
 (HERE/'control-properties.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 128 original control combinations;',result['executions'],'whole-byte executions')
