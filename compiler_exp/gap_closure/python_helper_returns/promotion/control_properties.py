"""Generated original return/loop/caller combinations against independent signed arithmetic."""
import hashlib,json,random,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
MASK=(1<<64)-1
def signed(x):
 bits=x&MASK;return bits-(1<<64)if bits&(1<<63)else bits
def reference(values,x,skip,stop,change):
 value=x;x=signed(x+change)
 for c in values:
  if c==skip:continue
  if x<c:break
  value=signed(value*3+c)
  if value<stop:return value
 else:return signed(value+17)
 return signed(value-11)
if __name__=='__main__':
 rng=random.Random(68193);rows=[];data=[-(1<<63),-19,-5,-1,0,1,3,5,7,19,(1<<63)-1]+[rng.randint(-31,31)for _ in range(5)]
 ap,out=0x100010000,0x100000000;guard=b'\xa5'*64;before=guard+struct.pack('<16Q',*[x&MASK for x in data])+guard
 inputs=[HERE/n for n in('return_flow.py','candidate-helpers.py','candidate-modules.py','candidate-pyfront.py','control_properties.py')];frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
 with tempfile.TemporaryDirectory(prefix='oa-return-control-properties-')as temp:
  for index in range(64):
   values=tuple(rng.randint(-7,7)for _ in range(rng.randrange(7)));skip=rng.randint(-7,7);stop=rng.randint(-11,11);change=rng.randint(-9,9)
   source=f'''from openassembler.pyfront import *
@device
def helper(x:i64)->i64:
    value=x
    x+=i64({change})
    for c in {values!r}:
        if c=={skip}:
            continue
        if x<i64(c):
            break
        value=value*i64(3)+i64(c)
        if value<i64({stop}):
            return value
    else:
        return value+i64(17)
    return value-i64(11)
@kernel
def k(a:Array(i64),out:Array(i64)):
    i=tid_x()
    x=a[i]
    out[i]=helper(x)+x+helper(x)
'''
   path=Path(temp)/'property.py';path.write_text(source);text,_=compile_file(path);module=parse_module(text)
   values_out=[(2*reference(values,x,skip,stop,change)+x)&MASK for x in data];wanted=guard+struct.pack('<16Q',*values_out)+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level);assert words
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(128)+guard)
     if engine=='ptx':Interp(module,'k',PLaunch(block=(16,1,1),params={'a':ap,'out':out}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(16,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(index,values,skip,stop,change,level,engine)
     assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
   rows.append(dict(index=index,source=source,expected_modulo64=[str(x)for x in values_out],executions=12,passed=True))
 assert frozen=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
 result=dict(schema=1,passed=True,seed=68193,inputs=[str(x)for x in data],results=rows,executions=768,script_sha256=frozen['control_properties.py'],source_hashes=frozen,scope='64 original signed return/skip/break/empty/exhaustion/caller snapshot combinations; independent wrapped signed arithmetic and full guarded unchanged inputs/outputs at four PTX/normal/poisoned levels; no extra hardware claim')
 (HERE/'control-properties.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 64 original return combinations; 768 whole-byte executions')
