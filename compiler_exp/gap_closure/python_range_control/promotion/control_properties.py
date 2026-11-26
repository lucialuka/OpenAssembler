"""Independent generated boundary ranges including snapshots, targets and else exits."""
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
if __name__=='__main__':
 rng=random.Random(79123);rows=[];inputs=[-(1<<63),-19,-1,0,1,3,7,(1<<63)-1];ap,out=0x100010000,0x100000000;guard=b'\xa5'*64;before=guard+struct.pack('<8Q',*[x&MASK for x in inputs])+guard
 sources=[HERE/n for n in('candidate-pyfront.py','candidate-return_flow.py','control_properties.py')];frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
 with tempfile.TemporaryDirectory(prefix='oa-boundary-range-properties-')as temporary:
  for index in range(128):
   step=rng.choice([-(1<<31),-2147483647,-9,-3,-2,-1,1,2,3,7,2147483647]);stop=rng.choice([-(1<<31),-(1<<31)+rng.randrange(10),-11,0,11,(1<<31)-1-rng.randrange(10),(1<<31)-1]);start=max(-(1<<31),min((1<<31)-1,stop-step*rng.randrange(-2,5)+rng.randrange(-2,3)))
   values=list(range(start,stop,step));assert len(values)<8
   skip=rng.randrange(-1,7);halt=rng.randrange(-1,7);mutation=rng.randrange(-9,10)
   source=f'''from openassembler.pyfront import *
@device
def helper(x:i64)->i64:
    start={start}
    stop={stop}
    target=17
    value=x
    count=0
    for target in range(start,stop,{step}):
        if count=={halt} and x&i64(1):
            break
        count+=1
        if count=={skip}:
            continue
        value=value*i64(3)+i64(target)
        target={mutation}
        start=0
        stop=0
    else:
        value+=i64(19)
    return value+i64(target)+i64(count)
@kernel
def k(a:Array(i64),out:Array(i64)):
    i=tid_x()
    out[i]=helper(a[i])
'''
   expected=[]
   for x in inputs:
    target=17;value=x;count=0
    for target in values:
     if count==halt and x&1:break
     count+=1
     if count==skip:continue
     value=(value*3+target)&MASK;target=mutation
    else:value=(value+19)&MASK
    expected.append((value+target+count)&MASK)
   path=Path(temporary)/'property.py';path.write_text(source);text,_=compile_file(path);module=parse_module(text);wanted=guard+struct.pack('<8Q',*expected)+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(64)+guard)
     if engine=='ptx':Interp(module,'k',PLaunch(block=(8,1,1),params={'a':ap,'out':out}),memory).run(max_steps=5000)
     else:Machine(load_listing(listing(code,labels)),Launch(block=(8,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run(max_steps=5000)
     assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(index,start,stop,step,level,engine)
     assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
   rows.append(dict(index=index,start=start,stop=stop,step=step,range_values=values,source=source,expected_modulo64=[str(x)for x in expected],passed=True))
 assert frozen=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
 result=dict(schema=1,passed=True,seed=79123,inputs=[str(x)for x in inputs],results=rows,executions=1536,source_hashes=frozen,script_sha256=frozen['control_properties.py'],scope='128 finite independent generated boundary/step/runtime-bound snapshot/target mutation/skip/break/exhaustion combinations, full output/input/guard bytes, all four PTX/normal/poisoned levels; no additional GPU or universal loop proof')
 (HERE/'control-properties.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 128 boundary/control combinations; 1536 executions')
