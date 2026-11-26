"""Original condition chains including atomic single-evaluation/short-circuit effects."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
SOURCE='''from openassembler.pyfront import *
@kernel
def k(a:Array(i32),out:Array(i32),count:Array(i32)):
    i=tid_x()
    value=11
    if -3<a[i]<7:
        value+=1
    if not (0<=a[i]<=3):
        value+=2
    if a[i]<0<atomic_add(count,i,1):
        value+=4
    if -1<atomic_add(count,i,1)<1:
        value+=8
    out[i]=value
'''
if __name__=='__main__':
 data=[-19,-3,-1,0,1,3,7,19];expected=[];counters=[]
 for x in data:
  value=11+(1 if-3<x<7 else 0)+(2 if not 0<=x<=3 else 0);counter=0
  if x<0:
   old=counter;counter+=1
   if 0<old:value+=4
  old=counter;counter+=1
  if-1<old<1:value+=8
  expected.append(value);counters.append(counter)
 rows=[];ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-comparison-chain-')as temporary:
  path=Path(temporary)/'source.py';path.write_text(SOURCE);text,_=compile_file(path);module=parse_module(text)
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
   for engine in('ptx','normal','poisoned'):
    memory=Memory()
    for address,values in((ap,data),(out,[0]*8),(cp,[0]*8)):memory.write_bytes(address-64,guard+struct.pack('<8i',*values)+guard)
    if engine=='ptx':Interp(module,'k',PLaunch(block=(8,1,1),params={'a':ap,'out':out,'count':cp}),memory).run()
    else:Machine(load_listing(listing(code,labels)),Launch(block=(8,1,1),params=struct.pack('<QQQ',ap,out,cp),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
    for address,values in((ap,data),(out,expected),(cp,counters)):
     assert bytes(memory.read(address-64+i,1)for i in range(160))==guard+struct.pack('<8i',*values)+guard,(level,engine,address)
    rows.append(dict(level=level,engine=engine,passed=True))
 result=dict(schema=1,passed=True,source=SOURCE,inputs=data,expected_outputs=expected,expected_atomic_counters=counters,results=rows,executions=12,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-pyfront.py',HERE/'candidate.py',Path(__file__))},scope='Isolated original scalar comparison chains in conditions, complete independent output/guard/input/atomic effect bytes, four PTX/normal/poisoned levels. No production, first-class Boolean, full mixed-type/NaN/helper/source-ownership/hardware claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed 12 condition-chain executions including atomic effect counts')
