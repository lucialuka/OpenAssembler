"""Independent Boolean truth and operand-returning atomic short-circuit references."""
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
    x=a[i]
    first=x and atomic_add(count,i,1)
    second=x or atomic_add(count,i,1)
    flag=0<x<7
    inverted=not (flag or False)
    enabled=bool(x) and True
    out[i]=first+second+i32(flag)*2+i32(inverted)*4+i32(enabled)*8
'''
if __name__=='__main__':
 data=[-19,-3,-1,0,1,3,7,19];wanted=[];counter=[]
 for x in data:
  calls=0
  if x:first=calls;calls+=1
  else:first=x
  if x:second=x
  else:second=calls;calls+=1
  flag=0<x<7;inverted=not flag;enabled=bool(x)
  wanted.append(first+second+int(flag)*2+int(inverted)*4+int(enabled)*8);counter.append(calls)
 rows=[];ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-boolean-values-')as temporary:
  path=Path(temporary)/'source.py';path.write_text(SOURCE);text,_=compile_file(path);module=parse_module(text)
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
   for engine in('ptx','normal','poisoned'):
    memory=Memory()
    for address,values in((ap,data),(out,[0]*8),(cp,[0]*8)):memory.write_bytes(address-64,guard+struct.pack('<8i',*values)+guard)
    if engine=='ptx':Interp(module,'k',PLaunch(block=(8,1,1),params={'a':ap,'out':out,'count':cp}),memory).run()
    else:Machine(load_listing(listing(code,labels)),Launch(block=(8,1,1),params=struct.pack('<QQQ',ap,out,cp),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
    for address,values in((ap,data),(out,wanted),(cp,counter)):assert bytes(memory.read(address-64+j,1)for j in range(160))==guard+struct.pack('<8i',*values)+guard,(level,engine,address)
    rows.append(dict(level=level,engine=engine,passed=True))
 report=dict(schema=1,passed=True,source=SOURCE,inputs=data,expected_outputs=wanted,expected_atomic_counts=counter,results=rows,executions=12,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate.py',HERE/'candidate-pyfront.py',Path(__file__))},scope='Isolated Boolean locals/literals/not/bool/chain values with explicit integer conversion and homogeneous operand-returning and/or atomic short circuit, full independent output/counter/input/guard bits at four PTX/normal/poisoned levels; no production, Boolean ABI/arrays, mixed-kind unions, full source/remote/ownership or hardware claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 12 Boolean-value/operand-selection executions with exact atomic counts')
