"""Independent chained condition truth for scalar widths and FP32 special values."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler import build,oracle
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 rows=[];artifacts={};cases=[];guard=b'\xa5'*64;ap,out=0x100010000,0x100000000;HERE.joinpath('oracle').mkdir(exist_ok=True)
 domains=[('i32','i',[-2147483648,-19,-2,-1,0,1,2,3,19,2147483647]),('u32','I',[0,1,2,3,19,2147483647,2147483648,4294967295]),('i64','q',[-(1<<63),-19,-2,-1,0,1,2,3,19,(1<<63)-1]),('u64','Q',[0,1,2,3,19,(1<<63)-1,1<<63,(1<<64)-1]),('f32','f',[float('-inf'),-19.0,-2.0,-1.0,-0.0,0.0,1.0,2.0,3.0,19.0,float('inf'),float('nan')])]
 for dtype,fmt,data in domains:
  low,high=('0','3')if dtype[0]=='u'else('-2','3');literal=lambda n:str(float(n))if dtype=='f32'else f'{dtype}({n})'
  source=f'''from openassembler.pyfront import *
@kernel
def k(a:Array({dtype}),out:Array(i32)):
    i=tid_x()
    x=a[i]
    value=0
    if {literal(low)}<x<{literal(high)}:
        value+=1
    if {literal(low)}!=x!={literal(high)}:
        value+=2
    if not ({literal(low)}<=x<={literal(high)}):
        value+=4
    step=0
    while 0<=step<3 and {literal(low)}<=x<={literal(high)}:
        step+=1
        value+=8
    out[i]=value
'''
  path=HERE/'oracle'/(dtype+'.py');path.write_text(source);text,_=compile_file(path);ptx=path.with_suffix('.ptx');ptx.write_text(text);binary=path.with_suffix('.cubin');result=oracle.run_tool('ptxas','-arch=sm_80','-o',oracle.tool_path(binary),oracle.tool_path(ptx),check=False);path.with_suffix('.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr;path.with_suffix('.sass').write_text(oracle.disasm_text(binary))
  module=parse_module(text);n=len(data);lo,hi=int(low),int(high);expected=[(1 if lo<x<hi else 0)+(2 if lo!=x!=hi else 0)+(4 if not lo<=x<=hi else 0)+(24 if lo<=x<=hi else 0)for x in data];before=guard+struct.pack('<'+str(n)+fmt,*data)+guard;wanted=guard+struct.pack('<'+str(n)+'i',*expected)+guard
  cases.append(dict(dtype=dtype,source=source,input_hex=before.hex(),expected_hex=wanted.hex(),lanes=n))
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
   for engine in('ptx','normal','poisoned'):
    memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(n*4)+guard)
    if engine=='ptx':Interp(module,'k',PLaunch(block=(n,1,1),params={'a':ap,'out':out}),memory).run()
    else:Machine(load_listing(listing(code,labels)),Launch(block=(n,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
    assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(dtype,level,engine)
    assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
    rows.append(dict(dtype=dtype,level=level,engine=engine,passed=True))
  for artifact in HERE.joinpath('oracle').glob(dtype+'.*'):artifacts[str(artifact.relative_to(HERE))]=hashlib.sha256(artifact.read_bytes()).hexdigest()
 report=dict(schema=1,passed=True,executions=60,results=rows,cases=cases,artifact_hashes=artifacts,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate.py',HERE/'candidate-pyfront.py',Path(__file__))},scope='Five typed condition-chain domains with signed/unsigned boundaries, FP32 NaN/infinity/signed zeros, ordered/unequal/negated chains and while/and composition; 60 complete PTX/normal/poisoned executions and actual ptxas cubins/disassembly. No GPU execution, production or general first-class Boolean claim')
 (HERE/'domains.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 60 typed/special-value executions and five original toolkit controls')
