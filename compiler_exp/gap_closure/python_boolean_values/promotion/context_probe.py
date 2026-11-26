"""Resolved leading-literal Boolean operand values across all numeric scalar kinds."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import contract_from_ptx
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-boolean-context-')as temporary:
  for dtype,fmt,data in(('i32','i',[-19,-1,0,1,3,19]),('u32','I',[0,1,3,19,1<<31,(1<<32)-1]),('i64','q',[-(1<<63),-1,0,1,3,(1<<63)-1]),('u64','Q',[0,1,3,19,1<<63,(1<<64)-1]),('f32','f',[-19.0,-0.0,0.0,3.0,float('inf'),float('nan')])):
   source=f'from openassembler.pyfront import *\n@kernel\ndef k(a:Array({dtype}),out:Array({dtype})):\n    i=tid_x()\n    out[i]=0 or a[i]\n';path=Path(temporary)/(dtype+'.py');path.write_text(source);text,_=compile_file(path);assert contract_from_ptx(text)['boolean_values'];module=parse_module(text);before=struct.pack('<6'+fmt,*data)
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
    for engine in('ptx','normal','poisoned'):
     memory=Memory();ap,out=0x100010000,0x100000000;memory.write_bytes(ap,before);memory.write_bytes(out,bytes(len(before)))
     if engine=='ptx':Interp(module,'k',PLaunch(block=(6,1,1),params={'a':ap,'out':out}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(6,1,1),params=struct.pack('<QQ',ap,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     assert bytes(memory.read(out+i,1)for i in range(len(before)))==before,(dtype,level,engine)
     rows.append(dict(dtype=dtype,level=level,engine=engine,passed=True))
 report=dict(schema=1,passed=True,results=rows,executions=60,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'candidate-pyfront.py',HERE/'candidate-backend.py',HERE/'candidate-modules.py',Path(__file__))},scope='Five resolved leading numeric-literal operand contexts with original source records and exact selected scalar bits at four PTX/normal/poisoned levels, including FP32 NaN/Inf/signed zeros; no hardware/production claim')
 (HERE/'context-probe.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 60 leading-literal contextual operand executions')
