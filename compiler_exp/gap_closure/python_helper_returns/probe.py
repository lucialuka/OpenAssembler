"""Retained original early-return rejection and isolated complete-bit hypothesis."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from normalize import normalize
from openassembler.pyfront import compile_file,compile_kernel,find_kernel,NotSupported
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

def reference(name,x):
 if name=='classify_kernel':return-x if x<0 else 17 if x==0 else x+3
 if name=='find_kernel':
  result=0
  for step in range(5):
   result+=step
   if result>x:return result
  return result+11
 result=0
 for outer in(1,2,3):
  for inner in(3,5):
   result+=outer*inner
   if result>x:return result
 return result+19
if __name__=='__main__':
 original=HERE/'source.py';baseline=[]
 for name in('classify_kernel','find_kernel','nested_kernel'):
  try:compile_kernel(find_kernel(original,name))
  except NotSupported as error:baseline.append(dict(entry=name,rejected=True,error=str(error)))
  else:raise AssertionError(name)
 normalized=normalize(original.read_text());(HERE/'normalized.py').write_text(normalized);rows=[]
 with tempfile.TemporaryDirectory(prefix='oa-helper-return-')as temp:
  path=Path(temp)/'normalized.py';path.write_text(normalized);text,names=compile_file(path);module=parse_module(text)
  data=[-(1<<63),-19,-1,0,1,3,7,19,63,(1<<63)-1]+[i-59 for i in range(118)];mask=(1<<64)-1
  guard=b'\xa5'*64;ap,out=0x100010000,0x100000000;before=guard+struct.pack('<128Q',*[x&mask for x in data])+guard
  for name in names:
   wanted=guard+struct.pack('<128Q',*[(reference(name,x)&mask)if i<91 else 0x13579bdf2468ace0 for i,x in enumerate(data)])+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level);assert words
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+struct.pack('<128Q',*([0x13579bdf2468ace0]*128))+guard)
     if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':91}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,91),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(name,level,engine)
     assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
     rows.append(dict(entry=name,level=level,engine=engine,passed=True))
 result=dict(schema=1,passed=True,baseline=baseline,executions=len(rows),results=rows,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'source.py',HERE/'normalized.py',HERE/'normalize.py',Path(__file__))},scope='Three original pure-helper early-return rejections; isolated separately normalized flag-return hypothesis with required terminal fallback, whole independent modulo64 output/guard/tail/unchanged input through all four PTX/normal/poisoned levels. Original diagnostics/graph/control/effect/remote and hardware gates remain open; no production claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained 3 original rejections; passed',len(rows),'normalized whole-byte executions')
