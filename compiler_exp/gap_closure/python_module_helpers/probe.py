"""Independent complete-state local-package helper hypothesis, prior to integration."""
import ast,hashlib,json,random,struct,sys,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from resolver import Resolver,ResolutionError
from openassembler.pyfront import compile as frontend
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

def sha(data):return hashlib.sha256(data).hexdigest()
def compile_source(entry):
 tree,contract=Resolver(entry).resolve();names=frontend._Names(tree)
 kernels=frontend._file_kernels(tree,names,entry)
 text=frontend.HEADER.format(arch='sm_80')+'\n'.join(frontend._compile_entry(fn,names,'sm_80')for fn in kernels)
 return text,contract

if __name__=='__main__':
 text,contract=compile_source(HERE/'corpus/entry.py');assert text==(HERE/'candidate.ptx').read_text()
 module=parse_module(text);rng=random.Random(31881);mask=(1<<64)-1
 left=[0,mask,1<<63,(1<<63)-1,0xffffffff,0x100000000]+[rng.getrandbits(64)for _ in range(122)]
 right=[mask,mask,mask,2,0xffffffff,0x8000000000000001]+[rng.getrandbits(64)for _ in range(122)]
 guard=b'\xa5'*64;rows=[];ap,bp,out=0x100010000,0x100020000,0x100000000
 for name,kernel in module.kernels.items():
  if name=='floating':
   a=[(i-63)/8 for i in range(128)];b=[(i%11-5)/4 for i in range(128)];fmt='f';sentinel=-12345.0
  else:a,b=left,right;fmt='Q';sentinel=0x13579bdf2468ace0
  before_a=guard+struct.pack('<128'+fmt,*a)+guard;before_b=guard+struct.pack('<128'+fmt,*b)+guard
  for level in('safe','mined','sched','opt'):
   code,labels,words,_,registers=build.generate(kernel,'sm_80',level);assert words
   for n in(0,91,128):
    expected=[sentinel]*128
    for i in range(n):
     if name=='signed':expected[i]=(a[i]*pow(b[i],3))&mask
     elif name=='unsigned':expected[i]=(a[i]*b[i]+a[i])&mask
     else:expected[i]=a[i]*2+b[i]
    wanted=guard+struct.pack('<128'+fmt,*expected)+guard
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before_a);memory.write_bytes(bp-64,before_b);memory.write_bytes(out-64,guard+struct.pack('<128'+fmt,*([sentinel]*128))+guard)
     if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'b':bp,'out':out,'n':n}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,bp,out,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual==wanted,(name,level,n,engine)
     for address,before in((ap,before_a),(bp,before_b)):assert bytes(memory.read(address-64+i,1)for i in range(len(before)))==before
    rows.append(dict(entry=name,level=level,n=n,executions=3,passed=True,expected_sha256=sha(wanted),registers=registers))
 negatives=[]
 with tempfile.TemporaryDirectory(prefix='oa-helper-graph-negative-')as temporary:
  temp=Path(temporary)
  for name,old,new in[
   ('recursive','return x * y','return signed_product(x,y)'),
   ('capture','return x * y','return x * GLOBAL'),
   ('hostcall','return x * y','return print(x)'),
   ('missing','from .base import signed_product as product','from .missing import signed_product as product'),
   ('shadow','result = product(result, y)','product = result\n        result = product(result, y)'),
   ('rebound','raise RuntimeError(\'static compiler must never execute helper module code\')','signed_product = 7'),
  ]:
   folder=temp/name;shutil.copytree(HERE/'corpus',folder)
   target=folder/'arithmetic'/('transforms.py'if name in('missing','shadow')else'base.py')
   source=target.read_text();assert old in source;target.write_text(source.replace(old,new))
   try:compile_source(folder/'entry.py')
   except Exception as error:negatives.append(dict(name=name,rejected=True,error=type(error).__name__+': '+str(error)))
   else:raise AssertionError('unvalidated source graph was admitted: '+name)
 result=dict(schema=1,passed=True,results=rows,executions=sum(r['executions']for r in rows),negative_controls=negatives,contract=contract,source_hashes={str(p.relative_to(HERE)):sha(p.read_bytes())for p in sorted(HERE.rglob('*.py'))},production_hashes={name:sha((ROOT/name).read_bytes())for name in('openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/backend/ra.py','openassembler/backend/contracts.py')},ptx_sha256=sha(text.encode()),scope='Isolated static local-package normalization: three helper source routes, independent full modulo64/FP32 bits, input immutability/tail guards and PTX/normal/poisoned SASS; six negative controls. No toolkit/GPU/build/remote/dependency/preserved-source-diagnostic or production claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['executions'],'complete executions;',len(negatives),'negative controls')
