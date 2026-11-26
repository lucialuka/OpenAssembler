"""Independent Boolean relation truth tables, chains and once-only atomic effects."""
import hashlib,json,operator,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler.ptx.parse import parse_module
from openassembler import build
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
OPS={'eq':('==',operator.eq),'ne':('!=',operator.ne),'lt':('<',operator.lt),'le':('<=',operator.le),'gt':('>',operator.gt),'ge':('>=',operator.ge)}
HEADER='from openassembler.pyfront import *\nraise RuntimeError("Boolean comparison discovery must never execute")\n'
def source():
 text=HEADER+'''@device
def same(a:bool,b:bool)->bool:
    return a==b
'''
 for name,(symbol,_)in OPS.items():
  for variant,expression in [('pair',f'a[i] {symbol} flag'),('true',f'a[i] {symbol} True'),('false',f'False {symbol} a[i]'),('branch',f'a[i] {symbol} flag')]:
   body=f'out[i] = {expression}'if variant!='branch'else f'if {expression}:\n    out[i]=True\nelse:\n    out[i]=False'
   text+=f'\n@kernel\ndef {name}_{variant}(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):\n    i=tid_x()\n    if i<n:\n'+''.join('        '+line+'\n'for line in body.splitlines())
 for name,expression in [('eq_chain','a[i] == flag == bool(atomic_add(counter,i,1))'),('lt_chain','a[i] < flag <= bool(atomic_add(counter,i,1))'),('middle_once','a[i] == bool(atomic_add(counter,i,1)) == flag'),('helper','same(a[i],flag)'),('predicate_compare','(i<31)==flag')]:
  text+=f'\n@kernel\ndef {name}(a:Array(bool),out:Array(bool),counter:Array(i32),flag:bool,n:i32):\n    i=tid_x()\n    if i<n:\n        out[i]={expression}\n'
 return text
def expected(name,raw,flag,n):
 out=bytearray([0xd3]*len(raw));counter=bytearray(struct.pack('<'+str(len(raw))+'i',*((i%5)-2 for i in range(len(raw)))))
 for i in range(n):
  a=bool(raw[i]);old=(i%5)-2;effect=0
  if name=='eq_chain':effect=int(a==flag);value=a==flag and flag==bool(old)
  elif name=='lt_chain':effect=int(a<flag);value=a<flag and flag<=bool(old)
  elif name=='middle_once':effect=1;value=a==bool(old)==flag
  elif name=='helper':value=a==flag
  elif name=='predicate_compare':value=(i<31)==flag
  else:
   op,variant=name.split('_');_,function=OPS[op];value=function(False,a)if variant=='false'else function(a,True if variant=='true'else flag)
  out[i]=int(value)
  if effect:struct.pack_into('<i',counter,4*i,old+1)
 return bytes(out),bytes(counter)
if __name__=='__main__':
 text=source();(HERE/'matrix-source.py').write_text(text);rows=[];executions=0;guard=b'\xa5'*64;ap,out,cp=0x100010000,0x100000000,0x100020000;raw=bytes((0,1,2,127,128,254,255,0)*16);initial_counter=struct.pack('<128i',*((i%5)-2 for i in range(128)))
 with tempfile.TemporaryDirectory(prefix='oa-bool-comparison-')as temporary:
  path=Path(temporary)/'source.py';path.write_text(text);ptx,names=compile_file(path);module=parse_module(ptx)
  for name in names:
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level);program=load_listing(listing(code,labels))
    for flag in(False,True):
     for n in(0,1,91,128):
      wanted,wanted_counter=expected(name,raw,flag,n);params=dict(a=ap,out=out,counter=cp,flag=int(flag),n=n);packed=struct.pack('<QQQB3xi',ap,out,cp,int(flag),n)
      for engine in('ptx','normal','poisoned'):
       memory=Memory();memory.write_bytes(ap-64,guard+raw+guard);memory.write_bytes(out-64,guard+bytes([0xd3]*128)+guard);memory.write_bytes(cp-64,guard+initial_counter+guard)
       if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params=params),memory).run()
       else:Machine(program,Launch(block=(128,1,1),params=packed,poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
       for address,data in((ap,guard+raw+guard),(out,guard+wanted+guard),(cp,guard+wanted_counter+guard)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)))==data,(name,level,flag,n,engine)
       executions+=1
    rows.append(dict(entry=name,level=level,registers=registers,passed=True))
 paths=[HERE/name for name in('baseline-pyfront.py','candidate-pyfront.py','candidate.py','matrix-source.py','offline.py')]
 result=dict(schema=1,passed=True,results=rows,executions=executions,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='29 original Boolean relation/literal/branch/helper/predicate/chain cases, independent complete truth/effect/guard/tail/input bytes at all four PTX/normal/poisoned levels; no GPU or production support claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',executions,'complete Boolean comparison executions;',len(rows),'rows')
