"""Generated single-evaluation/short-circuit chains with independently counted atomics."""
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
RELATIONS={'<':lambda a,b:a<b,'<=':lambda a,b:a<=b,'>':lambda a,b:a>b,'>=':lambda a,b:a>=b,'==':lambda a,b:a==b,'!=':lambda a,b:a!=b}
if __name__=='__main__':
 rng=random.Random(86713);inputs=[-2147483648,-19,-3,-1,0,1,3,7,19,2147483647];rows=[];ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64;before=guard+struct.pack('<10i',*inputs)+guard
 paths=[HERE/n for n in('candidate-pyfront.py','candidate-modules.py','control_properties.py')];frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 with tempfile.TemporaryDirectory(prefix='oa-chain-properties-')as temporary:
  for index in range(128):
   operators=[rng.choice(list(RELATIONS))for _ in range(rng.randint(2,5))];middle=[rng.choice(('x','atomic','literal'))for _ in range(len(operators)-1)];numbers=[rng.randint(-3,7)for _ in range(len(operators)+1)];negated=bool(rng.getrandbits(1));last_atomic=bool(rng.getrandbits(1))
   operands=['x']+[('x'if kind=='x'else'atomic_add(count,i,1)'if kind=='atomic'else str(numbers[j+1]))for j,kind in enumerate(middle)]+['atomic_add(count,i,1)'if last_atomic else str(numbers[-1])]
   expression=operands[0]+''.join(' '+operator+' '+operand for operator,operand in zip(operators,operands[1:]));condition='not ('+expression+')'if negated else expression
   source='from openassembler.pyfront import *\n@kernel\ndef k(a:Array(i32),out:Array(i32),count:Array(i32)):\n    i=tid_x()\n    x=a[i]\n    value=11\n    if '+condition+':\n        value=17\n    out[i]=value\n'
   expected=[];counter=[]
   for x in inputs:
    value=x;calls=0;truth=True
    for position,operator in enumerate(operators):
     operand=operands[position+1]
     if operand=='atomic_add(count,i,1)':right=calls;calls+=1
     elif operand=='x':right=x
     else:right=int(operand)
     if not RELATIONS[operator](value,right):truth=False;break
     value=right
    if negated:truth=not truth
    expected.append(17 if truth else 11);counter.append(calls)
   path=Path(temporary)/'source.py';path.write_text(source);text,_=compile_file(path);module=parse_module(text);wanted=guard+struct.pack('<10i',*expected)+guard;counts=guard+struct.pack('<10i',*counter)+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(40)+guard);memory.write_bytes(cp-64,guard+bytes(40)+guard)
     if engine=='ptx':Interp(module,'k',PLaunch(block=(10,1,1),params={'a':ap,'out':out,'count':cp}),memory).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(10,1,1),params=struct.pack('<QQQ',ap,out,cp),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
     for address,data in((ap,before),(out,wanted),(cp,counts)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)))==data,(index,level,engine,address)
   rows.append(dict(index=index,source=source,operators=operators,operands=operands,negated=negated,expected_outputs=expected,expected_atomic_counts=counter,passed=True))
 assert frozen=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 report=dict(schema=1,passed=True,seed=86713,inputs=inputs,results=rows,executions=1536,source_hashes=frozen,script_sha256=frozen['control_properties.py'],scope='128 original 2..5-link comparison/negation/literal/memory-effect sequences with independent relational dispatch and per-thread atomic evaluation counters, full input/output/count/guard bytes at four PTX/normal/poisoned levels; no additional GPU or universal expression proof')
 (HERE/'control-properties.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 128 chain/effect combinations; 1536 executions')
