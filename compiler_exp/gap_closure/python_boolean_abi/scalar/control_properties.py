"""Generated scalar/Boolean value trees with independent lazy atomic/read semantics."""
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
rng=random.Random(67439)
def numeric(depth):
 if depth==0 or rng.random()<0.3:return[rng.choice(('x','atomic','read','literal')),rng.randint(-3,7)]
 return[rng.choice(('and','or')),numeric(depth-1),numeric(depth-1)]
def boolean(depth):
 if depth==0:return[rng.choice(('truth','chain','notnumeric','first','second')),numeric(1)]
 return[rng.choice(('and','or')),boolean(depth-1),boolean(depth-1)]if rng.random()<0.7 else['not',boolean(depth-1)]
def render(tree):
 op=tree[0]
 if op in('first','second'):return op
 if op=='x':return'x'
 if op=='atomic':return'atomic_add(count,i,1)'
 if op=='read':return'count[i]'
 if op=='literal':return str(tree[1])
 if op in('and','or'):return'('+render(tree[1])+' '+op+' '+render(tree[2])+')'
 if op=='truth':return'bool('+render(tree[1])+')'
 if op in('not','notnumeric'):return'(not '+render(tree[1])+')'
 if op=='chain':return'(-3<'+render(tree[1])+'<7)'
 raise AssertionError(op)
def evaluate(tree,x,state,first,second):
 op=tree[0]
 if op=='first':return first
 if op=='second':return second
 if op=='x':return x
 if op=='atomic':value=state[0];state[0]+=1;return value
 if op=='read':return state[0]
 if op=='literal':return tree[1]
 if op=='and':
  left=evaluate(tree[1],x,state,first,second);return evaluate(tree[2],x,state,first,second)if left else left
 if op=='or':
  left=evaluate(tree[1],x,state,first,second);return left if left else evaluate(tree[2],x,state,first,second)
 value=evaluate(tree[1],x,state,first,second)
 if op=='truth':return bool(value)
 if op in('not','notnumeric'):return not value
 if op=='chain':return-3<value and value<7
 raise AssertionError(op)
if __name__=='__main__':
 inputs=[-2147483648,-19,-3,-1,0,1,3,7,19,2147483647];rows=[];ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64;before=guard+struct.pack('<10i',*inputs)+guard
 paths=[HERE/n for n in('candidate-pyfront.py','candidate-verify.py','candidate-build.py','candidate-ownership.py','candidate-runner.py','control_properties.py')];frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 with tempfile.TemporaryDirectory(prefix='oa-boolean-properties-')as temporary:
  for index in range(128):
   tree=boolean(3);expression=render(tree);source='from openassembler.pyfront import *\n@kernel\ndef k(a:Array(i32),out:Array(i32),count:Array(i32),first:bool,second:bool):\n    i=tid_x()\n    x=a[i]\n    value='+expression+'\n    out[i]=i32(value)\n';expected=[];counts=[]
   for first,second in((False,False),(False,True),(True,False),(True,True)):
    expected=[];counts=[]
    for x in inputs:
     state=[0];expected.append(int(evaluate(tree,x,state,first,second)));counts.append(state[0])
    path=Path(temporary)/'source.py';path.write_text(source);text,_=compile_file(path);module=parse_module(text);wanted=guard+struct.pack('<10i',*expected)+guard;counter=guard+struct.pack('<10i',*counts)+guard
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
     for engine in('ptx','normal','poisoned'):
      memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+bytes(40)+guard);memory.write_bytes(cp-64,guard+bytes(40)+guard)
      if engine=='ptx':Interp(module,'k',PLaunch(block=(10,1,1),params={'a':ap,'out':out,'count':cp,'first':first,'second':second}),memory).run()
      else:Machine(load_listing(listing(code,labels)),Launch(block=(10,1,1),params=struct.pack('<QQQBB',ap,out,cp,int(first),int(second)),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
      for address,data in((ap,before),(out,wanted),(cp,counter)):assert bytes(memory.read(address-64+j,1)for j in range(len(data)))==data,(index,level,engine,address)
    rows.append(dict(index=index,first=first,second=second,tree=tree,source=source,expected_outputs=expected,expected_atomic_counts=counts,passed=True))
 assert frozen=={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 report=dict(schema=1,passed=True,seed=67439,inputs=inputs,results=rows,executions=6144,source_hashes=frozen,script_sha256=frozen['control_properties.py'],scope='128 original Boolean parameter/effect trees and four flag pairs with independent recursive operand selection and atomic/read state, full output/counter/unchanged input/guard bits at four PTX/normal/poisoned levels; no additional GPU or universal type/effect claim')
 (HERE/'control-properties.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 128 Boolean parameter/effect trees and four flag pairs; 6144 executions')
