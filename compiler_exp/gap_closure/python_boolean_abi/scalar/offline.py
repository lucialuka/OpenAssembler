"""Complete original Boolean source/host-layout matrix against independent bytes."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,NAMES,raw_input,initial,expected,expected_counter,values,pack
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

if __name__=='__main__':
 paths=[HERE/('candidate-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[HERE/n for n in('candidate.py','fixtures.py','offline.py')];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();frozen={p.name:sha(p)for p in paths}
 rows=[];executions=0;guard=b'\xa5'*64;ap,out,cp=0x100010000,0x100000000,0x100020000
 with tempfile.TemporaryDirectory(prefix='oa-logical-bool-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case);text,names=compile_file(source);assert names==list(NAMES);module=parse_module(text)
   for name in names:
    contract=parameter_contract_from_ptx(text,module.kernels[name]);assert any(row['dtype']=='bool'for row in contract['parameters'])
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level)
     for seed in(67311,67312):
      before=guard+raw_input(name,seed,128)+guard
      for first,second in((False,False),(False,True),(True,False),(True,True)):
       for n in(0,91,128):
        wanted=guard+expected(name,seed,first,second,n,128)+guard;counter=guard+expected_counter(name,seed,first,second,n,128)+guard;args=values(name,first,second,n)
        for engine in('ptx','normal','poisoned'):
         memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+initial(name,128)+guard);memory.write_bytes(cp-64,guard+bytes(512)+guard)
         if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params=args),memory).run()
         else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=pack(name,args),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
         for address,data in((ap,before),(out,wanted),(cp,counter)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)))==data,(case['name'],name,level,seed,first,second,n,engine,address)
         executions+=1
     rows.append(dict(configuration=case['name'],entry=name,level=level,registers=regs,passed=True))
 assert frozen=={p.name:sha(p)for p in paths}
 (HERE/'candidate-offline.json').write_text(json.dumps(dict(schema=1,passed=True,executions=executions,results=rows,source_hashes=frozen,scope='Three original source trees/twenty logical Boolean parameter kernels, all flag pairs/adjacent numeric and wide slots, helpers/reassignments/loops/raw FP32/effect paths, complete independent output/counter/input/guard/tails at four PTX/normal/poisoned levels; no hardware claim'),indent=2)+'\n');print('Passed',executions,'complete Boolean parameter executions;',len(rows),'rows')
