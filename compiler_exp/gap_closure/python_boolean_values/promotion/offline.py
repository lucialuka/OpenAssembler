"""Whole original source/typed truth/atomic snapshot outputs at all compiler levels."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,NAMES,raw_input,initial,expected,expected_counter,width
from openassembler.pyfront import compile_file
from openassembler.pyfront.modules import contract_from_ptx
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 inputs=[HERE/n for n in('candidate-pyfront.py','candidate-backend.py','candidate-modules.py','candidate.py','fixtures.py','offline.py')];frozen={p.name:sha(p.read_bytes())for p in inputs};rows=[];ap,out,cp=0x100010000,0x100000000,0x100020000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-original-chain-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case);text,names=compile_file(source);assert names==list(NAMES);graph=contract_from_ptx(text);assert graph and graph['static_values'] and graph['boolean_values'];assert{r['path']:r['sha256']for r in graph['files']}=={name:sha(t.encode())for name,t in case['files'].items()};module=parse_module(text)
   for name in names:
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level);assert words
     for seed in(42131,42132):
      before=guard+raw_input(name,seed,128)+guard
      for n in(0,91,128):
       wanted=guard+expected(name,seed,n,128)+guard;counter=guard+expected_counter(name,seed,n,128)+guard
       for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+initial(name,128)+guard);memory.write_bytes(cp-64,guard+bytes(512)+guard)
        if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'count':cp,'n':n}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,out,cp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        for address,data in((out,wanted),(ap,before),(cp,counter)):
         assert bytes(memory.read(address-64+i,1)for i in range(len(data)))==data,(case['name'],name,level,seed,n,engine,address)
     rows.append(dict(configuration=case['name'],entry=name,level=level,registers=regs,executions=18,passed=True))
 assert frozen=={p.name:sha(p.read_bytes())for p in inputs}
 report=dict(schema=1,passed=True,results=rows,executions=4320,source_hashes=frozen,scope='Three original source graphs/twenty typed/helper/effectful Boolean values and operand selections, independent complete truth/single-evaluation/short-circuit atomic/memory snapshots, inactive tails, unchanged inputs and guards at four PTX/normal/poisoned levels/two seeds/three lengths; hardware separate')
 (HERE/'candidate-offline.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed 4320 original Boolean-value executions; 240 rows')
