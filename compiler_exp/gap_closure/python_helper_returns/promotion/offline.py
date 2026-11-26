"""Full original-source numerical and exact source graph validation."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,NAMES,raw_input,initial,expected
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
 inputs=[HERE/n for n in('return_flow.py','candidate-pyfront.py','candidate-helpers.py','candidate-modules.py','candidate.py','fixtures.py','offline.py')]
 frozen={p.name:sha(p.read_bytes())for p in inputs}
 rows=[];ap,out=0x100010000,0x100000000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-static-original-')as temp:
  for case in configurations():
   root=Path(temp)/case['name'];source=write_tree(root,case);text,names=compile_file(source);assert names==list(NAMES)
   graph=contract_from_ptx(text);assert graph and graph['helpers']
   assert {r['path']:r['sha256']for r in graph['files']}=={p:sha(t.encode())for p,t in case['files'].items()}
   module=parse_module(text)
   for name in names:
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level);assert words
     for seed in(42131,42132):
      before=guard+raw_input(name,seed,128)+guard
      for n in(0,91,128):
       wanted=guard+expected(name,seed,n,128)+guard
       for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(ap-64,before);memory.write_bytes(out-64,guard+initial(name,128)+guard)
        if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':n}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(case['name'],name,level,seed,n,engine)
        assert bytes(memory.read(ap-64+i,1)for i in range(len(before)))==before
     rows.append(dict(configuration=case['name'],entry=name,level=level,executions=18,registers=regs,passed=True))
 assert frozen=={p.name:sha(p.read_bytes())for p in inputs},'candidate sources changed during run'
 result=dict(schema=1,passed=True,results=rows,executions=sum(r['executions']for r in rows),source_hashes=frozen,scope='Three original source trees, twenty entries, all four levels/PTX/normal/poisoned, exact graph hashes including original declarations, original return/helper identity, full numerical modulo64/FP32 output bits and guarded unchanged inputs; hardware separate')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['executions'],'executions;',len(rows),'rows')
