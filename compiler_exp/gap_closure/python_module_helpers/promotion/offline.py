"""Original source graph numerical, source hash, dependency and poison checks."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,raw_data,initial,expected
from openassembler.pyfront import compile_file,compile_kernel,find_kernel
from openassembler.pyfront.modules import contract_from_ptx
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 rows=[];ap,bp,out=0x100010000,0x100020000,0x100000000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-module-graph-offline-')as temporary:
  for case in configurations():
   root=Path(temporary)/case['name'];source=write_tree(root,case);options={'source_root':root}if case['explicit']else{}
   text,names=compile_file(source,**options);graph=contract_from_ptx(text);assert names==['signed','unsigned','floating']and graph
   assert graph['entry']==case['entry']and{row['path']:row['sha256']for row in graph['files']}=={name:sha(text.encode())for name,text in case['files'].items()}
   module=parse_module(text)
   for name in names:
    ref=find_kernel(source,name,**options);single=compile_kernel(ref);assert single==compile_kernel(ref)
    sm=parse_module(single);assert list(sm.kernels)==[name]
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,registers=build.generate(module.kernels[name],'sm_80',level);assert words
     _,_,single_words,_,single_registers=build.generate(sm.kernels[name],'sm_80',level);assert words==single_words and registers==single_registers
     for seed in(31881,31882):
      a,b=raw_data(name,seed,128)
      for n in(0,91,128):
       wanted=guard+expected(name,seed,n,128)+guard
       for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(ap-64,guard+a+guard);memory.write_bytes(bp-64,guard+b+guard);memory.write_bytes(out-64,guard+initial(name,128)+guard)
        if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'b':bp,'out':out,'n':n}),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQQI',ap,bp,out,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        assert bytes(memory.read(out-64+i,1)for i in range(len(wanted)))==wanted,(case['name'],name,level,seed,n,engine)
        for address,data in((ap,a),(bp,b)):assert bytes(memory.read(address-64+i,1)for i in range(len(data)+128))==guard+data+guard
     rows.append(dict(configuration=case['name'],entry=name,level=level,executions=18,passed=True,registers=registers,ptx_sha256=sha(text.encode())))
 result=dict(schema=1,passed=True,results=rows,executions=sum(r['executions']for r in rows),source_hashes={p.name:sha(p.read_bytes())for p in[HERE/name for name in('modules.py','candidate-pyfront.py','candidate-helpers.py','candidate-build.py','candidate-frontend.py','candidate-ownership.py','candidate-remote.py','candidate-cli.py','candidate.py','fixtures.py','offline.py')]},scope='Five original source-tree variants, exact dependency hashes including package initializers, repeated compile_kernel snapshots and whole independent modulo64/FP32 guarded input/output through all four PTX/normal/poisoned levels; remote transport/GPU separate')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['executions'],'executions;',len(rows),'rows')
