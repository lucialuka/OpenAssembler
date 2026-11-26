"""Complete original Boolean storage against independent raw-byte references."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import configurations,write_tree,NAMES,raw_input,initial,initial_counter,references,values,pack
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.verify import parameter_contract_from_ptx
from openassembler.pyfront.buffers import contract_from_ptx
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 paths=[HERE/('candidate-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[HERE/n for n in('candidate.py','fixtures.py','offline.py')];frozen={p.name:sha(p)for p in paths}
 rows=[];executions=0;guard=b'\xa5'*64;ap,out,cp=0x100010000,0x100000000,0x100020000;lanes=384
 with tempfile.TemporaryDirectory(prefix='oa-bool-storage-')as temporary:
  for case in configurations():
   source=write_tree(Path(temporary)/case['name'],case);text,names=compile_file(source,source_root=source.parent if case.get('explicit')else None);assert names==list(NAMES),(case['name'],names);module=parse_module(text)
   for name in names:
    contract=parameter_contract_from_ptx(text,module.kernels[name],allow_buffers=True);assert contract['parameters'][0]['dtype']=='bool';contract_from_ptx(text,module.kernels[name])
    for level in('safe','mined','sched','opt'):
     resources={};code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level,resource_info=resources);assert resources['local_bytes']==0
     for seed in(67517,67518):
      for flag in(False,True):
       for n in(0,1,259,384):
        wanted_in,wanted_out,wanted_counter=references(name,seed,flag,n,lanes);output=ap if name=='alias_invert'else out;args=values(flag,n,out=output)
        for engine in('ptx','normal','poisoned'):
         memory=Memory();memory.write_bytes(ap-64,guard+raw_input(seed,lanes)+guard)
         if output!=ap:memory.write_bytes(out-64,guard+initial(lanes)+guard)
         memory.write_bytes(cp-64,guard+initial_counter(lanes)+guard)
         if engine=='ptx':
          for block in range(3):Interp(module,name,PLaunch(grid=(3,1,1),block=(128,1,1),ctaid=(block,0,0),params=args),memory).run()
         else:
          program=load_listing(listing(code,labels))
          for block in range(3):Machine(program,Launch(grid=(3,1,1),block=(128,1,1),ctaid=(block,0,0),params=pack(args),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
         for address,data in((ap,guard+wanted_in+guard),(output,guard+wanted_out+guard),(cp,guard+wanted_counter+guard)):
          actual=bytes(memory.read(address-64+i,1)for i in range(len(data)))
          assert actual==data,(case['name'],name,level,seed,flag,n,engine,hex(address),next((i for i,(a,b)in enumerate(zip(actual,data))if a!=b),None))
         executions+=1
     rows.append(dict(configuration=case['name'],entry=name,level=level,registers=regs,passed=True));print(case['name'],name,level,flush=True)
 assert frozen=={p.name:sha(p)for p in paths}
 (HERE/'candidate-offline.json').write_text(json.dumps(dict(schema=1,passed=True,executions=executions,results=rows,source_hashes=frozen,scope='18 original Boolean raw/shaped/shared storage kernels across three unexecuted source trees, noncanonical input byte decoding, canonical stores, sparse/transposed/broadcast/bounds storage, shared mixed alignment/permutations/barriers, raw alias and lazy atomic effects; independent full input/output/counter/guard/tail bytes at four PTX/normal/poisoned levels; no hardware/promotion claim'),indent=2)+'\n');print('Passed',executions,'complete storage executions;',len(rows),'rows')
