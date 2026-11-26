"""Compare independent complete references with staged library resolver and code."""
import hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install()
from fixtures import cases,values,expected,SENTINEL
from openassembler.ptx.link import resolve
from openassembler.ptx.parse import parse_module
from openassembler import build
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 rows=[];guard=b'\xa5'*64;lanes=128;inp,out=0x100010000,0x100000000
 for case in cases():
  try:
   merged,contract=resolve([('entry',case['root'])]+[(f'library{i}',text)for i,text in enumerate(case['libraries'])]);module=parse_module(merged);kernel=module.kernels['probe']
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,registers=build.generate(kernel,'sm_80',level)
    for seed in(21941,21942):
     data=values(seed,lanes);before=guard+struct.pack('<128Q',*data)+guard
     for n in(0,91,128):
      wanted=guard+expected(data,case,n)+guard
      for engine in('ptx','normal','poisoned'):
       memory=Memory();memory.write_bytes(inp-64,before);memory.write_bytes(out-64,guard+struct.pack('<Q',SENTINEL)*lanes+guard)
       if engine=='ptx':Interp(module,'probe',PLaunch(block=(128,1,1),params={'out':out,'input':inp,'n':n}),memory).run()
       else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',out,inp,n),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
       actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual==wanted,(case['name'],level,n,seed,engine)
       assert bytes(memory.read(inp-64+i,1)for i in range(len(before)))==before
    rows.append(dict(name=case['name'],level=level,passed=True,executions=18,registers=registers,linked_ptx_sha256=sha(merged.encode()),contract=contract));print(case['name'],level,'passed',flush=True)
  except Exception as error:rows.append(dict(name=case['name'],passed=False,error=type(error).__name__+': '+str(error)));print(rows[-1],flush=True)
 result=dict(schema=1,passed=all(r['passed']for r in rows),results=rows,source_hashes={name:sha((HERE/name).read_bytes())for name in('link.py','fixtures.py','candidate-build.py','candidate-cppfront.py','candidate-cli.py','candidate-native_ownership.py','candidate-remote_build.py')},scope='Staged source-library resolver and independent complete PTX/normal/poisoned SASS bytes; not independent original linking or hardware evidence')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['passed'],len(rows))
