"""Source-defined typed Python/PTX/CUDA low64 products with full byte references."""
import argparse,hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from python_fixtures import cases,values,expected,SENTINEL,CHOICE

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 from candidate import install
 install(baseline=not args.candidate)
 from openassembler.pyfront import compile_file
 from openassembler import build
 from openassembler.ptx.parse import parse_module
 from openassembler.ptx.interp import Interp,Launch as PLaunch
 from openassembler.emu.machine import Memory,Machine,Launch
 from openassembler.emu.program import load_listing
 from tests.backend_harness import listing
 rows=[];lanes=128;inp,out=0x100010000,0x100000000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-low64-source-')as temporary:
  for case in cases():
   try:
    if case['mode']=='python':
     path=Path(temporary)/(case['name']+'.py');path.write_text(case['python_source']);ptx,names=compile_file(path);assert names==[case['entry']]
    elif case['mode']=='cuda':ptx=(HERE/'cuda.ptx').read_text()
    else:ptx=case['source']
    module=parse_module(ptx);kernel=module.kernels[case['entry']]
    for level in('safe','mined','sched','opt'):
     code,labels,words,_,registers=build.generate(kernel,'sm_80',level);assert words
     for seed in(27391,27392):
      data=values(seed,lanes);before=guard+struct.pack('<256Q',*data)+guard
      for n in(0,91,128):
       wanted=guard+expected(data,case,n,lanes)+guard
       if case['mode']=='cuda':params=[out,inp,inp+lanes*8,n];raw=struct.pack('<QQQI',*params)
       else:params=[out,inp,n,CHOICE];raw=struct.pack('<QQI4xQ',*params)
       mapping=dict(zip((name for _,name in kernel.params),params))
       for engine in('ptx','normal','poisoned'):
        memory=Memory();memory.write_bytes(inp-64,before);memory.write_bytes(out-64,guard+struct.pack('<Q',SENTINEL)*lanes+guard)
        if engine=='ptx':Interp(module,case['entry'],PLaunch(block=(lanes,1,1),params=mapping),memory).run()
        else:Machine(load_listing(listing(code,labels)),Launch(block=(lanes,1,1),params=raw,poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
        actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual==wanted,(case['name'],level,n,seed,engine,next((i,a,b)for i,(a,b)in enumerate(zip(actual,wanted))if a!=b))
        assert bytes(memory.read(inp-64+i,1)for i in range(len(before)))==before
     rows.append(dict(name=case['name'],level=level,passed=True,executions=18,registers=registers,ptx_sha256=sha(ptx.encode())))
   except Exception as error:rows.append(dict(name=case['name'],passed=False,error=type(error).__name__+': '+str(error)))
 result=dict(schema=1,passed=all(r['passed']for r in rows)if args.candidate else not any(r['passed']for r in rows),results=rows,source_hashes={p.name:sha(p.read_bytes())for p in[HERE/name for name in('candidate-ra.py','candidate-contracts.py','candidate-pyfront.py','baseline-ra.py','baseline-contracts.py','baseline-pyfront.py','python_fixtures.py','fixtures.py','cuda.cu','cuda.ptx','candidate.py','offline.py')]},scope='Independent full-bit products from original typed Python/PTX and actual emitted CUDA PTX, all levels/PTX/normal/poisoned SASS, full guarded inputs/outputs/tails; no native hardware claim')
 (HERE/('candidate-offline.json'if args.candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['passed'],len(rows),sum(r.get('executions',0)for r in rows))
 if not result['passed']:
  print([r for r in rows if not r['passed']]);raise SystemExit(1)
