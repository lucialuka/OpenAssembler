"""Bounded watchdog counterexamples for finite Python ranges at i32 boundaries."""
import argparse,hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
CASES=[('positive_overflow',2147483646,2147483647,2),('negative_overflow',-2147483647,-2147483648,-2),
       ('large_positive_step',-2147483648,2147483647,2147483647),('minimum_step',2147483647,-2147483648,-2147483648),
       ('ordinary',0,7,3),('empty',0,1,-2)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
 if args.candidate:
  from candidate import install
  install()
 from openassembler.pyfront import compile_file
 from openassembler import build
 from openassembler.ptx.parse import parse_module
 from openassembler.ptx.interp import Interp,Launch as PLaunch
 from openassembler.emu.machine import Memory,Machine,Launch
 from openassembler.emu.program import load_listing
 from tests.backend_harness import listing
 rows=[];out=0x100000000;guard=b'\xa5'*64;mask=(1<<64)-1
 with tempfile.TemporaryDirectory(prefix='oa-range-overflow-')as temporary:
  for name,start,stop,step in CASES:
   source=f'''from openassembler.pyfront import *
@kernel
def k(out:Array(i64)):
    count=0
    total=i64(0)
    target=17
    for target in range({start},{stop},{step}):
        total+=i64(target)
        count+=1
        if count==8:
            break
    out[0]=total
    out[1]=i64(count)
    out[2]=i64(target)
'''
   path=Path(temporary)/(name+'.py');path.write_text(source);text,_=compile_file(path);module=parse_module(text)
   values=list(range(start,stop,step));assert len(values)<8
   wanted=guard+struct.pack('<3Q',sum(values)&mask,len(values),(values[-1]if values else 17)&mask)+guard
   observations=[]
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels['k'],'sm_80',level)
    for engine in('ptx','normal','poisoned'):
     memory=Memory();memory.write_bytes(out-64,guard+bytes(24)+guard)
     if engine=='ptx':Interp(module,'k',PLaunch(params={'out':out}),memory).run(max_steps=2000)
     else:Machine(load_listing(listing(code,labels)),Launch(params=struct.pack('<Q',out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run(max_steps=2000)
     actual=bytes(memory.read(out-64+i,1)for i in range(len(wanted)));assert actual[:64]==actual[-64:]==guard
     correct=actual==wanted
     assert correct if args.candidate else correct==(name in('ordinary','empty')),(name,level,engine,actual.hex(),wanted.hex())
     observations.append(dict(level=level,engine=engine,matches_reference=correct,observed_signed=list(struct.unpack('<3q',actual[64:-64])),output_hex=actual.hex()))
   rows.append(dict(name=name,source=source,range_values=values,expected_hex=wanted.hex(),observations=observations))
 result=dict(schema=1,passed=True,production_fix=False,candidate=args.candidate,results=rows,executions=72,counterexamples=0 if args.candidate else 4,source_hashes={n:sha(ROOT/n)for n in('openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/pyfront/modules.py','openassembler/pyfront/return_flow.py')},experiment_source_hashes={p.name:sha(p)for p in[Path(__file__)]+([HERE/'candidate.py',HERE/'candidate-pyfront.py']if args.candidate else[])},scope='Six bounded watchdog controls: four finite i32-range wrap counterexamples and two ordinary/empty controls. Exact Python range values and full guards/outputs at four PTX/normal/poisoned levels. Watchdog caps eight body trips; no unbounded hang, GPU, build, source-ownership or production claim')
 (HERE/('candidate-offline.json'if args.candidate else'baseline.json')).write_text(json.dumps(result,indent=2)+'\n');print('Passed',result['executions'],'bounded observations;',result['counterexamples'],'counterexamples')
