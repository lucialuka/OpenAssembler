"""Replay retained watchdog results with the frozen frontend without rewriting history."""
import builtins,hashlib,importlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
if __name__=='__main__':
 baseline=json.loads((HERE/'baseline.json').read_text());assert baseline['passed']and baseline['counterexamples']==4
 frozen=HERE/'baseline-pyfront.py';assert hashlib.sha256(frozen.read_bytes()).hexdigest()==baseline['source_hashes']['openassembler/pyfront/compile.py']
 module=importlib.import_module('openassembler.pyfront.compile');exec(builtins.compile(frozen.read_bytes(),str(frozen),'exec'),module.__dict__)
 from openassembler.pyfront.compile import compile_file
 from openassembler import build
 from openassembler.ptx.parse import parse_module
 from openassembler.ptx.interp import Interp,Launch as PLaunch
 from openassembler.emu.machine import Memory,Machine,Launch
 from openassembler.emu.program import load_listing
 from tests.backend_harness import listing
 rows=[];out=0x100000000;guard=b'\xa5'*64
 with tempfile.TemporaryDirectory(prefix='oa-frozen-range-replay-')as temporary:
  for case in baseline['results']:
   path=Path(temporary)/(case['name']+'.py');path.write_text(case['source']);text,_=compile_file(path);ptx=parse_module(text)
   for expected in case['observations']:
    level,engine=expected['level'],expected['engine'];memory=Memory();memory.write_bytes(out-64,guard+bytes(24)+guard)
    if engine=='ptx':Interp(ptx,'k',PLaunch(params={'out':out}),memory).run(max_steps=2000)
    else:
     code,labels,_,_,_=build.generate(ptx.kernels['k'],'sm_80',level);Machine(load_listing(listing(code,labels)),Launch(params=struct.pack('<Q',out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run(max_steps=2000)
    actual=bytes(memory.read(out-64+i,1)for i in range(152));assert actual.hex()==expected['output_hex'];rows.append(dict(entry=case['name'],level=level,engine=engine,passed=True))
 result=dict(schema=1,passed=True,executions=72,results=rows,baseline_sha256=hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest(),frozen_frontend_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Exact 72 retained baseline watchdog output/guard bytes replayed with frozen frontend after production repair; original report remains unchanged; no new GPU or current production failure claim')
 (HERE/'baseline-replay.json').write_text(json.dumps(result,indent=2)+'\n');print('Replayed 72 frozen baseline observations without rewriting original evidence')
