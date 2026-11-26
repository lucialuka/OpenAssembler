"""Original byte-parameter isolation with a deliberately nonzero adjacent byte."""
import hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler.emu.machine import Memory,Machine,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 source=HERE/'narrow.ptx';text=source.read_text();module=parse_module(text);out=0x100000000;rows=[]
 for level in('safe','mined','sched','opt'):
  code,labels,words,_,regs=build.generate(module.kernels['narrow'],'sm_80',level)
  for engine in('ptx','normal','poisoned'):
   memory=Memory();memory.write_bytes(out,bytes(4))
   if engine=='ptx':Interp(module,'narrow',PLaunch(params={'flag':1,'neighbor':165,'out':out}),memory).run()
   else:Machine(load_listing(listing(code,labels)),Launch(params=struct.pack('<BB6xQ',1,165,out),poison=0xdeadbeef if engine=='poisoned'else None),memory).run()
   observed=memory.read(out,4);assert observed==(1 if engine=='ptx'else 42241),(level,engine,observed)
   rows.append(dict(level=level,engine=engine,observed=observed,expected=1,matches_reference=observed==1))
 result=dict(schema=1,passed=True,production_fix=False,results=rows,counterexamples=8,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backend_sha256=hashlib.sha256((ROOT/'openassembler/backend/ra.py').read_bytes()).hexdigest(),scope='Original admitted ld.param.u8 writes low-byte value 1; nonzero adjacent byte 165 exposes whole-word constant-bank alias in all four OA normal/poisoned emulators, output 42241. PTX interpretation requires 1. No GPU, parameter-fix candidate or Boolean ABI production claim')
 (HERE/'narrow-baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded eight narrow-parameter neighboring-byte counterexamples')
