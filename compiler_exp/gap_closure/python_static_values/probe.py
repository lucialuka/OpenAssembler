"""Independent source-defined scalar/tuple hypothesis, with normalized source identity."""
import hashlib,json,struct,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from constexpr import normalize,Rejected
from openassembler.pyfront import compile_file
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp,Launch as PLaunch
from openassembler import build
from openassembler.emu.machine import Machine,Memory,Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing
if __name__=='__main__':
 text=(HERE/'corpus/helpers.py').read_text();normalized,constants=normalize(text);(HERE/'normalized-helpers.py').write_text(normalized);rows=[];guard=b'\xa5'*64;ap,out=0x100010000,0x100000000
 with tempfile.TemporaryDirectory(prefix='oa-static-values-')as temp:
  root=Path(temp);(root/'helpers.py').write_text(normalized);(root/'entry.py').write_bytes((HERE/'corpus/entry.py').read_bytes());ptx,names=compile_file(root/'entry.py');(HERE/'candidate.ptx').write_text(ptx);module=parse_module(ptx)
  for name in names:
   if name=='floating':values=[(i-63)/8 for i in range(128)];wanted=[v*0.5 if i<91 else-12345.0 for i,v in enumerate(values)];fmt='f';sentinel=-12345.0
   else:
    values=[(0x123456789abcdef*i+((1<<63)if i&1 else 0))&((1<<64)-1)for i in range(128)];wanted=[((v*3-(1<<32)+7)if name=='signed'else(v*5+7))&((1<<64)-1)if i<91 else 0x13579bdf2468ace0 for i,v in enumerate(values)];fmt='Q';sentinel=0x13579bdf2468ace0
   original=struct.pack('<128'+fmt,*values);expected=guard+struct.pack('<128'+fmt,*wanted)+guard
   for level in('safe','mined','sched','opt'):
    code,labels,words,_,regs=build.generate(module.kernels[name],'sm_80',level);assert words
    for engine in('ptx','normal','poisoned'):
     mem=Memory();mem.write_bytes(ap-64,guard+original+guard);mem.write_bytes(out-64,guard+struct.pack('<128'+fmt,*([sentinel]*128))+guard)
     if engine=='ptx':Interp(module,name,PLaunch(block=(128,1,1),params={'a':ap,'out':out,'n':91}),mem).run()
     else:Machine(load_listing(listing(code,labels)),Launch(block=(128,1,1),params=struct.pack('<QQI',ap,out,91),poison=0xdeadbeef if engine=='poisoned'else None),mem).run()
     assert bytes(mem.read(out-64+i,1)for i in range(len(expected)))==expected
     assert bytes(mem.read(ap-64+i,1)for i in range(len(original)+128))==guard+original+guard
     rows.append(dict(entry=name,level=level,engine=engine,passed=True))
 negatives=[]
 for name,source in [('call','from typing import Final\nA:Final=side_effect()'),('rebind','from typing import Final\nA:Final=3\nA=7'),('annotated_rebind','from typing import Final\nA:Final=3\nA:int=7'),('mutate','from typing import Final\nA:Final=(3,5)\nA[0]=7'),('depth','from typing import Final\nA:Final=1<<99999999'),('capture','from typing import Final\nA:Final=UNKNOWN'),('list','from typing import Final\nA:Final=[1,2]')]:
  try:normalize(source)
  except Rejected as error:negatives.append(dict(name=name,rejected=True,error=str(error)))
  else:raise AssertionError(name)
 result=dict(schema=1,passed=True,executions=len(rows),results=rows,negative_controls=negatives,constants=constants,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'constexpr.py',HERE/'probe.py',HERE/'normalized-helpers.py')},scope='Isolated explicit-Final scalar/tuple normalization to a separate source tree, 36 whole guarded PTX/normal/poisoned executions and six negative controls. Normalized dependency identity is distinct from original source. Original diagnostic/dependency/host-effect integration, import aliases, comprehensive literal arithmetic and GPU gates remain open; no production claim')
 (HERE/'candidate-offline.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(rows),'executions;',len(negatives),'negative controls')
