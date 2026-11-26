"""Compile corrected original sources and compare admitted generic typed executions."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import candidate
candidate.install()
from openassembler.pyfront import compile_kernel,find_kernel
from openassembler.ptx.parse import parse_module
from openassembler import build
from openassembler.verify import run_and_diff,parameter_contract_from_ptx
if __name__=='__main__':
 rows=[]
 for name in('raw_array','shaped_buffer','shared_flags'):
  source=compile_kernel(find_kernel(HERE/'source.py',name));module=parse_module(source);kernel=module.kernels[name]
  (HERE/(name+'.ptx')).write_text(source)
  for level in(0,1,2,3):
   resource={};code,labels,words,_,registers=build.generate(kernel,'sm_80',level,resource_info=resource)
   contract=parameter_contract_from_ptx(source,kernel,allow_buffers=True)
   if name!='shaped_buffer':
    for seed in(0,1):
     diff=run_and_diff(module,name,code,labels,n_elems=128,grid=(1,1,1),block=(128,1,1),seed=seed,parameter_contract=contract)
     assert not diff,(name,level,seed,diff[:5])
   rows.append(dict(entry=name,level=level,registers=registers,local_bytes=resource['local_bytes'],words_sha256=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest(),passed=True))
 result=dict(schema=1,passed=True,results=rows,candidate_hashes={name:hashlib.sha256((HERE/('candidate-'+name+'.py')).read_bytes()).hexdigest()for name in json.loads((HERE/'inputs.json').read_text())},scope='Original corrected source compile at all four levels; raw/shared typed PTX/normal/poisoned parity for two Boolean seeds. Shaped Buffer numerical behavior requires the separate independent byte/layout reference; this smoke is not a hardware or promotion gate')
 (HERE/'smoke.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed original three compiles / two generic typed PTX/native parity cases at four levels')
