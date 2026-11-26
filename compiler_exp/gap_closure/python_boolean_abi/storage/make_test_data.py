"""Freeze independent complete bytes and original source trees for production regressions."""
import hashlib,json
from pathlib import Path
import fixtures,shape_cases
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 suites=[];lanes=128;seed=67517
 for module in(fixtures,shape_cases):
  cases=[]
  for name in module.NAMES:
   rows=[]
   for flag in(False,True):
    for n in(0,91,128):
     after,out,counter=module.references(name,seed,flag,n,lanes);values=module.values(flag,n,out=0x100010000 if name=='alias_invert'else 0x100000000)
     rows.append(dict(flag=flag,n=n,values=values,packed_hex=module.pack(values).hex(),input_after_hex=after.hex(),expected_hex=out.hex(),counter_hex=counter.hex()))
   cases.append(dict(name=name,input_hex=module.raw_input(seed,lanes).hex(),initial_hex=module.initial(lanes).hex(),initial_counter_hex=module.initial_counter(lanes).hex(),rows=rows))
  suites.append(dict(name=module.__name__,configurations=module.configurations(),cases=cases))
 result=dict(schema=1,suites=suites,reference_hashes={module.__name__+'.py':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()for module in(fixtures,shape_cases)},scope='Frozen original sources and independent complete byte references; canonical stores and noncanonical raw inputs, shaped/shared/lazy/alias paths; no GPU claim')
 (HERE/'boolean_storage_fixtures.json').write_text(json.dumps(result,indent=2)+'\n');print('Frozen 22 Boolean storage kernels across three source trees and six full-state rows per kernel')
