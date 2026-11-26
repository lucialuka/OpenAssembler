"""Next G05 boundary: explicit immutable declarations still reject as captures/memory."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel
if __name__=='__main__':
 rows=[]
 for name in('signed','unsigned','floating'):
  try:compile_kernel(find_kernel(HERE/'corpus/entry.py',name))
  except Exception as error:rows.append(dict(entry=name,rejected=True,error=type(error).__name__+': '+str(error)))
  else:rows.append(dict(entry=name,rejected=False))
 assert all(r['rejected']and('captured'in r['error']or'access memory'in r['error'])for r in rows)
 result=dict(schema=1,passed=True,results=rows,source_hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((HERE/'corpus').rglob('*.py'))},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Explicit typing.Final scalar/tuple source declarations rejected by current pure-helper capture/memory checks, without executing user entry/helper modules; next G05 research boundary, no candidate/admission claim')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded three Final scalar/tuple capture boundaries')
