"""Record remaining Boolean comparison boundaries after scalar/storage admission."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported
if __name__=='__main__':
 rows=[]
 for name in('scalar_equal','storage_order','helper_equal','chained_equal'):
  try:compile_kernel(find_kernel(HERE/'source.py',name))
  except NotSupported as error:
   assert 'comparing comparison results'in str(error)and error.path and error.lineno;rows.append(dict(entry=name,rejected=True,diagnostic=str(error)))
  else:raise AssertionError(name)
 names=('openassembler/pyfront/compile.py','openassembler/pyfront/helpers.py','openassembler/pyfront/modules.py')
 result=dict(schema=1,passed=True,production_fix=False,results=rows,source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in names},artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'source.py',Path(__file__))},scope='Four original scalar/storage/helper/chained Boolean comparison rejections after byte storage admission; no candidate or numerical/hardware support claim')
 (HERE/'baseline.json').write_text(json.dumps(result,indent=2)+'\n');print('Recorded four original Boolean comparison rejections')
