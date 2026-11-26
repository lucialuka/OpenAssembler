"""Preserve the three initial source rejections without rewriting the original observation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from candidate import install
install('baseline')
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported

if __name__=='__main__':
 rows=[]
 for name in('scalar_truth','scalar_helper','scalar_guard'):
  try:compile_kernel(find_kernel(HERE/'source.py',name))
  except NotSupported as error:
   assert 'unsupported type annotation'in str(error)and error.path and error.lineno;rows.append(dict(entry=name,rejected=True,diagnostic=str(error)))
  else:raise AssertionError(name)
 paths=[HERE/('baseline-'+suffix+'.py')for suffix in json.loads((HERE/'inputs.json').read_text())]+[HERE/'source.py',Path(__file__)]
 data=dict(schema=1,passed=True,results=rows,original_record_sha256=hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest(),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},scope='Three initial logical Boolean source rejections replayed using exact frozen source/build/verify/ownership/runner snapshots; original baseline observation unchanged, no GPU claim')
 (HERE/'original-replay.json').write_text(json.dumps(data,indent=2)+'\n');print('Replayed three exact initial Boolean source parameter rejections')
