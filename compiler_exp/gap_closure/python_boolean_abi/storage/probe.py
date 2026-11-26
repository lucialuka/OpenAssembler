"""Locate Boolean storage boundaries separately from validated scalar launch roles."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.pyfront import compile_kernel,find_kernel,NotSupported

if __name__=='__main__':
 rows=[]
 for name in('raw_array','shaped_buffer','shared_flags'):
  try:compile_kernel(find_kernel(HERE/'source.py',name))
  except NotSupported as error:
   assert 'annotation'in str(error)and error.path and error.lineno;rows.append(dict(entry=name,rejected=True,diagnostic=str(error)))
  else:raise AssertionError(name)
 toolkit_path=HERE.parent/'baseline-toolkit.json';toolkit=json.loads(toolkit_path.read_text());assert toolkit['passed']
 assert all(hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==digest for name,digest in toolkit['artifact_hashes'].items())
 names=('openassembler/pyfront/compile.py','openassembler/pyfront/buffers.py','openassembler/verify.py','openassembler/runner.py','openassembler/native_ownership.py')
 data=dict(schema=1,passed=True,production_fix=False,results=rows,source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in names},artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(HERE/'source.py',Path(__file__))},original_toolkit_record_sha256=hashlib.sha256(toolkit_path.read_bytes()).hexdigest(),scope='Three original Boolean Array/Buffer/shared storage annotation rejections after scalar Boolean source/host admission, exact original independent CUDA byte-storage artifacts retained; no candidate, new numerical execution or GPU claim')
 (HERE/'baseline.json').write_text(json.dumps(data,indent=2)+'\n');print('Recorded three original Boolean Array/Buffer/shared storage rejections')
