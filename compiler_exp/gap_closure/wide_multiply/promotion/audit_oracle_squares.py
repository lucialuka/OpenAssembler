"""Independent source, frontend output and raw toolkit artifact custody."""
import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from square_fixtures import cases
from candidate import install

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 install()
 from openassembler.pyfront import compile_file
 path=HERE/'oracle-squares-probe.json';report=json.loads(path.read_text());assert report['passed']and report['script_sha256']==sha((HERE/'oracle_squares_probe.py').read_bytes())
 assert all(sha((HERE/n).read_bytes())==h for n,h in report['source_hashes'].items())
 fixtures={c['name']:c for c in cases()};assert len(report['results'])==len(fixtures)==4
 files={}
 for row in report['results']:
  case=fixtures[row['name']];source=HERE/'oracle'/(row['name']+'.ptx');binary=source.with_suffix('.cubin')
  original=source.with_suffix('.py'if case['mode']=='python'else'.cu')if case['mode']in('python','cuda')else source
  text=case.get('python_source',case.get('cuda_source',case.get('source')))
  assert original.read_text()==text and row['accepted']and sha(original.read_bytes())==row['source_sha256']and sha(source.read_bytes())==row['ptx_sha256']
  if case['mode']=='python':
   regenerated,names=compile_file(original);assert names==[case['entry']]and regenerated==source.read_text()
  assert sha(binary.read_bytes())==row['cubin_sha256']
  for artifact in{original,source,binary,source.with_suffix('.sass'),source.with_suffix('.tool.log')}:files[str(artifact.relative_to(HERE))]=sha(artifact.read_bytes())
 result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),artifact_hashes=files,script_sha256=sha(Path(__file__).read_bytes()),scope='Four square PTX/CUDA/Python sources, exact current Python-emitted PTX, independent toolkit acceptance and complete raw source/log/cubin/listing identities; no execution')
 (HERE/'oracle-squares-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(fixtures),'sources;',len(files),'raw artifacts')
