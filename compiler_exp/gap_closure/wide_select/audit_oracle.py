"""Recheck toolkit fixtures, acceptance and retained cubin/listing identities."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import cases

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'oracle-probe.json';report=json.loads(path.read_text());assert report['script_sha256']==sha((HERE/'oracle_probe.py').read_bytes())
 fixtures={c['name']:c for c in cases()};assert len(report['results'])==len(fixtures)==36
 files={}
 for row in report['results']:
  name=row['name'];source=HERE/'oracle'/(name+'.ptx');binary=source.with_suffix('.cubin');listing=source.with_suffix('.sass')
  assert row['accepted']and source.read_text()==fixtures[name]['source']and sha(source.read_bytes())==row['source_sha256']
  assert sha(binary.read_bytes())==row['cubin_sha256']and listing.is_file()
  for artifact in(source,binary,listing,source.with_suffix('.tool.log')):files[str(artifact.relative_to(HERE))]=sha(artifact.read_bytes())
 result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),artifact_hashes=files,script_sha256=sha(Path(__file__).read_bytes()),scope='Thirty-six legal typed original PTX controls accepted by actual CUDA12.8 ptxas; raw source/tool/cubin/listing custody, no execution')
 (HERE/'oracle-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(fixtures),'typed sources;',len(files),'raw artifacts')
