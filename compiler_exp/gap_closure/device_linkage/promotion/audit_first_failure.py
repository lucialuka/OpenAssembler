"""Preserve the first partial failed metadata-control gate without relabeling it."""
import hashlib,json,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import cases,values,expected

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'candidate-a100-20261123T201424Z.json';report=json.loads(path.read_text());assert not report['passed']and len(report['results'])==48
 assert report['error']==dict(type='ValueError',message='local frame metadata record pair disagrees')
 assert report['harness_sha256']==sha((HERE/'first-run-modal.py').read_bytes())
 archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256'];case=next(c for c in cases()if c['name']=='direct');count=0;guard=b'\xa5'*256
 with zipfile.ZipFile(archive_path)as archive:
  for row in report['results']:
   assert row['name']=='direct'and row['passed'];prefix=f'direct/{row["compiler"]}/{row["n"]}/{row["seed"]}/';data=values(row['seed'],384)
   before=archive.read(prefix+'before-input.bin');unchanged=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin')
   assert before==guard+struct.pack('<384Q',*data)+guard and unchanged==before and actual==guard+expected(data,case,row['n'])+guard
   assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256'];count+=len(actual)+len(unchanged)
 result=dict(schema=1,passed=False,audit_completed=True,partial_launches=48,bytes_checked=count,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],script_sha256=sha(Path(__file__).read_bytes()),scope='Original failed metadata-control gate stays failed; complete earlier direct-case outputs passed, later original-linked callee-stack layout violated an OA-only harness assumption')
 (HERE/'first-failure-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Retained failed gate;',count,'partial bytes checked')
