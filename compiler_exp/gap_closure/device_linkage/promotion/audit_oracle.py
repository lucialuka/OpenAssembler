"""Read-only separate PTX object/link and merged control artifact custody."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import cases
from openassembler.ptx.link import resolve

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 path=HERE/'oracle-probe.json';report=json.loads(path.read_text());assert report['script_sha256']==sha((HERE/'oracle_probe.py').read_bytes())and report['linker_sha256']==sha((ROOT/'openassembler/ptx/link.py').read_bytes())
 fixtures={c['name']:c for c in cases()};assert len(fixtures)==len(report['results'])==10;files={}
 for row in report['results']:
  name=row['name'];case=fixtures[name];assert row['original_objects_accepted']and row['original_link_accepted']and row['merged_accepted']
  folder=HERE/'oracle'/name
  for i,text in enumerate([case['root'],*case['libraries']]):assert(folder/f'unit{i}.ptx').read_text()==text
  merged,contract=resolve([('entry',case['root'])]+[(f'library{i}',text)for i,text in enumerate(case['libraries'])]);assert row['contract']==contract and(folder/'merged.ptx').read_text()==merged
  assert sha((folder/'original-linked.cubin').read_bytes())==row['original_cubin_sha256']and sha((folder/'merged.cubin').read_bytes())==row['merged_cubin_sha256']
  for artifact in sorted(folder.iterdir()):
   if artifact.is_file():files[str(artifact.relative_to(HERE))]=sha(artifact.read_bytes())
 result=dict(schema=1,passed=True,report_sha256=sha(path.read_bytes()),artifact_hashes=files,script_sha256=sha(Path(__file__).read_bytes()),scope='Ten original separate-module PTX object/link controls and deterministic merged-source controls; complete raw artifact custody, no execution')
 (HERE/'oracle-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(fixtures),'sources;',len(files),'raw artifacts')
