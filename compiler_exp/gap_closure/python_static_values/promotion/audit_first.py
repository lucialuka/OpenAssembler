"""Retain source-exact first gate before bounded-power and len-shadow hardening."""
import builtins,hashlib,json,sys,types,runpy,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
if __name__=='__main__':
 path=HERE/'candidate-a100-20261124T021108Z.json';report=json.loads(path.read_text());assert report['passed']and len(report['results'])==1440
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 expected={'openassembler/pyfront/static_values.py':'first-static_values.py','compiler_exp/gap_closure/python_static_values/promotion/fixtures.py':'first-fixtures.py'}
 assert all(sha(HERE/name)==report['source_hashes'][target]for target,name in expected.items())
 # Preserve evidence custody independently; the final numerical audit uses its
 # exact final payload. The first report is never rebound to new source bytes.
 import zipfile
 archive=HERE/report['archive']['path'];assert sha(archive)==report['archive']['sha256']
 fixture_module=types.ModuleType('first_final_fixture');fixture_module.__file__=str(HERE/'first-fixtures.py');exec(builtins.compile(Path(fixture_module.__file__).read_bytes(),fixture_module.__file__,'exec'),fixture_module.__dict__)
 guard=b'\xa5'*256;count=0
 with zipfile.ZipFile(archive)as retained:
  for case in fixture_module.configurations():
   for name,text in case['files'].items():assert retained.read(case['name']+'/original/'+name)==text.encode()
  for row in report['results']:
   c,compiler,name,seed,n=(row[k]for k in('configuration','compiler','entry','seed','n'));prefix=f'{c}/{compiler}/{name}/{seed}/{n}/'
   before,after,actual=(retained.read(prefix+k+'.bin')for k in('before-a','a','output'))
   assert before==after==guard+fixture_module.raw_input(name,seed,384)+guard
   assert actual==guard+fixture_module.expected(name,seed,n,384)+guard and row['passed']
   count+=len(actual)+len(after)
 assert count==9338880
 result=dict(schema=1,passed=True,launches=1440,bytes_checked=count,report_sha256=sha(path),archive_sha256=sha(archive),retained_source_hashes={name:sha(HERE/name)for name in expected.values()},script_sha256=sha(Path(__file__)),scope='Complete independent original bytes/guards/input and exact retained pre-hardening source payload identities; no current-native rebinding, no additional execution or performance claim')
 (HERE/'first-hardware-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('First complete source/byte custody passed 1440 launches and',count,'bytes')
