"""Installed sources, typed Python regeneration and 156 exact GPU-native outputs."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler.container.cubin import text_words,regcount
from python_fixtures import cases
from square_fixtures import cases as squares

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 wheel,=list((HERE/'wheel').glob('*.whl'));proof_path=HERE/'promotion-audit.json';proof=json.loads(proof_path.read_text());assert proof['passed']and proof['audited_backend']=='production'
 fixtures={c['name']:c for c in list(cases())+list(squares())};sources={};requests=[]
 for row in proof['reports']:
  path=ROOT/row['path'];report=json.loads(path.read_text());assert report['passed']and sha(path.read_bytes())==row['sha256']
  archive=path.parent/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256']
  with zipfile.ZipFile(archive)as retained:
   for name in sorted({r['name']for r in report['results']}):
    case=fixtures[name]
    for level in('safe','mined','sched','opt'):
     source=retained.read(name+'/source.ptx').decode();binary=retained.read(name+'/oa-'+level+'/native.cubin');words=text_words(binary,case['entry'])
     requests.append(dict(name=name,entry=case['entry'],level=level,source=source,python_source=case.get('python_source'),registers=regcount(binary,case['entry']),words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
 assert len(requests)==156
 with zipfile.ZipFile(wheel)as package:
  for name in package.namelist():
   if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
    data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
  for name in('backend/ra','backend/contracts','pyfront/compile'):assert sources['openassembler/'+name+'.py']==report['source_hashes']['openassembler/'+name+'.py']
  with tempfile.TemporaryDirectory(prefix='oa-product-wheel-')as temporary:
   target=Path(temporary);package.extractall(target);(target/'requests.json').write_text(json.dumps(requests))
   code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 if row['python_source']:
  original=Path(row['name']+'.py');original.write_text(row['python_source']);ptx,names=compile_file(original)
  assert names==[row['entry']]and ptx==row['source']
 resource={};module=parse_module(row['source'])
 _,_,words,_,registers=build.generate(module.kernels[row['entry']],'sm_80',row['level'],resource_info=resource)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']and resource['local_bytes']==0
 rows.append(dict(name=row['name'],entry=row['entry'],level=row['level'],passed=True,words_sha256=digest,registers=registers,python_regenerated=bool(row['python_source'])))
print(json.dumps(rows))
'''
   run=subprocess.run([sys.executable,'-I','-c',code],cwd=target,text=True,capture_output=True);assert run.returncode==0,(run.stdout,run.stderr)
   rows=json.loads(run.stdout);assert len(rows)==156 and sum(r['python_regenerated']for r in rows)==80
 result=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),promotion_audit_sha256=sha(proof_path.read_bytes()),source_hashes=sources,native_comparisons=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Exact installed source/data, 80 typed Python source regenerations and 156 full hardware-native word/register/resource matches in isolated wheel subprocess; no hardware rerun')
 (HERE/'wheel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(sources),'installed source/data identities;',len(rows),'native comparisons; 80 Python PTX regenerations')
