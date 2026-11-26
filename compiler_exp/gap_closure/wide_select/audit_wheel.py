"""Installed wheel source/data identity and all retained wide-selection native bytes."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from openassembler.container.cubin import text_words,regcount
from fixtures import expanded_cases

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 wheel,=list((HERE/'wheel').glob('*.whl'));path=HERE/'candidate-a100-20261123T205126Z.json';report=json.loads(path.read_text());assert report['passed']
 archive=HERE/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256'];sources={};fixtures={c['name']:c for c in expanded_cases()}
 with zipfile.ZipFile(wheel)as package:
  for name in package.namelist():
   if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
    data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
  for name in('backend/ra','backend/contracts'):assert sources['openassembler/'+name+'.py']==report['source_hashes']['openassembler/'+name+'.py']
  with tempfile.TemporaryDirectory(prefix='oa-select-wheel-')as temporary:
   target=Path(temporary);package.extractall(target);requests=[]
   with zipfile.ZipFile(archive)as retained:
    for name,case in fixtures.items():
     for level in('safe','mined','sched','opt'):
      source=retained.read(name+'/source.ptx').decode();binary=retained.read(name+'/oa-'+level+'/native.cubin');words=text_words(binary,case['entry'])
      requests.append(dict(name=name,entry=case['entry'],level=level,source=source,registers=regcount(binary,case['entry']),words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
   (target/'requests.json').write_text(json.dumps(requests))
   code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 resource={};module=parse_module(row['source'])
 _,_,words,_,registers=build.generate(module.kernels[row['entry']],'sm_80',row['level'],resource_info=resource)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']and resource['local_bytes']==0
 rows.append(dict(name=row['name'],entry=row['entry'],level=row['level'],passed=True,words_sha256=digest,registers=registers))
print(json.dumps(rows))
'''
   run=subprocess.run([sys.executable,'-I','-c',code],cwd=target,text=True,capture_output=True);assert run.returncode==0,(run.stdout,run.stderr)
   rows=json.loads(run.stdout);assert len(rows)==148
 result=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),report_sha256=sha(path.read_bytes()),source_hashes=sources,native_comparisons=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Exact installed source/data and 148 complete hardware-native word/register/resource matches in isolated wheel subprocess; no hardware rerun')
 (HERE/'wheel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(sources),'installed source/data identities;',len(rows),'native comparisons')
