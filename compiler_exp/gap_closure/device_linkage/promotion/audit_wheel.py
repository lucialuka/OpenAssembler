"""Installed wheel source/data custody and current archived library-native replay."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler.container.cubin import text_words,regcount

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 wheel,=list((HERE/'wheel').glob('*.whl'));path=HERE/'candidate-a100-20261123T201545Z.json';report=json.loads(path.read_text());assert report['passed']
 archive=HERE/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256'];sources={}
 with zipfile.ZipFile(wheel)as package:
  for name in package.namelist():
   if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
    data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
  for name in('ptx/link','ptx/inlining','build','cppfront','cli','native_ownership','remote_build'):assert sources['openassembler/'+name+'.py']==report['source_hashes']['openassembler/'+name+'.py']
  with tempfile.TemporaryDirectory(prefix='oa-library-wheel-')as temporary:
   target=Path(temporary);package.extractall(target);requests=[]
   with zipfile.ZipFile(archive)as retained:
    for name in sorted({row['name']for row in report['results']}):
     for level in('safe','mined','sched','opt'):
      source=retained.read(name+'/'+level+'/merged.ptx').decode();binary=retained.read(name+'/'+level+'/oa.cubin');words=text_words(binary,'probe')
      requests.append(dict(name=name,level=level,source=source,registers=regcount(binary,'probe'),words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
   (target/'requests.json').write_text(json.dumps(requests))
   code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.link import resolve
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 resource={};module=parse_module(row['source'])
 _,_,words,_,registers=build.generate(module.kernels['probe'],'sm_80',row['level'],resource_info=resource)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']
 assert resource['local_bytes']==(32 if row['name']=='local_frame'else 0)
 rows.append(dict(name=row['name'],level=row['level'],passed=True,words_sha256=digest,registers=registers))
print(json.dumps(rows))
'''
   run=subprocess.run([sys.executable,'-I','-c',code],cwd=target,text=True,capture_output=True);assert run.returncode==0,(run.stdout,run.stderr)
   rows=json.loads(run.stdout);assert len(rows)==48
 result=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),report_sha256=sha(path.read_bytes()),source_hashes=sources,native_comparisons=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Exact installed source/data and forty-eight complete hardware-native word/register/resource matches in isolated wheel subprocess; no hardware rerun')
 (HERE/'wheel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(sources),'installed source/data identities;',len(rows),'native comparisons')
