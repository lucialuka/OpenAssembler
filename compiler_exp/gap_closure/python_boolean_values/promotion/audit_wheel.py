"""Installed source custody and original Python graph regeneration to retained native words."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import configurations,NAMES
from openassembler.container.cubin import text_words,regcount

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
 wheel,=list((HERE/'wheel').glob('*.whl'));report_path=HERE/'candidate-a100-20261124T120418Z.json';report=json.loads(report_path.read_text());assert report['passed']
 archive=HERE/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256'];sources={};requests=[];cases={c['name']:c for c in configurations()}
 with zipfile.ZipFile(archive)as retained:
  for configuration,case in cases.items():
   for level in('safe','mined','sched','opt'):
    prefix=configuration+'/'+level+'/';source=retained.read(prefix+'source.ptx').decode();binary=retained.read(prefix+'oa.cubin')
    for name in NAMES:
     words=text_words(binary,name);requests.append(dict(configuration=configuration,case={**case,'kernels':list(NAMES)},entry=name,level=level,source=source,registers=regcount(binary,name),words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
 with zipfile.ZipFile(wheel)as package:
  for name in package.namelist():
   if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
    data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
  for name in('backend/compile','pyfront/modules','pyfront/compile'):assert sources['openassembler/'+name+'.py']==report['source_hashes']['openassembler/'+name+'.py']
  with tempfile.TemporaryDirectory(prefix='oa-helper-graph-wheel-')as temporary:
   target=Path(temporary);package.extractall(target);(target/'requests.json').write_text(json.dumps(requests))
   code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.pyfront import compile_file
from openassembler.ptx.parse import parse_module
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 case=row['case'];root=Path('original')/row['configuration']
 for name,text in case['files'].items():
  path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
 source=root/case['entry'];ptx,names=compile_file(source,source_root=root if case['explicit']else None)
 assert ptx==row['source']and names==list(row['case']['kernels'])
 resources={};_,_,words,_,registers=build.generate(parse_module(ptx).kernels[row['entry']],'sm_80',row['level'],resource_info=resources)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']and resources['local_bytes']==0
 rows.append(dict(configuration=row['configuration'],entry=row['entry'],level=row['level'],passed=True,registers=registers,words_sha256=digest))
print(json.dumps(rows))
'''
   result=subprocess.run([sys.executable,'-I','-c',code],cwd=target,text=True,capture_output=True);assert result.returncode==0,(result.stdout,result.stderr)
   rows=json.loads(result.stdout);assert len(rows)==240
 proof=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),report_sha256=sha(report_path.read_bytes()),source_hashes=sources,native_comparisons=rows,script_sha256=sha(Path(__file__).read_bytes()),scope='Exact installed source/data and all original Python trees regenerated to byte-identical PTX plus 240 complete hardware-native word/register/frame matches in isolated wheel process; no GPU rerun')
 (HERE/'wheel-audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('Passed',len(sources),'installed identities; 240 Python source/native regenerations')
