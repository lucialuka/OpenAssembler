"""Isolated installed-wheel replay of archived direct-call native binaries."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.container.cubin import text_words,regcount

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    wheel,=list((HERE/'wheel').glob('*.whl'));path=HERE/'candidate-a100-20261123T181957Z.json';report=json.loads(path.read_text());assert report['passed']
    archive=HERE/report['archive']['path'];assert sha(archive.read_bytes())==report['archive']['sha256']
    sources={};checks=[]
    with zipfile.ZipFile(wheel)as package:
        for name in package.namelist():
            if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
                data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
        for name in('openassembler/ptx/parse.py','openassembler/ptx/inlining.py','openassembler/build.py','openassembler/native_ownership.py'):assert sources[name]==report['source_hashes'][name]
        with tempfile.TemporaryDirectory(prefix='oa-functions-wheel-')as temporary:
            root=Path(temporary);package.extractall(root);requests=[]
            with zipfile.ZipFile(archive)as retained:
                names=sorted({r['name']for r in report['results']})
                for name in names:
                    source=retained.read(name+'/source.ptx').decode()
                    for level in('safe','mined','sched','opt'):
                        binary=retained.read(name+'/'+level+'/oa.cubin');words=text_words(binary,'probe')
                        requests.append(dict(name=name,source=source,level=level,registers=regcount(binary,'probe'),words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words))))
            (root/'requests.json').write_text(json.dumps(requests))
            code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 module=parse_module(row['source']);resources={}
 _,_,words,_,registers=build.generate(module.kernels['probe'],'sm_80',row['level'],resource_info=resources)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']
 assert resources['local_bytes']==(64 if row['name']=='local'else 0)
 rows.append(dict(name=row['name'],level=row['level'],passed=True,words_sha256=digest,registers=registers))
print(json.dumps(rows))
'''
            run=subprocess.run([sys.executable,'-I','-c',code],cwd=root,capture_output=True,text=True);assert run.returncode==0,(run.stdout,run.stderr)
            checks=json.loads(run.stdout);assert len(checks)==60
    result=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),report_sha256=sha(path.read_bytes()),source_hashes=sources,native_comparisons=checks,script_sha256=sha(Path(__file__).read_bytes()),scope='Exact installed source/data and sixty hardware-native word/register/resource matches in isolated wheel subprocess; no hardware rerun or timing')
    (HERE/'wheel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(sources),'installed source/data identities;',len(checks),'native comparisons')
