"""Check exact promoted data and hardware-native words in an isolated wheel."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    wheel,=list((HERE/'wheel').glob('*.whl'))
    report_path=HERE/'candidate-a100-20261123T162930Z.json';report=json.loads(report_path.read_text());assert report['passed']
    archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    sources={}
    with zipfile.ZipFile(wheel)as package:
        for name in package.namelist():
            if name.startswith(('openassembler/','specs/sm_80/'))and name.endswith(('.py','.json')):
                data=package.read(name);assert data==(ROOT/name).read_bytes();sources[name]=sha(data)
        for name in('openassembler/backend/ra.py','openassembler/backend/compile.py','openassembler/pyfront/compile.py'):
            assert sources[name]==report['source_hashes'][name]
        with tempfile.TemporaryDirectory(prefix='oa-shared-wheel-')as temporary:
            target=Path(temporary);package.extractall(target)
            with zipfile.ZipFile(archive_path)as archive:
                fixtures=json.loads((ROOT/'tests/data/pyfront_shared_wide.json').read_text());expected=[]
                sys.path.insert(0,str(ROOT))
                from openassembler.container.cubin import text_words,regcount
                for level in('safe','mined','sched','opt'):
                    binary=archive.read(level+'/oa.cubin')
                    for case in fixtures:
                        words=text_words(binary,case['name'])
                        expected.append(dict(case=case,level=level,words_sha256=sha(b''.join(w.to_bytes(16,'little')for w in words)),registers=regcount(binary,case['name'])))
            (target/'expected.json').write_text(json.dumps(expected))
            code='''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
checks=[]
for row in json.loads(Path('expected.json').read_text()):
 case=row['case'];resources={}
 path=Path('fixture.py');path.write_text(case['source']);ptx,names=compile_file(path)
 _,_,words,_,registers=build.generate(parse_module(ptx).kernels[case['name']],'sm_80',row['level'],resource_info=resources)
 digest=hashlib.sha256(b''.join(w.to_bytes(16,'little')for w in words)).hexdigest()
 assert digest==row['words_sha256']and registers==row['registers']
 assert resources['local_bytes']==0
 checks.append(dict(name=case['name'],level=row['level'],passed=True,words_sha256=digest,registers=registers))
print(json.dumps(checks))
'''
            run=subprocess.run([sys.executable,'-I','-c',code],cwd=target,text=True,capture_output=True)
            assert run.returncode==0,(run.stdout,run.stderr)
            checks=json.loads(run.stdout);assert len(checks)==80
    result=dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),
                report_sha256=sha(report_path.read_bytes()),source_hashes=sources,native_comparisons=checks,
                script_sha256=sha(Path(__file__).read_bytes()),scope='Built wheel exact production source/data and 80 hardware-native word/register matches in isolated -I subprocess; no hardware rerun or timing')
    (HERE/'wheel-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed isolated wheel:',len(sources),'source/data identities;',len(checks),'native comparisons')
