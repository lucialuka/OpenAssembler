"""Installed exact ABI sources and regeneration to every retained native output."""
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.container.cubin import text_words,regcount


def sha(data):return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    wheel, = (HERE/'wheel').glob('*.whl')
    report_path = HERE/'candidate-a100-20261125T011357Z.json'
    report = json.loads(report_path.read_text())
    assert report['passed']
    archive_path = HERE/report['archive']['path']
    assert sha(archive_path.read_bytes()) == report['archive']['sha256']
    requests, sources = [],{}
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read('original.ptx').decode()
        for level in ('safe','mined','sched','opt'):
            binary = archive.read(level+'/oa.cubin')
            manifest = json.loads(archive.read(level+'/manifest.json'))
            for kernel in manifest['entries'][0]['kernels']:
                name = kernel['name']
                requests.append(dict(entry=name,level=level,source=text,registers=regcount(binary,name),
                                     words_sha256=sha(b''.join(w.to_bytes(16,'little') for w in text_words(binary,name)))))
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if name.startswith(('openassembler/','specs/sm_80/')) and name.endswith(('.py','.json')):
                data = package.read(name)
                assert data == (ROOT/name).read_bytes()
                sources[name] = sha(data)
        for name in json.loads((HERE/'inputs.json').read_text()).values():assert sources[name] == report['source_hashes'][name]
        with tempfile.TemporaryDirectory(prefix='oa-narrow-wheel-') as temporary:
            root = Path(temporary)
            package.extractall(root)
            (root/'requests.json').write_text(json.dumps(requests))
            code = '''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from openassembler import build
from openassembler.ptx.parse import parse_module
assert Path(build.__file__).resolve().is_relative_to(Path.cwd())
rows=[]
for row in json.loads(Path('requests.json').read_text()):
 resources={};_,_,words,_,regs=build.generate(parse_module(row['source']).kernels[row['entry']],'sm_80',row['level'],resource_info=resources)
 digest=hashlib.sha256(b''.join(word.to_bytes(16,'little')for word in words)).hexdigest()
 assert digest==row['words_sha256']and regs==row['registers']and resources['local_bytes']==0
 rows.append(dict(entry=row['entry'],level=row['level'],registers=regs,words_sha256=digest,passed=True))
print(json.dumps(rows))
'''
            result = subprocess.run([sys.executable,'-I','-c',code],cwd=root,text=True,capture_output=True)
            assert result.returncode == 0,(result.stdout,result.stderr)
            rows = json.loads(result.stdout)
            assert len(rows) == 240
    proof = dict(schema=1,passed=True,wheel=str(wheel.relative_to(HERE)),wheel_sha256=sha(wheel.read_bytes()),
                 report_sha256=sha(report_path.read_bytes()),source_hashes=sources,native_comparisons=rows,
                 script_sha256=sha(Path(__file__).read_bytes()),
                 scope='Exact installed source/data identities including eight coordinated ABI modules; all 240 original native word/register/frame outputs regenerated in isolated installed-wheel process. No GPU rerun')
    (HERE/'wheel-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Passed',len(sources),'installed identities and 240 exact native regenerations')
