"""Verify an isolated installed wheel regenerates every audited model kernel."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run():
    validation = json.loads((HERE/'validation.json').read_text())
    report = json.loads((HERE/validation['report']).read_text())
    wheel, = (HERE/'wheel').glob('*.whl')
    requests = []
    with zipfile.ZipFile(HERE/report['archive']['path']) as archive:
        for name in sorted(name for name in archive.namelist() if name.endswith('/native-ledger.json')):
            prefix = name.removesuffix('native-ledger.json')
            requests.append(dict(prefix=prefix, plan=json.loads(archive.read(prefix+'plan.json')),
                                 source=archive.read(prefix+'sources/native.ptx').decode(),
                                 native=[row for row in validation['native_comparisons'] if row['bundle']==prefix]))
    with zipfile.ZipFile(wheel) as package:
        sources = {}
        for name in package.namelist():
            if name.startswith('openassembler/') and name.endswith(('.py','.json')):
                assert package.read(name) == (ROOT/name).read_bytes(), name
                sources[name] = sha(package.read(name))
        with tempfile.TemporaryDirectory(prefix='oa-conv2d-wheel-') as temporary:
            target = Path(temporary)
            package.extractall(target)
            (target/'requests.json').write_text(json.dumps(requests))
            code = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from openassembler.modelfront.kernels import prepare
from openassembler.ptx.parse import parse_module
from openassembler import build
count=0
for request in json.loads((root/'requests.json').read_text()):
 _,_,kernels=prepare(request['plan'])
 source='.version 8.7\\n.target sm_80\\n.address_size 64\\n'+'\\n'.join(kernels[name].ptx for name in sorted(kernels))
 assert source==request['source']
 module=parse_module(source)
 for row in request['native']:
  _,_,words,_,_=build.generate(module.kernels[row['entry']],'sm_80',row['level'])
  digest=hashlib.sha256(b''.join(word.to_bytes(16,'little') for word in words)).hexdigest()
  assert digest==row['words_sha256']
  count+=1
print(count)
'''
            result = subprocess.run([sys.executable,'-I','-c',code,str(target)], cwd=target,
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert int(result.stdout.strip()) == 142
    proof = dict(schema=1, passed=True, wheel=wheel.name, wheel_sha256=sha(wheel.read_bytes()),
                 installed_sources=sources, native_comparisons=142, bundles=11,
                 validation_sha256=sha((HERE/'validation.json').read_bytes()), script_sha256=sha(Path(__file__).read_bytes()))
    (HERE/'wheel-audit.json').write_text(json.dumps(proof, indent=2)+'\n')
    print('Passed isolated wheel identities and 142 exact native outputs across 11 model bundles')

if __name__ == '__main__':
    run()
