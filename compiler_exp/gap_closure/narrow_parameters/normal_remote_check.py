"""Actual public CLI/Modal build of the original narrow-parameter source."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(HERE)]
from fixtures import module,NAMES

if __name__ == '__main__':
    source = HERE/'normal-remote-input.ptx'
    source.write_text(module())
    output = HERE/'normal-remote-build'
    args = [sys.executable,'-m','openassembler','compile',str(source),'--out-dir',str(output),'--build-backend','modal','--no-sass']
    result = subprocess.run(args,cwd=ROOT)
    assert result.returncode == 0
    from openassembler.native_ownership import validate
    manifest = json.loads((output/'manifest.json').read_text())
    entry, = manifest['entries']
    binary = (output/entry['cubin']).read_bytes()
    validate(entry,binary)
    assert [kernel['name'] for kernel in entry['kernels']] == list(NAMES)
    assert entry['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    proof = dict(schema=1,passed=True,args=args,cubin_sha256=hashlib.sha256(binary).hexdigest(),
                 manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 parameters={kernel['name']:kernel['params'] for kernel in entry['kernels']},
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Original complete narrow-parameter source through actual public CLI/Modal compilation/publication, exact source/parameter signatures and native ownership; numerical A100 launches have separate independent proof')
    (HERE/'normal-remote-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Passed public CLI/Modal original narrow scalar build and ownership')
