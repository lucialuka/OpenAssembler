"""Replay the original ABI rejections without rewriting historical toolkit artifacts."""
import builtins
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for name, suffix in (('compile', 'pyfront'), ('modules', 'modules')):
    module = importlib.import_module('openassembler.pyfront.'+name)
    path = HERE/('baseline-'+suffix+'.py')
    exec(builtins.compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
package = importlib.import_module('openassembler.pyfront')
module = importlib.import_module('openassembler.pyfront.compile')
for name in ('NotSupported', 'compile_kernel', 'find_kernel'):
    setattr(package, name, getattr(module, name))
from openassembler.pyfront import compile_kernel, find_kernel, NotSupported

if __name__ == '__main__':
    rows = []
    for name in ('scalar_flag', 'boolean_arrays', 'boolean_helper'):
        try:
            compile_kernel(find_kernel(HERE/'source.py', name))
        except NotSupported as error:
            assert error.path and error.lineno and 'annotation' in str(error)
            rows.append(dict(entry=name, rejected=True, diagnostic=str(error)))
        else:
            raise AssertionError(name)
    report = dict(schema=1, passed=True, results=rows,
                  original_toolkit_record_sha256=hashlib.sha256((HERE/'baseline-toolkit.json').read_bytes()).hexdigest(),
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [HERE/'source.py', HERE/'baseline-pyfront.py', HERE/'baseline-modules.py', Path(__file__)]},
                  scope='Three exact original annotation rejections using frozen pre-ABI frontend/source modules; original toolkit artifacts are unchanged. No GPU or current helper rejection claim')
    (HERE/'baseline-replay.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Replayed three original frozen Boolean ABI rejections; historical toolkit artifacts unchanged')
