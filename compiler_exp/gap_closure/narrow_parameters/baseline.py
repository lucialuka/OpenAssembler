"""Reproduce the original complete corpus rejection using frozen pre-entry ABI."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install('baseline')
from fixtures import HEADER, CASES, source
from openassembler.ptx.parse import parse_module

if __name__ == '__main__':
    rows = []
    for case in CASES:
        try:
            parse_module(HEADER+source(case))
        except SyntaxError as error:
            assert 'scalar 32/64-bit parameters required' in str(error)
            rows.append(dict(entry=case['name'],rejected=True,diagnostic=str(error)))
        else:
            raise AssertionError(case)
    paths = [HERE/('baseline-'+suffix+'.py') for suffix in json.loads((HERE/'inputs.json').read_text())]
    paths += [Path(__file__), HERE/'fixtures.py', HERE/'inputs.json']
    data = dict(schema=1,passed=True,results=rows,
                source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Sixty original rejected narrow-parameter entries with exact frozen modules and unchanged fixture generation; no numerical/GPU claim')
    (HERE/'baseline.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Reproduced all 60 original parser ABI rejections')
