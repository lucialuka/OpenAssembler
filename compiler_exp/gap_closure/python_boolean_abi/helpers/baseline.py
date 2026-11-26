"""Reproduce every original rejection with exact frozen pre-helper modules."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install('baseline')
from fixtures import configurations, write_tree, NAMES
from openassembler.pyfront import compile_kernel, find_kernel, NotSupported

if __name__ == '__main__':
    rows = []
    with tempfile.TemporaryDirectory(prefix='oa-boolean-helper-baseline-') as temporary:
        for case in configurations():
            source = write_tree(Path(temporary)/case['name'], case)
            for name in NAMES:
                try:
                    compile_kernel(find_kernel(source, name))
                except NotSupported as error:
                    assert 'unsupported type annotation' in str(error) and error.path and error.lineno
                    rows.append(dict(configuration=case['name'], entry=name, rejected=True, diagnostic=str(error)))
                else:
                    raise AssertionError(name)
    paths = [HERE/n for n in ('baseline-pyfront.py', 'baseline-helpers.py', 'baseline-modules.py', 'fixtures.py', 'baseline.py')]
    report = dict(schema=1, passed=True, results=rows,
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope='Sixty original logical Boolean helper annotation rejections in three unexecuted source graphs; exact frozen pre-helper compiler, no hardware claim')
    (HERE/'baseline.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Passed sixty frozen Boolean-helper annotation rejections')
