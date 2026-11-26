"""Audit complete baseline/current cubins and toolkit-dependent regression tests."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

from exp.model_perf.allocator_candidate import original_intervals
from exp.model_perf.campaign import environment
from openassembler import build
from openassembler.backend import ra


def run():
    root = Path('results')
    files = ('pressure_fixtures.json', 'spill_group_fixtures.json', 'spill_fixtures.json')
    cases = {case['name']: case for file in files
             for case in json.loads((Path('tests/data') / file).read_text())}
    records = []
    for name in ('scalar96', 'pred24', 'group4_auto', 'pressure_tight'):
        source = root / f'{name}.ptx'
        source.write_text(cases[name]['source'])
        binaries = []
        samples = {}
        for label, scan in [('baseline', original_intervals), ('production', ra._intervals)]:
            target = root / name / label
            with patch.object(ra, '_intervals', scan):
                started = perf_counter()
                status = build.compile_paths(source, target, build.Options(no_sass=True))
                samples[label] = perf_counter() - started
            assert status.rc == 0, (name, label)
            manifest = json.loads((target / 'manifest.json').read_text())
            entry = manifest['entries'][0]
            binary = (target / entry['cubin']).read_bytes()
            binaries.append(binary)
        assert binaries[0] == binaries[1], name
        records.append({'case': name, 'source_sha256': sha256(source.read_bytes()).hexdigest(),
                        'cubin_sha256': sha256(binaries[0]).hexdigest(), 'build_seconds': samples,
                        'scope': 'one complete build sample per arm, including toolkit invocation'})
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                            'tests/test_backend_spilling.py', 'tests/test_backend_spill_groups.py',
                            'tests/test_backend_pressure.py', 'tests/test_model_tensor.py',
                            'tests/test_model_tensor_schedule.py', 'tests/test_model_batched_tensor.py',
                            'tests/test_model_training_kernels.py', 'tests/test_model_runtime.py',
                            'tests/test_model_cache_tools.py'], capture_output=True, text=True, timeout=600)
    (root / 'tests.log').write_text(tests.stdout + tests.stderr)
    result = {'passed': tests.returncode == 0, 'environment': environment(), 'builds': records,
              'tests': {'returncode': tests.returncode, 'output': tests.stdout + tests.stderr}}
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    assert result['passed'], tests.stdout + tests.stderr
