"""PTX interpreter vs independent integer references on the identical fixtures."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cases import GUARD, N, THREADS, cases
from openassembler.emu.machine import Memory
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Launch, run_kernel


if __name__ == '__main__':
    results = []
    for case in cases():
        memory = Memory()
        memory.write_bytes(0x100000000, GUARD + case['input'] + GUARD)
        memory.write_bytes(0x200000000, GUARD + b'\x3c' * len(case['expected']) + GUARD)
        module = parse_module(case['source'])
        try:
            run_kernel(module, 'probe', Launch(block=(THREADS, 1, 1),
                       params={'input': 0x100000100, 'output': 0x200000100, 'n': N}), memory)
            output = bytes(memory.read(0x200000000 + i, 1) for i in range(len(case['expected']) + 512))
            expected = GUARD + case['expected'] + GUARD
            results.append({'name': case['name'], 'passed': output == expected,
                            'mismatches': [i for i, (a, b) in enumerate(zip(output, expected)) if a != b][:20]})
        except Exception as exc:
            results.append({'name': case['name'], 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)})
    report = {'schema': 1, 'passed': all(r['passed'] for r in results),
              'scope': 'PTX interpreter vs independent Python bit math; no SASS/toolkit/GPU proof', 'results': results}
    Path(__file__).with_name('ptx-reference-20261121.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PTX',len(results), 'passed', report['passed'], [r for r in results if not r['passed']][:3])
