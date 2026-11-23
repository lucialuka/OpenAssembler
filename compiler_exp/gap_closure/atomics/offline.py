"""Atomics offline gate against independent linearization invariants and PTX."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cases import GUARD, GRID, N, THREADS, cases, check, inputs
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch, run_kernel
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def execute(case, mode, level):
    module = parse_module(case['source'])
    code, labels, words, _, _ = generate(module.kernels['probe'], 'sm_80', level)
    before = inputs(case, mode)
    addresses = {name: 0x100000000 + i * 0x100000 for i, name in enumerate(before)}
    params = {**addresses, 'n': N, 'mode': mode}
    program = load_listing(listing(code, labels))
    results = []
    for engine in ('sass', 'poisoned', 'ptx'):
        memory = Memory()
        for name, data in before.items():
            memory.write_bytes(addresses[name] - len(GUARD), GUARD + data + GUARD)
        for cta in range(GRID[0]):
            if engine == 'ptx':
                run_kernel(module, 'probe', PTXLaunch(grid=GRID, block=(THREADS, 1, 1), ctaid=(cta, 0, 0), params=params), memory)
            else:
                packed = struct.pack('<QQQII', params['output'], params['state'], params['input'], N, mode)
                Machine(program, Launch(grid=GRID, block=(THREADS, 1, 1), ctaid=(cta, 0, 0), params=packed,
                        poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
        actual = {name: bytes(memory.read(addresses[name] + i, 1) for i in range(len(data))) for name, data in before.items()}
        guard_ok = all(bytes(memory.read(addresses[name] + offset + i, 1) for i in range(len(GUARD))) == GUARD
                       for name, data in before.items() for offset in (-len(GUARD), len(data)))
        result = check(case, mode, before, actual)
        result.update(engine=engine, guards=guard_ok,
                      output_hashes={name: hashlib.sha256(data).hexdigest() for name, data in actual.items()})
        results.append(result)
    return {'status': 'passed' if all(r['passed'] and r['guards'] for r in results) else 'failed',
            'results': results, 'words_sha256': hashlib.sha256(b''.join(w.to_bytes(16, 'little') for w in words)).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('output', type=Path);parser.add_argument('--candidate', action='store_true');args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    rows = []
    for case in cases():
        for mode in (0, 1):
            for level in ('safe', 'mined', 'sched', 'opt'):
                row = {'name': case['name'], 'mode': mode, 'level': level, 'source': case['source'], 'contract': case['contract']}
                try: row.update(execute(case, mode, level))
                except Exception as exc: row.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc))
                rows.append(row)
    files = ['openassembler/backend/ra.py','openassembler/backend/contracts.py','compiler_exp/gap_closure/atomics/cases.py','compiler_exp/gap_closure/atomics/candidate.py']
    report = {'schema': 1, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'scope': 'Sequential CTA SASS/PTX model and independent atomic linearization checks; no inter-CTA concurrency proof',
              'results': rows, 'passed': all(r['status'] == 'passed' for r in rows),
              'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output,{s:sum(r['status']==s for r in rows) for s in ('passed','failed','rejected')})
