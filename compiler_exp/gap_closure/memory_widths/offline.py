"""Retain exact byte-output and poisoned-register gates for every memory fixture."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cases import GUARD, N, THREADS, cases
from openassembler.build import generate
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.parse import parse_module
from tests.backend_harness import listing


def evaluate(case, level):
    kernel = parse_module(case['source']).kernels['probe']
    code, labels, words, _, _ = generate(kernel, 'sm_80', level)
    program = load_listing(listing(code, labels))
    actual = []
    for poison in (None, 0xdeadbeef):
        memory = Memory()
        memory.write_bytes(0x100000000, GUARD + case['input'] + GUARD)
        memory.write_bytes(0x200000000, GUARD + bytes([0x3c]) * len(case['input']) + GUARD)
        Machine(program, Launch(block=(THREADS, 1, 1), poison=poison,
                                params=struct.pack('<QQI', 0x200000100, 0x100000100, N)), memory).run()
        got = bytes(memory.read(0x200000000 + i, 1) for i in range(len(case['input']) + 512))
        expected = GUARD + case['expected'] + GUARD
        unchanged = bytes(memory.read(0x100000000 + i, 1) for i in range(len(case['input']) + 512)) == GUARD + case['input'] + GUARD
        actual.append({'poison': poison, 'output_sha256': hashlib.sha256(got).hexdigest(), 'inputs_unchanged': unchanged,
                       'mismatches': [i for i, (g, e) in enumerate(zip(got, expected)) if g != e]})
    return {'status': 'passed' if all(not r['mismatches'] and r['inputs_unchanged'] for r in actual) else 'failed',
            'native_words_sha256': hashlib.sha256(b''.join(w.to_bytes(16, 'little') for w in words)).hexdigest(),
            'results': actual}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    results = []
    for case in cases():
        for level in ('safe', 'mined', 'sched', 'opt'):
            row = {'name': case['name'], 'level': level, 'contract': case['contract'],
                   'source': case['source'], 'source_sha256': hashlib.sha256(case['source'].encode()).hexdigest()}
            try:
                row.update(evaluate(case, level))
            except Exception as exc:
                row.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc))
            results.append(row)
    files = ['openassembler/backend/ra.py', 'openassembler/backend/contracts.py',
             'compiler_exp/gap_closure/memory_widths/cases.py', 'compiler_exp/gap_closure/memory_widths/offline.py']
    report = {'schema': 1, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'scope': 'Offline byte-exact independent references, encoding and normal/poisoned registers; no hardware or performance claim',
              'source_hashes': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files},
              'results': results, 'passed': all(r['status'] == 'passed' for r in results)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output, {s: sum(r['status'] == s for r in results) for s in ('passed', 'failed', 'rejected')})
