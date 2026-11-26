"""Wide select hypothesis against independent full bits and poisoned allocation."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from fixtures import cases, values, expected, SENTINEL, CHOICE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    else:
        import builtins, importlib
        for name in ('ra', 'contracts'):
            module = importlib.import_module('openassembler.backend.' + name)
            path = HERE/('baseline-' + name + '.py')
            exec(builtins.compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.ptx.interp import Interp, Launch as PL
    from openassembler.emu.machine import Machine, Memory, Launch
    from openassembler.emu.program import load_listing
    from tests.backend_harness import listing
    rows = []
    inp, out = 0x100010000, 0x100000000
    guard = b'\xa5' * 64
    for case in cases():
        try:
            module = parse_module(case['source'])
            for level in ('safe', 'mined', 'sched', 'opt'):
                code, labels, words, _, registers = build.generate(module.kernels['probe'], 'sm_80', level)
                for seed in (27391, 27392):
                    data = values(seed)
                    before = guard + struct.pack('<256Q', *data) + guard
                    for n in (0, 91, 128):
                        wanted = guard + expected(data, case, n) + guard
                        for engine in ('ptx', 'normal', 'poisoned'):
                            memory = Memory()
                            memory.write_bytes(inp - 64, before)
                            memory.write_bytes(out - 64, guard + struct.pack('<Q', SENTINEL) * 128 + guard)
                            if engine == 'ptx':
                                Interp(module, 'probe', PL(block=(128, 1, 1), params={'out': out, 'input': inp, 'n': n, 'choice': CHOICE}), memory).run()
                            else:
                                Machine(load_listing(listing(code, labels)), Launch(block=(128, 1, 1), params=struct.pack('<QQI4xQ', out, inp, n, CHOICE), poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
                            assert bytes(memory.read(out - 64 + i, 1) for i in range(len(wanted))) == wanted, (case['name'], level, n, seed, engine)
                            assert bytes(memory.read(inp - 64 + i, 1) for i in range(len(before))) == before
                rows.append(dict(name=case['name'], level=level, passed=True, executions=18))
        except Exception as error:
            rows.append(dict(name=case['name'], passed=False, error=type(error).__name__ + ': ' + str(error)))
    result = dict(schema=1, passed=all(r['passed'] for r in rows) if args.candidate else not any(r['passed'] for r in rows), results=rows,
                  scope='Independent complete-bit low64 multiply PTX/normal/poisoned SASS, full input/output guards; toolkit/hardware gates separate',
                  source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'candidate-ra.py', HERE/'candidate-contracts.py', HERE/'fixtures.py', Path(__file__))})
    (HERE/('candidate-offline.json' if args.candidate else 'baseline.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(result['passed'], len(rows), sum(r.get('executions', 0) for r in rows))
    if not result['passed']:
        print([r for r in rows if not r['passed']])
        raise SystemExit(1)
