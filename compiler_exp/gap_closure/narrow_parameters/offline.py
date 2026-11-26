"""Independent complete-buffer narrow parameter matrix at four production levels."""
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install()
from fixtures import module, CASES, fields, inputs, pack, expected, initial
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp, Launch as PLaunch
from openassembler.emu.machine import Memory, Machine, Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    parsed = parse_module(module())
    rows, executions = [], 0
    guard = b'\xa5'*64
    out, lanes = 0x100000000, 64
    paths = [HERE/('candidate-'+suffix+'.py') for suffix in json.loads((HERE/'inputs.json').read_text())]
    paths += [HERE/name for name in ('candidate.py', 'fixtures.py', 'offline.py', 'inputs.json')]
    frozen = {p.name: sha(p) for p in paths}
    for case in CASES:
        for level in ('safe', 'mined', 'sched', 'opt'):
            code, labels, words, _, regs = build.generate(parsed.kernels[case['name']], 'sm_80', level)
            for seed in (31917, 31918):
                for sample in range(6):
                    for n in (0, 1, 37, 64):
                        values = inputs(case, seed, sample, out, n)
                        wanted = guard+expected(case, values, lanes)+guard
                        for engine in ('ptx', 'normal', 'poisoned'):
                            memory = Memory()
                            memory.write_bytes(out-64, guard+initial(lanes)+guard)
                            if engine == 'ptx':
                                Interp(parsed, case['name'], PLaunch(block=(64, 1, 1), params=values), memory).run()
                            else:
                                Machine(load_listing(listing(code, labels)), Launch(block=(64, 1, 1), params=pack(case, values), poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
                            actual = bytes(memory.read(out-64+i, 1) for i in range(len(wanted)))
                            assert actual == wanted, (case, level, seed, sample, n, engine)
                            executions += 1
            rows.append(dict(entry=case['name'], level=level, registers=regs, passed=True))
    assert frozen == {p.name: sha(p) for p in paths}
    report = dict(schema=1, passed=True, results=rows, executions=executions, source_hashes=frozen,
                  scope='60 original narrow parameter entries: every byte position/natural halfword position, b/u/s8/16, adjacent parameters, guarded/loop/aliased snapshots, partial larger-slot offsets and full output/guard/tails independently checked at four PTX/normal/poisoned levels, two seeds/six exact edge samples/four lengths; no GPU or performance claim')
    (HERE/'candidate-offline.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Passed', executions, 'complete independent narrow-parameter executions;', len(rows), 'native rows')
