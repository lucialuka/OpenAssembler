"""Retain exact G01 replay and instruction trace without asserting hardware truth."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import diff_emu
from probe import COUNTEREXAMPLE, exact_division
from openassembler.emu import ops
from openassembler.emu.machine import Launch, Machine, Memory
from openassembler.emu.probe import IN_BASE, OUT_BASE, probe_params
from openassembler.emu.program import load_listing


def run():
    source = ROOT / 'corpus/sm_80/ptx/p_div_rn_f32.ptx'
    listing = ROOT / 'corpus/sm_80/sass/p_div_rn_f32.sass'
    memory = Memory()
    memory.write_bytes(IN_BASE, struct.pack('<4I', *COUNTEREXAMPLE))
    trace = []
    original = ops.execute
    def execute(machine, warp, group, insn):
        original(machine, warp, group, insn)
        if insn.op in ('FCHK', 'MUFU', 'FFMA'):
            trace.append({'address': hex(insn.addr), 'instruction': insn.text,
                          'registers_0_to_15': [hex(x) for x in group[0].r[:16]],
                          'predicates': list(group[0].p)})
    ops.execute = execute
    try:
        Machine(load_listing(listing.read_text()), Launch(block=(1, 1, 1), params=probe_params(0, 0)), memory).run()
    finally:
        ops.execute = original
    actual = memory.read(OUT_BASE, 4)
    independent = exact_division(*COUNTEREXAMPLE[:2])
    interpreted = diff_emu.run_ptx(source.read_text(), 'p_div_rn_f32', [COUNTEREXAMPLE], 0, 0)[0][0][0]
    files = [source, listing, ROOT / 'openassembler/emu/ops.py', ROOT / 'openassembler/emu/fp.py',
             Path(__file__), Path(__file__).with_name('probe.py')]
    return {'schema': 1, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Offline emulator diagnostic, independent integer-ratio RNE reference; no hardware or timing claim',
            'python': sys.version, 'inputs': [hex(x) for x in COUNTEREXAMPLE],
            'emulator': hex(actual), 'independent_reference': hex(independent),
            'ptx_interpreter': hex(interpreted), 'matches': actual == independent == interpreted,
            'trace': trace, 'source_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in files}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output, 'matches', result['matches'])
