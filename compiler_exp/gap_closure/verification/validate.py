"""Replay a retained A100 corpus through the current emulator and exact reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openassembler.emu.probe import run_probe
from openassembler.emu import ops
from openassembler.emu.machine import Launch, Machine
from openassembler.emu.program import load_listing, parse_insn


def validate(report_path, output):
    report = json.loads(report_path.read_text())
    archive_path = report_path.with_name(report['artifact_archive']['path'])
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_hash != report['artifact_archive']['sha256']:
        raise ValueError('hardware archive identity changed')
    with zipfile.ZipFile(archive_path) as archive:
        inputs = json.loads(archive.read('inputs.json'))
        expected = json.loads(archive.read('expected.json'))
        frozen_sass = archive.read('frozen.sass.txt').decode()
        diagnostic = json.loads(archive.read('frozen_fchk.outputs.json'))
    actual = []
    for offset in range(0, len(inputs), 32):
        actual.extend(word[0] for word in run_probe(frozen_sass, inputs[offset:offset + 32], 0, 0))
    mismatches = [{'index': i, 'input': [hex(x) for x in inputs[i]],
                   'actual': hex(a), 'expected': hex(e)}
                  for i, (a, e) in enumerate(zip(actual, expected)) if a != e]
    machine = Machine(load_listing(''), Launch(block=(1, 1, 1)))
    warp = machine.warps[0]
    lane = warp.lanes[0]
    instruction = parse_insn(0, 'FCHK P0, R4, R5', 0)
    missed_hardware_slow, extra_emulated_slow = [], []
    for i, ((a, b, _, _), (_, predicate)) in enumerate(zip(inputs, diagnostic)):
        lane.r[4], lane.r[5] = a, b
        ops.ALU['FCHK'](machine, lane, warp, instruction)
        hardware = predicate == 0
        if hardware and not lane.p[0]:
            missed_hardware_slow.append(i)
        elif not hardware and lane.p[0]:
            extra_emulated_slow.append(i)
    raw_path = output.with_suffix('.outputs.json')
    raw_path.write_text(json.dumps(actual) + '\n')
    result = {'schema': 1, 'scope': 'Current offline emulator replay of independently gated A100 corpus; no new hardware run',
              'python': sys.version, 'pairs': len(inputs), 'mismatches': mismatches,
              'passed': not mismatches, 'hardware_report': report_path.name,
              'archive_sha256': archive_hash,
              'fchk_missed_hardware_slow': missed_hardware_slow,
              'fchk_extra_emulated_slow': extra_emulated_slow,
              'fchk_note': 'Operational partial model; equality to every hardware predicate is not claimed',
              'outputs': {'path': raw_path.name, 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()},
              'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['openassembler/emu/ops.py', 'openassembler/emu/fp.py',
                                             'compiler_exp/gap_closure/verification/validate.py']}}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output, 'pairs', len(inputs), 'mismatches', len(mismatches),
          'missed_slow', len(missed_hardware_slow), 'extra_slow', len(extra_emulated_slow))
    return result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hardware_report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sys.exit(0 if validate(args.hardware_report, args.output) else 1)
