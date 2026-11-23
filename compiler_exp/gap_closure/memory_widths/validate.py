"""Audit retained byte outputs and exact promoted backend identities; no GPU dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
from cases import GUARD, cases


def validate(report_path):
    report = json.loads(report_path.read_text())
    archive_path = report_path.with_name(report['artifact_archive']['path'])
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == report['artifact_archive']['sha256']
    fixtures = {case['name']: case for case in cases()}
    domains = {(name, level, compiler) for name in fixtures for level in ('safe', 'mined', 'sched', 'opt')
               for compiler in ('oa', 'ptxas')}
    assert len(report['results']) == len(domains)
    assert {(row['name'], row['level'], row['compiler']) for row in report['results']} == domains
    with zipfile.ZipFile(archive_path) as archive:
        for name, case in fixtures.items():
            assert archive.read(name + '/input.bin') == case['input']
            assert archive.read(name + '/expected.bin') == case['expected']
        for row in report['results']:
            prefix = row['level'] + '/' + row['name'] + '/' + row['compiler']
            output, binary = archive.read(prefix + '.output.bin'), archive.read(prefix + '.cubin')
            assert hashlib.sha256(binary).hexdigest() == row['cubin_sha256']
            assert hashlib.sha256(output).hexdigest() == row['output_sha256']
            assert output == GUARD + fixtures[row['name']]['expected'] + GUARD
            assert row['mismatches'] == 0 and row['guards'] and row['inputs_unchanged']
    backend = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ('openassembler/backend/ra.py', 'openassembler/backend/contracts.py')}
    assert all(value == report['source_hashes'][name] for name, value in backend.items())
    return {'schema': 1, 'passed': True, 'scope': 'Offline retained hardware-output/binary custody audit and exact promoted backend hashes',
            'python': sys.version, 'hardware_report': report_path.name,
            'launches_checked': len(domains), 'output_bytes_checked': sum(r['compared_bytes'] for r in report['results']),
            'backend_hashes': backend, 'archive_sha256': report['artifact_archive']['sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = validate(args.report)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
