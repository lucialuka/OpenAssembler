"""Read-only audit of all archived atomic outputs and negative reference controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
from cases import GUARD, HEADER, cases, check, exchange_chain, inputs


def audit(report_path):
    report = json.loads(report_path.read_text())
    archive_path = report_path.with_name(report['artifact_archive']['path'])
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == report['artifact_archive']['sha256']
    fixtures = {case['name']: case for case in cases()}
    backend = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
               for name in ('openassembler/backend/ra.py','openassembler/backend/contracts.py')}
    assert all(value == report['source_hashes'][name] for name,value in backend.items())
    expected = {(name, level, compiler, mode, repeat) for name in fixtures for level in ('safe','mined','sched','opt')
                for compiler in ('oa','ptxas') for mode in (0,1) for repeat in range(3 if mode else 1)}
    assert len(report['results']) == len(expected)
    assert {(r['name'],r['level'],r['compiler'],r['mode'],r['repeat']) for r in report['results']} == expected
    with zipfile.ZipFile(archive_path) as archive:
        combined = HEADER + '\n'.join(case['source'][len(HEADER):].replace('probe(',case['name']+'(') for case in fixtures.values())
        assert archive.read('fixtures.ptx') == combined.encode()
        for row in report['results']:
            case = fixtures[row['name']]
            before = inputs(case,row['mode'])
            raw = {name: archive.read(f'{row["level"]}/{row["name"]}/{row["compiler"]}/{row["mode"]}/{row["repeat"]}/{name}.bin') for name in before}
            for name,data in before.items():
                assert archive.read(row['name']+'/'+str(row['mode'])+'/before-'+name+'.bin') == data
                assert len(raw[name]) == len(data)+512
                assert raw[name][:256] == raw[name][-256:] == GUARD
                assert hashlib.sha256(raw[name]).hexdigest() == row['output_hashes'][name]
            actual = {name:data[256:-256] for name,data in raw.items()}
            validation = check(case,row['mode'],before,actual)
            assert validation == row['validation'] and validation['passed']
            binary = archive.read(row['level']+'/'+row['name']+'/'+row['compiler']+'.cubin')
            assert hashlib.sha256(binary).hexdigest() == row['cubin_sha256']
        # Reference checker must reject complete-state and old-return corruption,
        # independently of ordinary output hash custody.
        first = next(row for row in report['results'] if row['name']=='atom_global_add_u32' and row['mode']==1)
        before = inputs(fixtures[first['name']],1)
        prefix = f'{first["level"]}/{first["name"]}/{first["compiler"]}/1/{first["repeat"]}/'
        actual = {name:archive.read(prefix+name+'.bin')[256:-256] for name in before}
        negatives = []
        for name,offset in (('state',4),('output',0)):
            broken = dict(actual);data=bytearray(broken[name]);data[offset] ^= 1;broken[name]=bytes(data)
            result = check(fixtures[first['name']],1,before,broken)
            assert not result['passed']
            negatives.append({'mutation':name,'offset':offset,'rejected':True,'failures':result['failures']})
        assert not exchange_chain(0,1,[1,2,3],[0,3,2])
        negatives.append({'mutation':'disconnected exchange cycle with conserved value multiset',
                          'initial':0,'final':1,'values':[1,2,3],'returns':[0,3,2],'rejected':True})
    return {'schema':1,'passed':True,'python':sys.version,
            'scope':'read-only full atomic output/reference and binary/source identity audit; no new hardware dispatch',
            'hardware_report':report_path.name,'launches_checked':len(expected),
            'bytes_checked':sum(r['compared_bytes'] for r in report['results']),
            'archive_sha256':report['artifact_archive']['sha256'],'negative_controls':negatives,
            'hardware_fixture_sha256':report['source_hashes']['compiler_exp/gap_closure/atomics/cases.py'],
            'current_reference_sha256':hashlib.sha256(Path(__file__).with_name('cases.py').read_bytes()).hexdigest(),
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'backend_hashes':backend}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=audit(args.report);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
