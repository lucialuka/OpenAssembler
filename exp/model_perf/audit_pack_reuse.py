"""Independently check retained complete states, native modules and timing samples."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

import torch

from openassembler.container.cubin import regcount, text_words


def digest(data):
    return sha256(data).hexdigest()


def check_text(binary, name, record):
    words = text_words(binary, name)
    assert digest(b''.join(word.to_bytes(16, 'little') for word in words)) == record['text_sha256']
    assert regcount(binary, name) == record['registers']
    assert record['compiled_by'] == 'openassembler'


def run(directory):
    envelope_path = directory / 'result.json'
    envelope = json.loads(envelope_path.read_text())
    assert envelope['returncode'] == 0 and envelope['result']['passed'] is True
    result = envelope['result']
    expected_arms = {'incumbent', 'candidate', 'incumbent_ptxas', 'candidate_ptxas', 'triton'}
    report = {'passed': False, 'result_sha256': digest(envelope_path.read_bytes()),
              'archive_sha256': digest((directory / 'artifacts.zip').read_bytes()),
              'environment': result['environment'], 'auditor_python': sys.version,
              'auditor_torch': torch.__version__, 'models': []}
    report['production_sources'] = {name: value for name, value in envelope['source_sha256'].items()
                                    if name.startswith('openassembler/')}
    assert all(Path(name).is_file() and digest(Path(name).read_bytes()) == value
               for name, value in report['production_sources'].items())
    combinations = set()
    with zipfile.ZipFile(directory / 'artifacts.zip') as archive:
        for model in result['models']:
            key = (model['task'], model['storage_views'], model['reuse_storage'])
            assert key not in combinations
            combinations.add(key)
            root = 'results/' + '-'.join(str(x) for x in key) + '/'
            assert model['ledgers']['incumbent'] == model['ledgers']['candidate']
            ledger = model['ledgers']['candidate']
            native = archive.read(root + 'incumbent/' + ledger['cubin'])
            assert digest(native) == ledger['cubin_sha256']
            for name, record in ledger['kernels'].items():
                check_text(native, name, record)
            for name, record in ledger['tensor_kernels'].items():
                binary = archive.read(root + 'incumbent/' + record['module'])
                assert digest(binary) == record['cubin_sha256']
                check_text(binary, name, record)
                ptx = Path(record['module']).with_name('kernel.ptx').as_posix()
                assert digest(archive.read(root + 'incumbent/' + ptx)) == record['source_sha256']

            checks = {(row['iteration'], row['arm'], row['field']): row for row in model['validation']}
            assert len(checks) == len(model['validation'])
            assert {arm for _, arm, _ in checks} == expected_arms
            assert {i for i, _, _ in checks} == set(range(12))
            assert len(set(Counter((i, arm) for i, arm, _ in checks).values())) == 1
            for row in checks.values():
                assert row['mismatches'] == 0 and row['elements'] > 0
                assert math.isfinite(row['max_abs']) and row['max_abs'] >= 0
            snapshot_fields, snapshot_elements = 0, 0
            for i in (0, 6):
                for arm in sorted(expected_arms):
                    data = archive.read(root + f'complete-state-{i}-{arm}.pt')
                    snapshot = torch.load(BytesIO(data), weights_only=True, map_location='cpu')
                    assert snapshot['actual'].keys() == snapshot['expected'].keys()
                    for field, actual in snapshot['actual'].items():
                        expected = snapshot['expected'][field]
                        row = checks[i, arm, field]
                        assert digest(actual.numpy().tobytes()) == row['actual_sha256']
                        assert digest(expected.numpy().tobytes()) == row['expected_sha256']
                        assert actual.numel() == row['elements']
                        torch.testing.assert_close(actual, expected, rtol=.002, atol=2e-5)
                        snapshot_fields += 1
                        snapshot_elements += actual.numel()

            intervals = [model['total'], *model['parts'].values()]
            for rounds in intervals:
                assert len(rounds) == 3
                for row in rounds:
                    assert set(row['samples_ms']) == expected_arms
                    assert len(row['orders']) == 15
                    assert all(len(order) == len(expected_arms) and set(order) == expected_arms
                               for order in row['orders'])
                    for arm, samples in row['samples_ms'].items():
                        assert len(samples) == 15 and all(math.isfinite(x) and x > 0 for x in samples)
                        assert statistics.median(samples) == row['medians_ms'][arm]
            assert all(audit['passed'] for audit in model['audits'].values())
            assert model['audits']['candidate']['kernel_counts'] == model['audits']['candidate_ptxas']['kernel_counts']
            expected_launches = (14, 12) if model['task'] == 'projection_inference' else (32, 32)
            assert (model['audits']['incumbent']['observed_compute_launches'],
                    model['audits']['candidate']['observed_compute_launches']) == expected_launches
            report['models'].append({'task': key[0], 'storage_views': key[1], 'reuse_storage': key[2],
                                     'complete_field_checks': len(checks), 'snapshot_fields': snapshot_fields,
                                     'snapshot_elements': snapshot_elements,
                                     'timing_samples': sum(len(r['orders']) * len(expected_arms) for rounds in intervals for r in rounds)})
    assert combinations == {(task, views, reuse) for task in ('dense_regression_fp16', 'projection_inference')
                            for views in (False, True) for reuse in (False, True)}
    report['passed'] = True
    report['auditor_sha256'] = digest(Path(__file__).read_bytes())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.directory)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'models': len(result['models']),
                      'snapshot_elements': sum(row['snapshot_elements'] for row in result['models'])}))
