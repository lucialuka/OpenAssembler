"""Audit the observed toolkit layout against an independent natural scalar layout."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    path = HERE/'baseline-layout.json'
    report = json.loads(path.read_text())
    assert report['passed'] and report['production_fix'] is False
    assert all(sha(HERE/name) == digest for name, digest in report['artifact_hashes'].items())
    assert report['script_sha256'] == sha(HERE/'probe.py')
    assert len(report['scalar_load_rejections']) == 6 and all(row['rejected'] for row in report['scalar_load_rejections'])
    parameters = re.findall(r'\.param\s+\.([us][0-9]+)\s+(\w+)', (HERE/'source.ptx').read_text())
    assert len(parameters) == 7
    expected, offset = [], 0
    for ordinal, (dtype, name) in enumerate(parameters):
        size = int(dtype[1:])//8
        offset = (offset+size-1)//size*size
        expected.append(dict(ordinal=ordinal, name=name, dtype=dtype, offset=offset, size=size))
        offset += size
    observed = {}
    for row in report['toolkit_parameter_records']:
        if row['name'] == 'KPARAM_INFO':
            index, packed, attributes = row['u32_words']
            assert index == 0
            ordinal, slot, size = packed & 65535, packed >> 16, attributes >> 18
            assert ordinal not in observed
            observed[ordinal] = dict(offset=slot, size=size)
        elif row['name'] == 'CBANK_PARAM_SIZE':
            bank_size = int.from_bytes(bytes.fromhex(row['payload_hex']), 'little')
    assert set(observed) == set(range(7)) and bank_size == offset == 26
    for row in expected:
        assert observed[row['ordinal']] == dict(offset=row['offset'], size=row['size'])
    result = dict(schema=1, passed=True, parameters=expected, parameter_bank_bytes=bank_size,
                  toolkit_record_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
                  scope='This exact seven-parameter ptxas control matches independently computed natural scalar sizes/alignment: offsets 0/1/2/4/8/16/24, 26 bytes. KPARAM fields interpreted only for the observed records; no universal metadata schema, OA admission, GPU execution or numerical claim')
    (HERE/'layout-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Passed exact seven-parameter toolkit layout: offsets 0/1/2/4/8/16/24, 26 bytes')
