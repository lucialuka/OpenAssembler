"""Read-only full-output, executable and source custody for ordering candidates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE), str(HERE.parent / 'atomics')]
from cases import HEADER, check, exchange_chain, inputs
from ordered_cases import cases
from openassembler.build import generate
from openassembler.container.cubin import regcount, text_words
from openassembler.ptx.parse import parse_module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_probe(name, repeat, raw):
    for value in raw.values():
        assert value[:256] == value[-256:] == b'\xa5' * 256
    data = {key: value[256:-256] for key, value in raw.items()}
    if name == 'pointer_branch':
        expected = struct.pack('<32I', *[0x12340000 + i + repeat * 37 for i in range(32)])
        assert data == {'input': expected, 'output': expected}
    elif name == 'wide_snapshot':
        assert data == {'output': struct.pack('<32Q', *[i * 7 for i in range(32)])}
    elif name.startswith('shared'):
        assert set(data) == {'output'} and len(data['output']) == 2320
        output = struct.unpack('<580I', data['output'])
        for cta in range(4):
            assert all(output[cta * 128 + i] == 0x13579bdf for i in range(1, 128, 2))
            for bin_index in range(17):
                lanes = (list(range(0, 128, 2)) if bin_index == 1 else []) if name.endswith('_symbol') else [i for i in range(bin_index, 128, 17) if i % 2 == 0]
                old = [output[cta * 128 + i] for i in lanes]
                final = output[512 + cta * 17 + bin_index]
                if 'add' in name:
                    initial = 0xffffffff if name.endswith('_s32') else 0
                    assert final == (initial + len(lanes)) & 0xffffffff
                    assert Counter(old) == Counter((initial + i) & 0xffffffff for i in range(len(lanes)))
                else:
                    assert exchange_chain(0, final, [cta * 128 + i + 1 for i in lanes], old)
    else:
        assert set(data) == {'output', 'payload', 'flag', 'ack'}
        assert len(data['output']) == 12 and data['output'][:8] == bytes(8)
        assert data['flag'] == data['ack'] == struct.pack('<I', 128)
        assert data['payload'] == struct.pack('<16I', *[(128 * 0x1f123bb5 + i) & 0xffffffff for i in range(16)])


def audit(path, *, candidate=False):
    report = json.loads(path.read_text())
    descriptor = report.get('archive', report.get('artifact_archive'))
    archive_path = path.parent / descriptor['path']
    assert sha(archive_path.read_bytes()) == descriptor['sha256']
    assert report['passed'] and report['device']['cc'] == '8.0'
    proofs, compared_bytes, negative_controls = [], 0, []
    forms = 'artifact_archive' in report
    fixtures = {c['name']: c for c in cases()}
    backend = {f'openassembler/backend/{name}.py': sha((HERE / f'candidate-{name}.py' if candidate else ROOT / f'openassembler/backend/{name}.py').read_bytes())
               for name in ('ra', 'contracts', 'compile', 'sched')}
    assert all(report['source_hashes'][name] == value for name, value in backend.items())
    with zipfile.ZipFile(archive_path) as archive:
        source = archive.read('fixtures.ptx' if forms else 'probes.ptx').decode()
        module = parse_module(source)
        if forms:
            combined = HEADER + '\n'.join(c['source'][len(HEADER):].replace('probe(', c['name'] + '(') for c in fixtures.values())
            assert source == combined
            assert set(module.kernels) == set(fixtures)
            expected = {(name, level, compiler, mode, repeat)
                        for name in fixtures for level in ('safe', 'mined', 'sched', 'opt')
                        for compiler in ('oa', 'ptxas') for mode in (0, 1)
                        for repeat in range(3 if mode else 1)}
            actual = {(r['name'], r['level'], r['compiler'], r['mode'], r['repeat']) for r in report['results']}
        else:
            fixed = json.loads((ROOT / 'tests/data/atomic_ordering.json').read_text())
            acqrel = fixed['publication'].replace('publication(', 'both_acqrel(').replace('atom.release.gpu.', 'atom.acq_rel.gpu.').replace('atom.acquire.gpu.', 'atom.acq_rel.gpu.')
            probes = [fixed['publication'], acqrel, *[c['source'] for c in fixed['shared']], fixed['pointer'], fixed['snapshot']]
            assert source == HEADER + '\n'.join(text[len(HEADER):] for text in probes)
            assert set(module.kernels) == {'publication', 'both_acqrel', 'shared_add', 'shared_add_explicit',
                'shared_exch', 'shared_exch_explicit', 'shared_add_s32', 'shared_add_symbol',
                'shared_exch_symbol', 'pointer_branch', 'wide_snapshot'}
            expected = {(name, level, compiler, repeat)
                        for name in module.kernels for level in ('safe', 'mined', 'sched', 'opt')
                        for compiler in ('oa', 'ptxas') for repeat in range(16 if name in ('publication', 'both_acqrel') else 3)}
            actual = {(r['name'], r['level'], r['compiler'], r['repeat']) for r in report['results']}
        assert actual == expected and len(report['results']) == len(expected)
        for level in ('safe', 'mined', 'sched', 'opt'):
            for name, kernel in module.kernels.items():
                binary = archive.read(f'{level}/{name}/oa.cubin' if forms else f'{level}/oa.cubin')
                _, _, words, _, registers = generate(kernel, 'sm_80', level)
                assert words == text_words(binary, name) and registers == regcount(binary, name), (level, name)
                proofs.append({'level': level, 'entry': name, 'words_sha256': sha(b''.join(w.to_bytes(16, 'little') for w in words)),
                               'registers': registers, 'passed': True})
        for row in report['results']:
            name, level, compiler, repeat = (row[k] for k in ('name', 'level', 'compiler', 'repeat'))
            if forms:
                before = inputs(fixtures[name], row['mode'])
                prefix = f'{level}/{name}/{compiler}/{row["mode"]}/{repeat}/'
                raw = {key: archive.read(prefix + key + '.bin') for key in before}
                for key, value in raw.items():
                    assert len(value) == len(before[key]) + 512
                    assert value[:256] == value[-256:] == b'\xa5' * 256
                    assert sha(value) == row['output_hashes'][key]
                    assert archive.read(f'{name}/{row["mode"]}/before-{key}.bin') == before[key]
                assert check(fixtures[name], row['mode'], before, {k: v[256:-256] for k, v in raw.items()}) == row['validation']
                assert row['validation']['passed'] and row['guards'] and row['inputs_unchanged']
            else:
                prefix = f'{level}/{compiler}/{name}/{repeat}/'
                raw = {key: archive.read(prefix + key + '.bin') for key in row['output_sha256']}
                assert all(sha(value) == row['output_sha256'][key] for key, value in raw.items())
                check_probe(name, repeat, raw)
                assert row['passed'] and row['guards'] and not row['failures']
                if level == 'safe' and compiler == 'oa' and repeat == 0:
                    key = 'payload' if name == 'publication' else 'output'
                    broken = dict(raw)
                    value = bytearray(broken[key]); value[256] ^= 1; broken[key] = bytes(value)
                    try:
                        check_probe(name, repeat, broken)
                    except AssertionError:
                        negative_controls.append({'case': name, 'corrupted': key, 'rejected': True})
                    else:
                        raise AssertionError('reference admitted corrupted ' + name)
            binary = archive.read(f'{level}/{name}/{compiler}.cubin' if forms else f'{level}/{compiler}.cubin')
            assert sha(binary) == row['cubin_sha256']
            compared_bytes += sum(map(len, raw.values()))
    return {'schema': 1, 'passed': True, 'scope': 'Complete archived state/reference/cubin custody and current exact executable regeneration; no new hardware or timing',
            'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path.read_bytes()),
            'archive_sha256': descriptor['sha256'], 'launches': len(expected), 'bytes_checked': compared_bytes,
            'native_comparisons': proofs, 'negative_controls': negative_controls,
            'audited_backend': 'isolated candidate' if candidate else 'production',
            'source_hashes': backend,
            'hardware_source_hashes': report['source_hashes'], 'auditor_sha256': sha(Path(__file__).read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    result = audit(args.report.resolve(), candidate=args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['report'], result['launches'], result['bytes_checked'], len(result['native_comparisons']))
