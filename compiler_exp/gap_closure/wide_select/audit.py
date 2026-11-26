"""Read-only independent bytes, source custody, native words and resource audit."""
import argparse
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from fixtures import expanded_cases, values, expected

def sha(data):
    return hashlib.sha256(data).hexdigest()

def compare_bytes(actual, wanted):
    assert actual == wanted, 'complete byte comparison failed'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.native_ownership import validate
    from openassembler.container.cubin import text_words, regcount
    from openassembler.backend.storage import local_frame_inventory
    path = args.report.resolve()
    report = json.loads(path.read_text())
    fixtures = {case['name']: case for case in expanded_cases()}
    assert len(fixtures) == 37 and report['passed'] and report['error'] is None
    replacements = {'openassembler/backend/ra.py': 'candidate-ra.py',
                    'openassembler/backend/contracts.py': 'candidate-contracts.py'}
    for name, digest in report['source_hashes'].items():
        current = HERE/replacements[name] if args.candidate and name in replacements else ROOT/name
        assert sha(current.read_bytes()) == digest, name
    assert report['harness_sha256'] == sha((HERE/'run_modal.py').read_bytes())
    archive_path = path.parent/report['archive']['path']
    assert sha(archive_path.read_bytes()) == report['archive']['sha256']
    domain = {(name, compiler, n, seed) for name in fixtures
              for compiler in ('ptxas', 'oa-safe', 'oa-mined', 'oa-sched', 'oa-opt')
              for n in (0, 1, 259, 384) for seed in (27391, 27392)}
    keys = [tuple(row[k] for k in ('name', 'compiler', 'n', 'seed')) for row in report['results']]
    assert len(keys) == len(set(keys)) == len(domain) == 1480 and set(keys) == domain
    native, count, guard = [], 0, b'\xa5'*256
    with zipfile.ZipFile(archive_path) as archive:
        binaries, contracts = {}, {}
        for name, case in fixtures.items():
            cuda = case['mode'] == 'cuda'
            source = case['cuda_source'] if cuda else case['source']
            assert archive.read(name+'/input'+('.cu' if cuda else '.ptx')) == source.encode()
            emitted = archive.read(name+'/source.ptx')
            if not cuda:
                assert emitted == source.encode()
            else:
                assert b'selp.b64' in emitted and b'call.uni' in emitted and b'.func' in emitted
            module = parse_module(emitted.decode())
            for compiler in ('ptxas', 'oa-safe', 'oa-mined', 'oa-sched', 'oa-opt'):
                binary = archive.read(name+'/'+compiler+'/native.cubin')
                binaries[name, compiler] = binary
                assert text_words(binary, case['entry'])
                if compiler == 'ptxas':
                    continue
                level = compiler.removeprefix('oa-')
                prefix = name+'/'+level+'/'
                assert archive.read(prefix+'emitted.ptx') == emitted
                manifest = json.loads(archive.read(prefix+'manifest.json'))
                entry, = manifest['entries']
                contract, = entry['kernels']
                assert entry['source_sha256'] == sha(source.encode())
                assert entry['language'] == ('cuda' if cuda else 'ptx')
                validate(entry, binary)
                resources = {}
                _, _, words, _, registers = build.generate(module.kernels[case['entry']], 'sm_80', level, resource_info=resources)
                assert words == text_words(binary, case['entry'])
                assert registers == regcount(binary, case['entry']) == contract['registers']
                assert resources['local_bytes'] == contract.get('local_bytes', 0) == local_frame_inventory(binary)[case['entry']] == 0
                if cuda:
                    assert set(module.inlined_calls[case['entry']]) == {'mix', 'next'}
                    assert set(contract['inlined_device_calls']['functions']) == {'mix', 'next'}
                contracts[name, compiler] = contract
                native.append(dict(name=name, level=level, registers=registers, local_bytes=0, passed=True))
        for row in report['results']:
            name, compiler, n, seed = (row[k] for k in ('name', 'compiler', 'n', 'seed'))
            prefix = f'{name}/{compiler}/{n}/{seed}/'
            before, unchanged, actual = (archive.read(prefix+key+'.bin') for key in ('before-input', 'input', 'output'))
            data = values(seed, 384)
            assert before == guard+struct.pack('<768Q', *data)+guard
            wanted = guard+expected(data, fixtures[name], n, 384)+guard
            assert len(actual) == 3584 and len(unchanged) == 6656
            compare_bytes(actual, wanted)
            compare_bytes(unchanged, before)
            assert row['passed'] and sha(actual) == row['output_sha256'] and sha(unchanged) == row['input_sha256']
            binary = binaries[name, compiler]
            assert sha(binary) == row['cubin_sha256']
            assert row['resources']['num_regs'] == regcount(binary, fixtures[name]['entry'])
            if compiler.startswith('oa-'):
                assert row['resources']['local_bytes'] == row['local_frame_bytes'] == contracts[name, compiler].get('local_bytes', 0) == 0
            else:
                assert row['local_frame_bytes'] is None
            count += len(actual)+len(unchanged)
        corrupted = bytearray(actual)
        corrupted[256] ^= 1
        try:
            compare_bytes(corrupted, wanted)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output control was not rejected')
    assert count == 15155200 and len(native) == 148
    result = dict(schema=1, passed=True, report=str(path.relative_to(ROOT)), report_sha256=sha(path.read_bytes()),
        archive_sha256=report['archive']['sha256'], launches=len(domain), bytes_checked=count,
        native_comparisons=native, auditor_sha256=sha(Path(__file__).read_bytes()),
        audited_backend='isolated candidate' if args.candidate else 'production', corrupted_output_rejected=True,
        scope='Exact original PTX/CUDA, all original-control and OA complete guarded input/output bytes, native words/registers/resources and ownership; independent bits include NaN payloads, signed zero, alias/false guard/predicate negation/parameter/immediate snapshots; no performance claim')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Passed', len(domain), 'launches;', count, 'bytes;', len(native), 'native comparisons')
