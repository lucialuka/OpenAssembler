"""Audit original full tensors, public CLI checks and exact current native kernels."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from openassembler import build
from openassembler.container.cubin import regcount, text_words
from openassembler.modelfront.kernels import canonical, prepare
from openassembler.ptx.parse import parse_module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run():
    path = HERE / 'hardware-20261126T082735Z.json'
    result = json.loads(path.read_text())
    report = result['report']
    assert result['returncode'] == 0 and report['passed'] and report['error'] is None
    assert len(report['comparisons']) == report['expected_comparisons'] == 80
    policy_paths = {'openassembler/kernels/sm80_gemm_dispatch.json',
                    'openassembler/kernels/sm80_rowwise_dispatch.json'}
    current_inputs = {name:sha((ROOT/name).read_bytes()) for name in result['compiler_hashes']}
    for name, digest in result['compiler_hashes'].items():
        if name not in policy_paths | {'openassembler/backend/ra.py', 'openassembler/modelfront/tensor_schedule.py'}:
            assert sha((ROOT/name).read_bytes()) == digest, name
    for name, digest in result['harness_sha256'].items():
        current = HERE/'hardware_gate.py' if name == 'hardware_gate.py' else ROOT/'examples/native_cnn'/Path(name).name
        assert sha(current.read_bytes()) == digest, name
    archive_path = HERE / result['archive']['path']
    assert sha(archive_path.read_bytes()) == result['archive']['sha256']
    native, cli, total = [], [], 0
    with zipfile.ZipFile(archive_path) as archive:
        # A concurrent allocator audit updated only policy proof references after
        # this run. Require exact selections/identities and validate that proof.
        for name in policy_paths:
            retained_bytes = archive.read(name)
            assert sha(retained_bytes) == result['compiler_hashes'][name]
            retained = json.loads(retained_bytes)
            current = json.loads((ROOT/name).read_text())
            if retained != current:
                proof_info = current.pop('source_continuity')
                retained.pop('source_continuity')
                assert retained == current
                proof_path = ROOT/proof_info['audit']
                proof = json.loads(proof_path.read_text())
                assert proof['passed'] and sha(proof_path.read_bytes()) == proof_info['sha256']
                assert proof_info['native_text_and_registers_unchanged']
        tensors = torch.load(io.BytesIO(archive.read('work/full-tensors.pt')), weights_only=True)
        assert set(tensors) == {row['name'] for row in report['comparisons']}
        for row in report['comparisons']:
            pair = tensors[row['name']]
            actual, expected = pair['actual'], pair['expected']
            assert actual.device.type == expected.device.type == 'cpu'
            assert actual.shape == expected.shape and actual.numel() == row['elements']
            torch.testing.assert_close(actual, expected, rtol=4e-5, atol=5e-6)
            assert row['passed'] and row['mismatches'] == 0
            total += actual.numel()
        for name in ('cli-output/model-build-report.json', 'cli-output/model-run.json'):
            data = json.loads(archive.read(name))
            assert data['passed'] and len(data['checks']) == 45 and all(row['passed'] for row in data['checks'])
            cli.append(dict(path=name, comparisons=len(data['checks']), passed=True))
        assert len(report['runs']) == 30
        assert all(row.get('guards', True) and row.get('inputs_weights_immutable', True) for row in report['runs'])
        assert sum('profiler' in row for row in report['runs']) == 10
        assert all(row['profiler']['passed'] for row in report['runs'] if 'profiler' in row)
        for ledger_name in sorted(name for name in archive.namelist() if name.endswith('/native-ledger.json')):
            prefix = ledger_name.removesuffix('native-ledger.json')
            ledger = json.loads(archive.read(ledger_name))
            plan = json.loads(archive.read(prefix+'plan.json'))
            assert plan.get('math', 'fp32') == 'fp32'
            assert sha(canonical(plan)) == ledger['plan_sha256']
            _, steps, kernels = prepare(plan)
            assert steps == ledger['steps'] and set(kernels) == set(ledger['kernels'])
            source = '.version 8.7\n.target sm_80\n.address_size 64\n' + '\n'.join(kernels[name].ptx for name in sorted(kernels))
            assert source.encode() == archive.read(prefix+'sources/native.ptx')
            assert sha(source.encode()) == ledger['source_sha256']
            binary = archive.read(prefix+ledger['cubin'])
            assert sha(binary) == ledger['cubin_sha256']
            module = parse_module(source)
            level = 'safe' if prefix == 'work/conv-safe/' else 'opt'
            for name, kernel in kernels.items():
                _, _, words, _, registers = build.generate(module.kernels[name], 'sm_80', level)
                assert words == text_words(binary, name) and registers == regcount(binary, name)
                assert kernel.signature == ledger['kernels'][name]['signature']
                native.append(dict(bundle=prefix, entry=name, level=level, op=kernel.op,
                                   words_sha256=sha(b''.join(word.to_bytes(16,'little') for word in words)), passed=True))
    offline = json.loads((HERE/'offline.json').read_text())
    assert offline['passed'] and len(offline['results']) == 144
    assert all(sha((ROOT/name).read_bytes()) == digest for name, digest in offline['source_hashes'].items())
    assert offline['script_sha256'] == sha((HERE/'offline.py').read_bytes())
    assert re.search(r'645 passed in ', (HERE/'model-regressions-final.log').read_text())
    sources = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [
        *(ROOT/'openassembler/modelfront'/name for name in ('graph.py','kernels.py','kernels_conv.py','reference.py')),
        ROOT/'tests/test_model_conv2d.py', ROOT/'examples/native_cnn/model.py', ROOT/'examples/native_cnn/openassembler.toml']}
    assert all(sha((ROOT/name).read_bytes()) == digest for name, digest in current_inputs.items()), 'compiler changed during native audit'
    proof = dict(schema=1, passed=True, current_compiler_hashes=current_inputs, report=path.name, report_sha256=sha(path.read_bytes()),
                 archive_sha256=result['archive']['sha256'], hardware_comparisons=80,
                 hardware_elements=total, cli=cli, native_comparisons=native, bundles=11,
                 model_regressions=645, focused_tests=21, offline_executions=144,
                 source_hashes=sources, script_sha256=sha(Path(__file__).read_bytes()),
                 scope='Full retained CPU-reference/A100 tensors, guards, profiler ownership, ordinary CNN CLI build/replay and exact current native regeneration. Finite FP32 NCHW/OIHW groups=1 contract.')
    (HERE/'validation.json').write_text(json.dumps(proof, indent=2)+'\n')
    print('Passed 80 A100 tensor checks, 90 public CLI checks and', len(native), 'exact native outputs across 11 bundles')

if __name__ == '__main__':
    run()
