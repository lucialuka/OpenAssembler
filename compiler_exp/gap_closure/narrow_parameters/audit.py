"""Independent complete numerical/native custody, including actual explicit launch specs."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(HERE)]
from fixtures import module, CASES, NAMES, fields, inputs, expected


def sha(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--candidate',action='store_true')
    args = parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.container.cubin import text_words,regcount
    from openassembler.backend.storage import local_frame_inventory
    from openassembler.native_ownership import validate
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    assert report['passed'] and report['error'] is None
    replacements = {name:HERE/('candidate-'+suffix+'.py') for suffix,name in json.loads((HERE/'inputs.json').read_text()).items()}
    for name,digest in report['source_hashes'].items():
        path = replacements[name] if args.candidate and name in replacements else ROOT/name
        if name in ('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json') and sha(path.read_bytes()) != digest:
            previous=json.loads((HERE.parent/'python_boolean_abi/helpers/validation.json').read_text())
            assert previous['source_hashes'][name] == digest
            table=json.loads(path.read_text());policy_path=ROOT/table['source_continuity']['audit'];policy=json.loads(policy_path.read_text())
            assert policy['passed'] and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
            assert sha(policy_path.read_bytes())==table['source_continuity']['sha256']
            assert all(sha((ROOT/key).read_bytes())==value for key,value in policy['compiler_inputs_sha256'].items())
            continue
        assert sha(path.read_bytes()) == digest,name
    assert report['harness_sha256'] == sha((HERE/'run_modal.py').read_bytes())
    archive_path = report_path.parent/report['archive']['path']
    assert sha(archive_path.read_bytes()) == report['archive']['sha256']
    domain = {(compiler,name,seed,sample,n) for compiler in ('ptxas','oa-safe','oa-mined','oa-sched','oa-opt') for name in NAMES for seed in (49117,49118) for sample in range(6) for n in (0,1,91,128)}
    keys = [tuple(row[k] for k in ('compiler','entry','seed','sample','n')) for row in report['results']]
    assert len(keys) == len(set(keys)) == len(domain) == 14400 and set(keys) == domain
    spec_keys = [(row['entry'],row['level']) for row in report['explicit_launch_results']]
    assert len(spec_keys) == len(set(spec_keys)) == 240 and set(spec_keys) == {(name,level) for name in NAMES for level in ('safe','mined','sched','opt')}
    cases = {case['name']:case for case in CASES}
    native, count, explicit_bytes, binaries = [],0,0,{}
    guard = b'\xa5'*256
    with zipfile.ZipFile(archive_path) as archive:
        source = archive.read('original.ptx')
        assert source == module().encode()
        parsed = parse_module(source.decode())
        for level in ('safe','mined','sched','opt'):
            binary = archive.read(level+'/oa.cubin')
            manifest = json.loads(archive.read(level+'/manifest.json'))
            entry, = manifest['entries']
            assert entry['language'] == 'ptx' and entry['source_sha256'] == sha(source)
            validate(entry,binary)
            binaries['oa-'+level] = binary
            for record in entry['kernels']:
                name = record['name']
                assert record['params'] == [list(field) for field in fields(cases[name])]
                resources = {}
                _,_,words,_,registers = build.generate(parsed.kernels[name],'sm_80',level,resource_info=resources)
                assert words == text_words(binary,name) and registers == regcount(binary,name) == record['registers']
                assert resources['local_bytes'] == local_frame_inventory(binary)[name] == 0
                native.append(dict(entry=name,level=level,registers=registers,passed=True))
        binaries['ptxas'] = archive.read('ptxas.cubin')
        for row in report['results']:
            compiler,name,seed,sample,n = (row[k] for k in ('compiler','entry','seed','sample','n'))
            actual = archive.read(f'{compiler}/{name}/{seed}/{sample}/{n}/output.bin')
            values = inputs(cases[name],seed,sample,n=n)
            assert actual == guard+expected(cases[name],values,128)+guard
            assert row['passed'] and sha(actual) == row['output_sha256']
            assert row['cubin_sha256'] == sha(binaries[compiler])
            assert row['resources']['num_regs'] == regcount(binaries[compiler],name)
            if compiler.startswith('oa-'):assert row['resources']['local_bytes'] == 0
            count += len(actual)
        for row in report['explicit_launch_results']:
            name,level = row['entry'],row['level']
            prefix = 'explicit/'+level+'/'+name+'/'
            spec = json.loads(archive.read(prefix+'spec.json'))
            actual = archive.read(prefix+'output.bin')
            values = inputs(cases[name],49117,4,out=0,n=37)
            assert actual == guard+expected(cases[name],values,128)+guard
            assert spec['steps'][0]['kernel'] == name and spec['steps'][0]['block'] == 128
            arguments = spec['steps'][0]['args']
            assert len(arguments) == len(fields(cases[name]))
            for (dtype,parameter),argument in zip(fields(cases[name]),arguments):
                if parameter == 'out':assert argument == dict(buf='out',offset=256)
                else:
                    bits = int(dtype[1:]);raw = values[parameter]&((1 << bits)-1)
                    if dtype.startswith('s') and raw & (1 << (bits-1)):raw -= 1 << bits
                    assert argument == {('i' if dtype.startswith('s') else 'u')+str(bits):raw}
            assert row['passed'] and row['output_sha256'] == sha(actual)
            explicit_bytes += len(actual)
        corrupt = bytearray(actual)
        corrupt[256] ^= 1
        assert corrupt != guard+expected(cases[name],values,128)+guard
    assert len(native) == 240 and count == 66355200 and explicit_bytes == 1105920
    result = dict(schema=1,passed=True,report=str(report_path.relative_to(ROOT)),report_sha256=sha(report_path.read_bytes()),
                  archive_sha256=report['archive']['sha256'],auditor_sha256=sha(Path(__file__).read_bytes()),
                  audited_backend='isolated candidate' if args.candidate else 'production',launches=14400,
                  explicit_launches=240,bytes_checked=count+explicit_bytes,native_comparisons=native,
                  corrupted_output_rejected=True,
                  scope='Complete independent raw bytes for all 14400 original PTX hardware executions plus 240 actual explicit run-spec launches; exact original source/parameter signatures, spec numeric values/widths, native words/register/frame/ownership and guards/tails. No timing or universal ABI claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Passed 14400 original launches and 240 explicit launches;',count+explicit_bytes,'complete bytes; 240 native outputs')
