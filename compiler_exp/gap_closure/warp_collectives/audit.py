"""Read-only complete byte/reference/source/native audit for shuffle promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from cases import HEADER,cases,expected
from openassembler import build
from openassembler.container.cubin import regcount,text_words
from openassembler.ptx.parse import parse_module


def sha(data):return hashlib.sha256(data).hexdigest()


def audit(path,candidate):
    report=json.loads(path.read_text());assert report['passed']
    archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    for name in('ra','contracts'):
        source=HERE/f'candidate-{name}.py'if candidate else ROOT/f'openassembler/backend/{name}.py'
        assert sha(source.read_bytes())==report['source_hashes'][f'openassembler/backend/{name}.py']
    assert sha((HERE/'cases.py').read_bytes())==report['source_hashes']['compiler_exp/gap_closure/warp_collectives/cases.py']
    assert sha((HERE/'run_modal.py').read_bytes())==report['harness_sha256']
    fixtures={case['name']:case for case in cases()}
    domain={(name,level,compiler,threads,delta)for name in fixtures for level in('safe','mined','sched','opt')
            for compiler in('oa','ptxas')for threads in(32,128)for delta in(0,1,17,31,32,33,0xffffffff)}
    assert len(report['results'])==len(domain)
    assert{tuple(r[k]for k in('name','level','compiler','threads','delta'))for r in report['results']}==domain
    native=[];count=0;sentinel=0x13579bdf;guard=b'\xa5'*256
    with zipfile.ZipFile(archive_path)as archive:
        source=HEADER+'\n'.join(c['source'][len(HEADER):]for c in fixtures.values())
        assert archive.read('shuffle.ptx')==source.encode()
        module=parse_module(source)
        for level in('safe','mined','sched','opt'):
            binary=archive.read(level+'/oa.cubin')
            for name,kernel in module.kernels.items():
                _,_,words,_,registers=build.generate(kernel,'sm_80',level)
                assert words==text_words(binary,name)and registers==regcount(binary,name)
                native.append(dict(name=name,level=level,registers=registers,passed=True))
        for row in report['results']:
            name,level,compiler,threads,delta=(row[k]for k in('name','level','compiler','threads','delta'))
            prefix=f'{level}/{compiler}/{name}/{threads}/{delta}/'
            original=archive.read(prefix+'before-input.bin');after=archive.read(prefix+'input.bin');out=archive.read(prefix+'output.bin')
            assert len(original)==len(after)==len(out)==1024
            assert original[:256]==original[-256:]==out[:256]==out[-256:]==guard and original==after
            values=struct.unpack('<128I',original[256:-256]);reference=expected(fixtures[name],values[:threads],delta)+[sentinel]*(128-threads)
            wanted=guard+struct.pack('<128I',*reference)+guard
            assert out==wanted and row['passed']
            assert sha(after)==row['input_sha256']and sha(out)==row['output_sha256']
            assert sha(archive.read(level+'/'+compiler+'.cubin'))==row['cubin_sha256']
            count+=len(out)+len(after)
        corrupted=bytearray(out);corrupted[256]^=1;assert corrupted!=wanted
    return dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),
                archive_sha256=report['archive']['sha256'],audited_backend='isolated candidate'if candidate else'production',
                scope='Read-only complete shuffle state/input/reference and executable custody; no new hardware or timing',
                launches=len(domain),bytes_checked=count,native_comparisons=native,corrupted_output_rejected=True,
                auditor_sha256=sha(Path(__file__).read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    result=audit(args.report.resolve(),args.candidate);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['launches'],result['bytes_checked'],len(result['native_comparisons']))
