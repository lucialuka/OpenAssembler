"""Read-only full-output/source/native/resources audit of typed Python shared arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import HEADER,cases,values,expected,delta,SENTINEL
from openassembler import build
from openassembler.pyfront import compile_file
from openassembler.container.cubin import text_words,regcount
from openassembler.container.elf import Elf
from openassembler.backend.storage import local_frame_inventory
from openassembler.ptx.parse import parse_module

def sha(data):return hashlib.sha256(data).hexdigest()
AUDITOR=sha(Path(__file__).read_bytes())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and len(report['results'])==1280
    replacements={'openassembler/pyfront/compile.py':'candidate-compile.py','openassembler/backend/compile.py':'candidate-backend-compile.py','openassembler/backend/ra.py':'candidate-ra.py'}
    for name,digest in report['source_hashes'].items():
        source=HERE/replacements[name]if args.candidate and name in replacements else ROOT/name
        assert sha(source.read_bytes())==digest,name
    assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
    archive_path=path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    fixtures={c['name']:c for c in cases()};domain={(name,level,compiler,n,seed)for name in fixtures for level in('safe','mined','sched','opt')for compiler in('oa','ptxas')for n in(0,1,259,384)for seed in(14301,14302)}
    assert{tuple(r[k]for k in('name','level','compiler','n','seed'))for r in report['results']}==domain
    native=[];byte_count=0
    with zipfile.ZipFile(archive_path)as archive:
        source=HEADER+'\n'.join(c['source'][len(HEADER):]for c in fixtures.values());assert archive.read('sources.py')==source.encode()
        # Compile from the archived exact ordinary source path, without importing
        # or executing the user module. The frontend ignores its basename.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='oa-shared-audit-')as temporary:
            py=Path(temporary)/'sources.py';py.write_text(source);generated,names=compile_file(py)
        assert archive.read('generated.ptx')==generated.encode()and names==list(fixtures)
        module=parse_module(generated)
        for level in('safe','mined','sched','opt'):
            binary=archive.read(level+'/oa.cubin');manifest=json.loads(archive.read(level+'/manifest.json'));entry=manifest['entries'][0]
            from openassembler.native_ownership import validate
            validate(entry,binary);assert entry['language']=='python'and entry['source_sha256']==sha(source.encode())
            for name,kernel in module.kernels.items():
                _,_,words,_,registers=build.generate(kernel,'sm_80',level)
                assert words==text_words(binary,name)and registers==regcount(binary,name)
                assert Elf.parse(binary).section('.nv.shared.'+name).size==fixtures[name]['shared_bytes']
                native.append(dict(name=name,level=level,registers=registers,shared_bytes=fixtures[name]['shared_bytes'],passed=True))
        guard=b'\xa5'*256
        for row in report['results']:
            name,level,compiler,n,seed=(row[k]for k in('name','level','compiler','n','seed'));prefix=f'{level}/{compiler}/{name}/{n}/{seed}/'
            before=archive.read(prefix+'before-input.bin');unchanged=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin')
            data=values(seed,384);assert before==guard+struct.pack('<384Q',*data)+guard
            assert len(actual)==12800 and unchanged==before
            wanted=guard+expected(data,fixtures[name],n,delta(fixtures[name],seed))+guard
            assert actual==wanted and row['passed']
            assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']
            binary=archive.read(level+'/'+compiler+'.cubin');assert sha(binary)==row['cubin_sha256']
            assert local_frame_inventory(binary)[name]==row['local_frame_bytes']==row['resources']['local_bytes']==0
            assert Elf.parse(binary).section('.nv.shared.'+name).size==row['resources']['shared_bytes']==fixtures[name]['shared_bytes']
            if compiler=='oa':assert row['resources']['num_regs']==regcount(binary,name)
            byte_count+=len(actual)+len(unchanged)
        corrupted=bytearray(actual);corrupted[256]^=1;assert corrupted!=wanted
    result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],
                launches=len(domain),bytes_checked=byte_count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',
                auditor_sha256=AUDITOR,corrupted_output_rejected=True,scope='Exact ordinary Python source/frontend PTX, complete independent bytes/input/guards/tails, all native words/registers/resources and ownership; no new hardware or timing')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(result['launches'],byte_count,len(native))
