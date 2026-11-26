"""Complete original 512-launch gate custody without rebinding it to production."""
import hashlib,json,struct,types,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    path=HERE/'candidate-a100-20261123T173558Z.json';report=json.loads(path.read_text());assert report['passed']and len(report['results'])==512
    assert report['harness_sha256']==sha((HERE/'first-hardware-run-modal.py').read_bytes())
    assert report['source_hashes']['openassembler/ptx/parse.py']==sha((HERE/'first-hardware-parse.py').read_bytes())
    assert report['source_hashes']['openassembler/ptx/inlining.py']==sha((HERE/'first-hardware-inlining.py').read_bytes())
    assert report['source_hashes']['compiler_exp/gap_closure/device_functions/fixtures.py']==sha((HERE/'first-hardware-fixtures.py').read_bytes())
    reference=types.ModuleType('original_reference');reference.__file__=str(HERE/'first-hardware-fixtures.py');exec((HERE/'first-hardware-fixtures.py').read_bytes(),reference.__dict__)
    cases={c['name']:c for c in reference.cases()};assert len(cases)==8
    archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    count=0;guard=b'\xa5'*256
    with zipfile.ZipFile(archive_path)as archive:
        for name,case in cases.items():assert archive.read(name+'/source.ptx')==case['source'].encode()
        for row in report['results']:
            name,level,compiler,n,seed=(row[k]for k in('name','level','compiler','n','seed'));prefix=f'{name}/{level}/{compiler}/{n}/{seed}/'
            before=archive.read(prefix+'before-input.bin');actual=archive.read(prefix+'output.bin');unchanged=archive.read(prefix+'input.bin');data=reference.values(seed,384,cases[name]['mode'])
            assert before==guard+struct.pack('<384Q',*data)+guard and unchanged==before
            assert actual==guard+reference.expected(data,cases[name],n)+guard and row['passed']
            assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']
            assert sha(archive.read(name+'/'+level+'/'+compiler+'.cubin'))==row['cubin_sha256']
            count+=len(actual)+len(unchanged)
    result=dict(schema=1,passed=True,launches=512,bytes_checked=count,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],script_sha256=sha(Path(__file__).read_bytes()),scope='Original complete bytes/harness/source/archive custody; earlier candidate remains separate from final production')
    (HERE/'first-hardware-audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(report['results']),count)
