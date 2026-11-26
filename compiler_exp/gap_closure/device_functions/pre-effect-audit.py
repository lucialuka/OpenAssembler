"""Source-exact direct-call custody, complete independent byte/resource/native audit."""
import argparse,hashlib,json,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(HERE)]
from fixtures import expanded_cases as cases,values,expected

def sha(data):return hashlib.sha256(data).hexdigest()
AUDITOR=sha(Path(__file__).read_bytes())
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true');args=parser.parse_args()
    if args.candidate:
        from candidate import install
        install()
    from openassembler import build
    from openassembler.ptx.parse import parse_module
    from openassembler.container.cubin import text_words,regcount
    from openassembler.backend.storage import local_frame_inventory
    from openassembler.native_ownership import validate
    path=args.report.resolve();report=json.loads(path.read_text());assert report['passed']and len(report['results'])==960
    replacements={'openassembler/ptx/parse.py':'candidate-parse.py','openassembler/ptx/inlining.py':'inlining.py','openassembler/build.py':'candidate-build.py','openassembler/native_ownership.py':'candidate-native-ownership.py'}
    for name,digest in report['source_hashes'].items():
        p=HERE/replacements[name]if args.candidate and name in replacements else ROOT/name
        assert sha(p.read_bytes())==digest,name
    assert report['harness_sha256']==sha((HERE/'run_modal.py').read_bytes())
    archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    fixtures={c['name']:c for c in cases()};domain={(name,level,compiler,n,seed)for name in fixtures for level in('safe','mined','sched','opt')for compiler in('oa','ptxas')for n in(0,1,259,384)for seed in(17341,17342)}
    assert{tuple(row[k]for k in('name','level','compiler','n','seed'))for row in report['results']}==domain
    native=[];count=0;guard=b'\xa5'*256
    with zipfile.ZipFile(archive_path)as archive:
        binaries={};manifests={}
        for name,case in fixtures.items():
            kind='cu'if'cuda_source'in case else'ptx';source=case.get('cuda_source',case['source'])
            assert archive.read(name+'/input.'+kind)==source.encode()
            emitted=archive.read(name+'/source.ptx').decode()
            if kind=='ptx':assert emitted==case['source']
            else:assert '.func'in emitted and 'call.uni'in emitted
            module=parse_module(emitted)
            for level in('safe','mined','sched','opt'):
                prefix=name+'/'+level+'/';assert archive.read(prefix+'emitted.ptx')==emitted.encode()
                manifest=json.loads(archive.read(prefix+'manifest.json'));entry=manifest['entries'][0];manifests[name,level]=entry
                assert entry['source_sha256']==sha(source.encode())and entry['language']==('cuda'if kind=='cu'else'ptx')
                contract=entry['kernels'][0];observed=contract['inlined_device_calls'];wanted={'schema':1,'strategy':'direct-scalar-static-inlining','functions':{target:{**module.functions[target],'call_sites':number}for target,number in module.inlined_calls[case['entry']].items()}}
                assert observed==json.loads(json.dumps(wanted))
                binary=archive.read(prefix+'oa.cubin');validate(entry,binary)
                resources={};_,_,words,_,registers=build.generate(module.kernels[case['entry']],'sm_80',level,resource_info=resources)
                assert words==text_words(binary,case['entry'])and registers==regcount(binary,case['entry'])
                assert resources['local_bytes']==contract.get('local_bytes',0)==(64 if name=='local'else 0)
                native.append(dict(name=name,level=level,registers=registers,local_bytes=resources['local_bytes'],passed=True))
                for compiler in('oa','ptxas'):binaries[name,level,compiler]=archive.read(prefix+compiler+'.cubin')
        for row in report['results']:
            name,level,compiler,n,seed=(row[k]for k in('name','level','compiler','n','seed'));prefix=f'{name}/{level}/{compiler}/{n}/{seed}/'
            before=archive.read(prefix+'before-input.bin');unchanged=archive.read(prefix+'input.bin');actual=archive.read(prefix+'output.bin')
            data=values(seed,384,fixtures[name]['mode']);assert before==guard+struct.pack('<384Q',*data)+guard
            wanted=guard+expected(data,fixtures[name],n)+guard
            assert len(actual)==3584 and actual==wanted and before==unchanged and row['passed']
            assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']
            binary=binaries[name,level,compiler];assert sha(binary)==row['cubin_sha256']
            frames=local_frame_inventory(binary);assert frames[fixtures[name]['entry']]==row['local_frame_bytes']
            if compiler=='oa':
                contract=manifests[name,level]['kernels'][0]
                assert row['resources']['num_regs']==regcount(binary,fixtures[name]['entry'])
                assert row['resources']['local_bytes']==contract.get('local_bytes',0)==row['local_frame_bytes']
            count+=len(actual)+len(unchanged)
        corrupted=bytearray(actual);corrupted[256]^=1;assert corrupted!=wanted
    result=dict(schema=1,passed=True,report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],launches=len(domain),bytes_checked=count,native_comparisons=native,audited_backend='isolated candidate'if args.candidate else'production',auditor_sha256=AUDITOR,corrupted_output_rejected=True,scope='Exact original PTX/CUDA source, emitted PTX, direct-call inventories/ownership, complete independent bytes/guards/input, native words/registers and local resources; no new execution or performance claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(len(domain),count,len(native))
