"""Independent full tensor, native executable and explicit-build custody audit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.container.cubin import text_words,regcount
from openassembler.modelfront.kernels import canonical
from openassembler.modelfront.profiling import audit_trace
from openassembler.ptx.parse import parse_module


def sha(data):return hashlib.sha256(data).hexdigest()


def audit(path,candidate):
    import torch
    r=json.loads(path.read_text());q=r['report']
    assert r['returncode']==0 and q['passed']
    archive_path=path.parent/r['archive']['path']
    assert sha(archive_path.read_bytes())==r['archive']['sha256']
    source=HERE/'candidate-decode.py' if candidate else ROOT/'openassembler/modelfront/decode.py'
    assert sha(source.read_bytes())==q['source_hashes']['openassembler/modelfront/decode.py']==r['compiler_hashes']['openassembler/modelfront/decode.py']
    for name,digest in r['compiler_hashes'].items():
        if name.endswith('.py') and name!='openassembler/modelfront/decode.py':assert sha((ROOT/name).read_bytes())==digest,name
    assert sha((HERE/'hardware_gate.py').read_bytes())==r['harness_sha256']['hardware_gate.py']
    native=[];cache={};bundle_count=0
    with zipfile.ZipFile(archive_path)as z:
        assert z.read('openassembler/modelfront/decode.py')==source.read_bytes()
        tensors=torch.load(io.BytesIO(z.read('work/full-tensors.pt')),map_location='cpu',weights_only=True)
        assert set(tensors)=={c['name']for c in q['comparisons']} and len(q['comparisons'])==126
        for comparison in q['comparisons']:
            pair=tensors[comparison['name']];actual,expected=pair['actual'],pair['expected']
            assert actual.shape==expected.shape and actual.numel()==comparison['elements']
            assert torch.isclose(actual,expected,rtol=1e-4,atol=1e-5).all() and comparison['passed']
        first=next(iter(tensors.values()))
        corrupted=first['actual'].clone().reshape(-1);corrupted[0]+=1
        assert not torch.isclose(corrupted,first['expected'].reshape(-1),rtol=1e-4,atol=1e-5).all()
        for name in z.namelist():
            if not name.endswith('/native-ledger.json'):continue
            bundle_count+=1;prefix=name.removesuffix('native-ledger.json')
            ledger=json.loads(z.read(name));plan=json.loads(z.read(prefix+'plan.json'))
            assert sha(canonical(plan))==ledger['plan_sha256']
            source_ptx=z.read(prefix+'sources/native.ptx');assert sha(source_ptx)==ledger['source_sha256']
            binary=z.read(prefix+ledger['cubin']);assert sha(binary)==ledger['cubin_sha256']
            for entry,kernel in parse_module(source_ptx.decode()).kernels.items():
                key=sha(canonical(__import__('dataclasses').asdict(kernel)))
                if key not in cache:
                    _,_,words,_,registers=build.generate(kernel,'sm_80','opt');cache[key]=(words,registers)
                words,registers=cache[key]
                assert words==text_words(binary,entry) and registers==regcount(binary,entry)==ledger['kernels'][entry]['registers']
                native.append(dict(bundle=prefix,entry=entry,registers=registers,passed=True))
        assert bundle_count==25,bundle_count
        for run in q['runs']:
            assert run['guards']['passed']
            result=audit_trace(json.loads(z.read('work/'+run['name']+'.trace.json')),run['launches'])
            assert result==run['audit'] and result['passed']
        assert len(q['runs'])==40 and all(n['passed']for n in q['failure_controls'])
        requests=q['requests'];assert len(requests)==16
        for suffix in ('00','01','10','11'):
            requested=[x for x in requests if x['artifact_key']=='attention-cache-'+suffix+'-v1']
            assert [x['past_length']for x in requested]==[3,4,5,6]
            assert all(x['next_length']==x['past_length']+1 and x['arch']=='sm_80'and x['math']=='fp32'for x in requested)
    return dict(schema=1,passed=True,scope='Read-only independent tensor/trace/source/archive and native word/register audit; no new GPU run or timing claim',
                report=str(path.relative_to(ROOT)),report_sha256=sha(path.read_bytes()),archive_sha256=r['archive']['sha256'],
                audited_backend='isolated candidate'if candidate else'production',decode_sha256=sha(source.read_bytes()),
                bundles=bundle_count,native_comparisons=native,unique_kernels=len(cache),
                comparisons=len(q['comparisons']),elements=sum(x['elements']for x in q['comparisons']),
                profiled_runs=len(q['runs']),profiled_launches=sum(x['audit']['observed_compute_launches']for x in q['runs']),
                corrupted_tensor_rejected=True,auditor_sha256=sha(Path(__file__).read_bytes()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args();result=audit(args.report.resolve(),args.candidate)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['bundles'],len(result['native_comparisons']),result['comparisons'],result['elements'],result['profiled_launches'])
