"""Retain complete failed and isolating A100 outputs without relabeling failures."""
import hashlib,json,struct,sys,zipfile,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
    fixtures=types.ModuleType('retained_reference');exec((HERE/'failed-fixtures.py').read_bytes(),fixtures.__dict__)
    cases={c['name']:c for c in fixtures.cases()};results=[]
    reports=['candidate-a100-20261123T152514Z.json','candidate-a100-20261123T153328Z.json','isolation-predication-20261123T153955Z.json','isolation-base_zero-20261123T154046Z.json','isolation-writers-20261123T154238Z.json','isolation-sync_writers-20261123T154641Z.json']
    for name in reports:
        path=HERE/name;report=json.loads(path.read_text());assert len(report['results'])==768
        harness={
            reports[0]:'failed-execution-run-modal.py',reports[1]:'barrier-wait-run-modal.py',
            reports[2]:'isolation-predication-run-modal.py',reports[3]:'isolation-base-zero-run-modal.py',
            reports[4]:'isolation-writers-run-modal.py',reports[5]:'isolation-run-modal.py'}[name]
        assert report['harness_sha256']==sha((HERE/harness).read_bytes())
        sources={'openassembler/pyfront/compile.py':'failed-candidate-compile.py',
                 'openassembler/backend/ra.py':'baseline-ra.py',
                 'openassembler/backend/compile.py':'barrier-wait-backend-compile.py'if name==reports[1]else'baseline-backend-compile.py'}
        assert all(report['source_hashes'][key]==sha((HERE/value).read_bytes())for key,value in sources.items())
        archive_path=HERE/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
        count=0;failures=[]
        with zipfile.ZipFile(archive_path)as archive:
            assert report['source_hashes']['compiler_exp/gap_closure/python_shared_wide/fixtures.py']==sha((HERE/'failed-fixtures.py').read_bytes())
            for row in report['results']:
                prefix='/'.join(str(row[k])for k in('level','compiler','name','n','seed'))+'/'
                before=archive.read(prefix+'before-input.bin');actual=archive.read(prefix+'output.bin');unchanged=archive.read(prefix+'input.bin')
                data=fixtures.values(row['seed'],384);guard=b'\xa5'*256
                assert before==guard+struct.pack('<384Q',*data)+guard and unchanged==before
                wanted=guard+fixtures.expected(data,cases[row['name']],row['n'],fixtures.delta(cases[row['name']],row['seed']))+guard
                passed=actual==wanted;assert passed==row['passed']
                assert actual[:256]==actual[-256:]==guard
                assert sha(actual)==row['output_sha256']and sha(unchanged)==row['input_sha256']
                assert sha(archive.read(row['level']+'/'+row['compiler']+'.cubin'))==row['cubin_sha256']
                if not passed:failures.append({k:row[k]for k in('name','level','compiler','n','seed')})
                count+=len(actual)+len(unchanged)
        assert report['passed']==(not failures)
        results.append(dict(report=name,report_sha256=sha(path.read_bytes()),archive_sha256=report['archive']['sha256'],passed=report['passed'],launches=768,bytes_checked=count,failures=failures))
    assert[sum(not row['passed']for row in json.loads((HERE/name).read_text())['results'])for name in reports]==[64,64,0,64,0,0]
    result=dict(schema=1,audit_completed=True,results=results,script_sha256=sha(Path(__file__).read_bytes()),scope='Full independent outputs, guards/input, binary identities and retained failed/ablation outcomes; does not bind varied sources to production or infer general convergence')
    (HERE/'failure-isolation-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(len(results),'reports;',sum(r['bytes_checked']for r in results),'bytes; original failures preserved')
