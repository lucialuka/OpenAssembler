"""Custody, exact executable continuity and complete checkpoint state audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.container.cubin import regcount,text_words
from openassembler.modelfront.kernels import canonical
from openassembler.ptx.parse import parse_module

SOURCES=('checkpoint.py','runtime.py','autograd.py','replay.py','optimizer.py','kernels_optimizer.py',
         'kernels_semantics.py','kernels.py','reference.py','frontend.py','worker.py')

def sha(data):return hashlib.sha256(data).hexdigest()
def successful(pattern):
    for p in sorted(HERE.glob(pattern),reverse=True):
        r=json.loads(p.read_text())
        if r.get('returncode')==0 and r.get('report',{}).get('passed'):return p
    raise ValueError('no successful hardware gate '+pattern)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);parser.add_argument('--maximum-report',type=Path);args=parser.parse_args()
    path=(args.report or successful('hardware-*.json')).resolve();max_path=(args.maximum_report or successful('maximum-20*.json')).resolve()
    result={'schema':1,'passed':False,'hardware':'not run; source/archive/native text custody only','reports':[],
            'auditor_sha256':sha(Path(__file__).read_bytes()),'kernel_comparisons':0,'checkpoint_pairs':0}
    cpu=json.loads((HERE/'cpu-results.json').read_text());candidate=json.loads((HERE/'candidate-results.json').read_text())
    assert cpu['passed'] and candidate['passed'] and candidate['candidate_sha256']==sha((HERE/'candidate.py').read_bytes())
    cache={}
    for report_path,kind in ((path,'full-state'),(max_path,'maximum-ieee')):
        r=json.loads(report_path.read_text());q=r['report'];assert r['returncode']==0 and q['passed']
        archive=report_path.parent/r['archive']['path'];assert sha(archive.read_bytes())==r['archive']['sha256']
        for name in SOURCES:
            local=ROOT/'openassembler/modelfront'/name;digest=sha(local.read_bytes())
            assert digest==q['source_hashes']['modelfront/'+name],(report_path,name)
            assert digest==cpu['source_hashes'][local.relative_to(ROOT).as_posix()],name
        if kind=='full-state':
            for name,digest in r['harness_sha256'].items():
                local=(HERE.parent/'optimizers/hardware_gate.py') if name=='optimizer_gate.py' else (HERE.parent/'optimizers'/name) if name.startswith('repo/') else HERE/name
                assert sha(local.read_bytes())==digest,name
        else:assert sha((HERE/'maximum_gate.py').read_bytes())==r['harness_sha256']['hardware_gate.py']
        rows=[]
        with zipfile.ZipFile(archive) as z:
            assert 'work/full-tensors.pt' in z.namelist()
            for name in z.namelist():
                if not name.endswith('/native-ledger.json'):continue
                prefix=name.removesuffix('native-ledger.json');ledger=json.loads(z.read(name));plan=json.loads(z.read(prefix+'plan.json'))
                assert sha(canonical(plan))==ledger['plan_sha256']
                source=z.read(prefix+'sources/native.ptx');assert sha(source)==ledger['source_sha256']
                binary=z.read(prefix+ledger['cubin']);assert sha(binary)==ledger['cubin_sha256']
                module=parse_module(source.decode());assert set(module.kernels)==set(ledger['kernels'])
                for entry,kernel in module.kernels.items():
                    key=sha(canonical({'kernel':__import__('dataclasses').asdict(kernel),'opt':'opt'}))
                    if key not in cache:
                        generated=build.generate(kernel,'sm_80','opt');cache[key]=(generated[2],generated[4])
                    words,registers=cache[key]
                    assert words==text_words(binary,entry) and registers==regcount(binary,entry)==ledger['kernels'][entry]['registers']
                    assert sha(b''.join(w.to_bytes(16,'little') for w in words))==ledger['kernels'][entry]['text_sha256']
                    result['kernel_comparisons']+=1
                rows.append(prefix)
            if kind=='full-state':
                assert len(rows)==26
                assert len(q['cases'])==12 and all(c['all_checkpoint_bytes_equal'] for c in q['cases'])
                for case in q['cases']:
                    prefix='work/'+case['name']
                    assert z.read(prefix+'-resumed-final.oa-checkpoint')==z.read(prefix+'-final.oa-checkpoint')
                    result['checkpoint_pairs']+=1
                    if case['compiled_module_resume']:
                        assert z.read(prefix+'-module-final.oa-checkpoint')==z.read(prefix+'-final.oa-checkpoint')
                        result['checkpoint_pairs']+=1
                assert all(row['passed'] for row in q['optimizer_gate']['comparisons'])
                for reports in q['optimizer_gate']['cli_reports'].values():
                    for report in reports.values():assert report['passed'] and report['vendor_compute_launches']==0 and report['iterations']==6
            else:
                assert len(rows)==2 and len(q['runs'])==2
                for case in q['runs']:
                    prefix='work/'+case['optimizer']
                    assert z.read(prefix+'-ieee.oa-checkpoint')==z.read(prefix+'-resaved.oa-checkpoint')
                    result['checkpoint_pairs']+=1
            assert all(row['passed'] for row in q['comparisons'])
        result['reports'].append({'kind':kind,'path':report_path.relative_to(ROOT).as_posix(),'sha256':sha(report_path.read_bytes()),
          'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':r['archive']['sha256'],'bundles':len(rows),
          'full_comparisons':len(q['comparisons']),'elements':sum(x['elements'] for x in q['comparisons']),'device':q['device']})
    result.update(passed=True,unique_generated_kernels=len(cache),source_identities_checked=list(SOURCES))
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
