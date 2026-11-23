"""Read-only optimizer experiment custody and default-policy continuity."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.modelfront.kernels import canonical,emit_kernel

if __name__=='__main__':
    path=HERE.parent/'checkpoints/hardware-20261121T140736Z.json';hardware=json.loads(path.read_text());report=hardware['report']['optimizer_gate']
    cpu=json.loads((HERE.parent/'checkpoints/cpu-results.json').read_text());candidate=json.loads((HERE/'candidate-results.json').read_text())
    assert hardware['returncode']==0 and report['passed'] and cpu['passed'] and candidate['passed']
    assert hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest()==candidate['candidate_sha256']
    sources=['optimizer.py','kernels_optimizer.py','kernels_semantics.py','kernels.py','runtime.py','reference.py','frontend.py','worker.py','replay.py','autograd.py']
    for name in sources:
        local=ROOT/'openassembler/modelfront'/name;digest=hashlib.sha256(local.read_bytes()).hexdigest()
        assert digest==report['source_hashes']['modelfront/'+name]
        assert digest==cpu['source_hashes'][local.relative_to(ROOT).as_posix()]
    assert hashlib.sha256((HERE/'hardware_gate.py').read_bytes()).hexdigest()==hardware['harness_sha256']['optimizer_gate.py']
    archive=path.parent/hardware['archive']['path'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==hardware['archive']['sha256']
    bundles=[]
    with zipfile.ZipFile(archive) as z:
        assert 'work/full-tensors.pt' in z.namelist()
        for name in z.namelist():
            if not name.endswith('/native-ledger.json') or not (name.startswith('work/optimizer-options/') or name.startswith('cli-')):continue
            prefix=name.removesuffix('native-ledger.json');ledger=json.loads(z.read(name));plan=json.loads(z.read(prefix+'plan.json'))
            assert hashlib.sha256(canonical(plan)).hexdigest()==ledger['plan_sha256']
            assert hashlib.sha256(z.read(prefix+ledger['cubin'])).hexdigest()==ledger['cubin_sha256']
            assert hashlib.sha256(z.read(prefix+'sources/native.ptx')).hexdigest()==ledger['source_sha256']
            bundles.append(prefix)
    assert len(bundles)==14
    plain=json.loads((HERE/'plain-sgd-before.json').read_text())
    for shape in ([],[3],[129],[2,7]):
        kernel=emit_kernel('sgd',[{'shape':shape}]*2,{'shape':shape})
        assert plain[str(shape)]=={'name':kernel.name,'ptx_sha256':hashlib.sha256(kernel.ptx.encode()).hexdigest()}
    assert all(row['passed'] for row in report['comparisons'])
    cli={name:{key:len(value['checks']) for key,value in reports.items()} for name,reports in report['cli_reports'].items()}
    for reports in report['cli_reports'].values():
        for result in reports.values():
            assert result['passed'] and result['iterations']==6 and result['vendor_compute_launches']==0
            assert all(row['passed'] for row in result['checks'])
    result={'schema':1,'passed':True,'candidate_complete_updates':len(candidate['rows']),
            'production_sources_match_hardware_and_cpu':sources,'bundles_checked':len(bundles),
            'plain_sgd_kernel_names_and_ptx_hashes_unchanged':True,'plain_sgd_contracts_checked':4,
            'hardware_comparisons':len(report['comparisons']),'hardware_elements':sum(r['elements'] for r in report['comparisons']),
            'complete_update_records':sum('step' in r for r in report['runs']),
            'profiler_audits':sum('profile' in r for r in report['runs']),'cli_checks':cli,
            'hardware_report':str(path.relative_to(ROOT)),'current_host_source_revalidation':True,'hardware_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
