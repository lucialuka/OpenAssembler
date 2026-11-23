"""Check native operator source, fixture and retained archive identities."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from openassembler.modelfront.kernels import canonical

if __name__=='__main__':
    path=HERE/'hardware-20261121T122827Z.json'
    hardware=json.loads(path.read_text());report=hardware['report']
    cpu=json.loads((HERE/'cpu-results.json').read_text())
    candidate=json.loads((HERE/'candidate-results.json').read_text())
    assert candidate['passed'] and cpu['passed'] and report['passed'] and hardware['returncode']==0
    assert hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest()==candidate['candidate_sha256']
    for name in ('graph.py','kernels_nn.py','reference.py'):
        local=ROOT/'openassembler/modelfront'/name
        digest=hashlib.sha256(local.read_bytes()).hexdigest()
        assert digest==report['source_hashes']['modelfront/'+name]
        assert digest==cpu['source_hashes'][local.relative_to(ROOT).as_posix()]
    for name,digest in hardware['harness_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    archive=HERE/hardware['archive']['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==hardware['archive']['sha256']
    bundles=[]
    with zipfile.ZipFile(archive) as z:
        assert 'work/full-tensors.pt' in z.namelist()
        for name in z.namelist():
            if not name.endswith('/native-ledger.json'):continue
            prefix=name.removesuffix('native-ledger.json');ledger=json.loads(z.read(name))
            plan=json.loads(z.read(prefix+'plan.json'))
            assert hashlib.sha256(canonical(plan)).hexdigest()==ledger['plan_sha256']
            assert hashlib.sha256(z.read(prefix+ledger['cubin'])).hexdigest()==ledger['cubin_sha256']
            assert hashlib.sha256(z.read(prefix+'sources/native.ptx')).hexdigest()==ledger['source_sha256']
            bundles.append(prefix)
        cli={name:json.loads(z.read('cli-output/'+name)) for name in ('model-build-report.json','model-run.json')}
        for result in cli.values():
            assert result['passed'] and result['iterations']==5 and result['vendor_compute_launches']==0
            assert all(row['passed'] for row in result['checks'])
    assert all(row['passed'] for row in report['comparisons'])
    result={'schema':1,'passed':True,'production_operator_sources_match_hardware_and_cpu':True,
            'candidate_harness_archive_identities_match':True,'bundles_checked':len(bundles),
            'hardware_comparisons':len(report['comparisons']),
            'hardware_elements':sum(r['elements'] for r in report['comparisons']),
            'guarded_runs':sum('iteration' in r for r in report['runs']),
            'profiler_audits':sum('profiler' in r for r in report['runs']),
            'cli_checks':{n:len(v['checks']) for n,v in cli.items()},
            'hardware_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
