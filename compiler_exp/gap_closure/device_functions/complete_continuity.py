"""Complete prior-native replay with the preceding Python-source archive format."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.container.cubin import text_words,regcount

def sha(data):return hashlib.sha256(data).hexdigest()
if __name__=='__main__':
    base_path=HERE/'base-application-continuity.json';base=json.loads(base_path.read_text());assert base['passed']and base['native_comparison_count']==1598
    assert base['auditor_sha256']==sha((HERE/'base-continuity.py').read_bytes())
    assert all(sha((ROOT/name).read_bytes())==digest for name,digest in base['backend_hashes'].items())
    validation_path=HERE.parent/'python_shared_wide/validation.json';validation=json.loads(validation_path.read_text());assert validation['passed']
    proof_path=HERE.parent/'python_shared_wide/promotion-audit.json';proof=json.loads(proof_path.read_text());assert proof['passed']and proof['audited_backend']=='production'
    report_path=ROOT/proof['report'];report=json.loads(report_path.read_text());assert report['passed']
    assert sha(proof_path.read_bytes())==validation['evidence_sha256'][proof_path.name]and sha(report_path.read_bytes())==validation['evidence_sha256'][report_path.name]
    assert proof['report_sha256']==sha(report_path.read_bytes())
    archive_path=report_path.parent/report['archive']['path'];assert sha(archive_path.read_bytes())==report['archive']['sha256']
    rows=[]
    with zipfile.ZipFile(archive_path)as archive:
        module=parse_module(archive.read('generated.ptx').decode());assert len(module.kernels)==20
        for level in('safe','mined','sched','opt'):
            binary=archive.read(level+'/oa.cubin')
            for name,kernel in module.kernels.items():
                _,_,words,_,registers=build.generate(kernel,'sm_80',level)
                assert words==text_words(binary,name)and registers==regcount(binary,name)
                rows.append(dict(source=str(report_path.relative_to(ROOT)),entry=name,level=level,registers=registers,passed=True))
    assert len(rows)==80
    result=dict(base)
    result.update(base_audit=str(base_path.relative_to(ROOT)),base_audit_sha256=sha(base_path.read_bytes()),
                  scope='Prior source-exact numerical/ownership chain plus complete current native words/registers, including twenty preceding Python shared-source kernels whose archive uses generated.ptx; direct scalar function expansion has its own original-source A100 gate',
                  supplemental_auditor_sha256=sha(Path(__file__).read_bytes()),supplemental_native_comparisons=rows,
                  auditor_sha256=sha(Path(__file__).read_bytes()),native_comparisons=base['native_comparisons']+rows,native_comparison_count=1678)
    result['retained_hashes'].update({str(p.relative_to(ROOT)):sha(p.read_bytes())for p in(proof_path,report_path,archive_path,validation_path)})
    (HERE/'application-continuity.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed',len(result['native_comparisons']),'prior native occurrences')
