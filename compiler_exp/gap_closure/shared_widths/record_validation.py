"""Bind the shared scalar-width promotion to its unchanged prior integration."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def checked(name):
    obj=json.loads((HERE/name).read_text());assert obj['passed'],name;return obj

def suite(path):
    text=path.read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text)
    assert match and 'FAILED 'not in text,path
    return dict(passed=int(match[1]),skipped=int(match[2]or 0),failed=0,seconds=float(match[3]),log=path.name)

if __name__=='__main__':
    import torch
    assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
    baseline=HERE.parent/'constant_expressions/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/backend/ra.py','openassembler/backend/contracts.py','openassembler/backend/compile.py','openassembler/isa/encode.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    candidates={'ra':'openassembler/backend/ra.py','contracts':'openassembler/backend/contracts.py','compile':'openassembler/backend/compile.py','encode':'openassembler/isa/encode.py'}
    for name,path in candidates.items():
        assert sha(HERE/f'baseline-{name}.py')==previous['source_hashes'][path]
        assert(ROOT/path).read_bytes()==(HERE/f'candidate-{name}.py').read_bytes()
    data_path='specs/sm_80/memory_templates.json'
    assert(ROOT/data_path).read_bytes()==(HERE/'candidate-templates.json').read_bytes()
    changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest}
    assert changed_tests=={'tests/test_backend_memory_widths.py'},changed_tests
    assert sha(HERE/'baseline-test_backend_memory_widths.py')==previous['source_hashes']['tests/test_backend_memory_widths.py']
    for name in('test_backend_shared_widths.py','data/shared_width_fixtures.json'):
        assert(ROOT/'tests'/name).read_bytes()==(HERE/name).read_bytes()
    proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==1216
    assert len(proof['native_comparisons'])==76 and proof['bytes_checked']==34865152
    assert proof['auditor_sha256']==sha(HERE/'audit.py')
    report=checked('candidate-a100-20261123T144327Z.json');assert proof['report_sha256']==sha(HERE/'candidate-a100-20261123T144327Z.json')
    assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
    assert report['harness_sha256']==sha(HERE/'run_modal.py')
    continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==1522
    assert continuity['auditor_sha256']==sha(HERE/'continuity.py')
    assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
    offline=checked('candidate-offline.json');assert len(offline['results'])==76 and sum(r['executions']for r in offline['results'])==1368
    assert all(sha(HERE/name)==digest for name,digest in offline['source_hashes'].items())
    templates=checked('template-audit.json');assert len(templates['templates'])==4 and templates['field_sweep_words']==12096
    assert templates['script_sha256']==sha(HERE/'audit_templates.py')
    assert all(sha(HERE/name)==digest for name,digest in templates['oracle_artifacts'].items())
    assert templates['template_sha256']==sha(HERE/'candidate-templates.json')and templates['field_sweep_sha256']==sha(HERE/'field-sweep.json')
    matrix=checked('width-matrix.json');assert len(matrix['results'])==336
    assert all(sha(ROOT/name)==digest for name,digest in matrix['source_hashes'].items())
    assert matrix['fixtures_sha256']==sha(HERE/'width-matrix-fixtures.json')
    matrix_audit=checked('width-matrix-audit.json');assert matrix_audit['rows']==336 and matrix_audit['native_comparisons']==172
    assert matrix_audit['matrix_sha256']==sha(HERE/'width-matrix.json')
    assert matrix_audit['script_sha256']==sha(HERE/'audit_width_matrix.py')
    failed=json.loads((HERE/'failed-execution-audit.json').read_text());assert not failed['passed']and failed['audit_completed']and failed['failed_oa_launches']==90 and failed['failed_ptxas_launches']==0
    assert failed['report_sha256']==sha(HERE/'candidate-a100-20261123T143647Z.json')
    wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==76 and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])
    assert wheel['script_sha256']==sha(HERE/'audit_wheel.py')
    assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
    policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T144838Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']
    assert policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
    assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
    supported=suite(HERE/'supported-suite.log');assert supported['passed']==3099 and supported['skipped']==0
    supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');assert core['passed']==2614 and core['skipped']==15
    core.update(python='3.14.5',torch='not installed')
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    evidence=('oracle-probe.json','extra-type-oracle.json','template-recovery.json','field-sweep.json','template-audit.json','width-matrix.json','candidate-offline.json','candidate-a100-20261123T143055Z.json','negative-syntax-oracle.json','candidate-a100-20261123T143647Z.json','failed-execution-audit.json','candidate-a100-20261123T144011Z.json','candidate-a100-20261123T144327Z.json','candidate-audit.json','promotion-audit.json','application-continuity.json','wheel-audit.json')
    evidence+=('width-matrix-audit.json',)
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/data_path]
    result=dict(schema=1,date='2026-11-23',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='G12 bounded scalar narrow/64-bit shared memory, exact recovered store templates and hardware-validated barrier issue gap; original 51-gap program remains incomplete',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),
                added_production_inputs=[data_path],changed_existing_test_inputs=sorted(changed_tests),
                evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),
                # This invocation's stdout is still being written while this
                # record is constructed, so it cannot hash its own final log.
                logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},
                experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.txt')and'__pycache__'not in p.parts},
                docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(launches=1216,bytes_audited=34865152,native_comparisons=76,prior_application_native_comparisons=1522,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
