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
    baseline=HERE.parent/'shared_widths/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/pyfront/compile.py','openassembler/backend/compile.py','openassembler/backend/ra.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    candidates={'openassembler/pyfront/compile.py':('candidate-compile.py','baseline-compile.py'),
                'openassembler/backend/compile.py':('candidate-backend-compile.py','baseline-backend-compile.py'),
                'openassembler/backend/ra.py':('candidate-ra.py','baseline-ra.py')}
    for name,(candidate,old)in candidates.items():
        assert sha(HERE/old)==previous['source_hashes'][name]
        assert(ROOT/name).read_bytes()==(HERE/candidate).read_bytes()
    changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest}
    assert changed_tests=={'tests/test_pyfront_wide.py'},changed_tests
    assert sha(HERE/'baseline-test_pyfront_wide.py')==previous['source_hashes']['tests/test_pyfront_wide.py']
    for name in('test_pyfront_shared_wide.py','data/pyfront_shared_wide.json'):
        assert(ROOT/'tests'/name).read_bytes()==(HERE/name).read_bytes()
    proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==1280
    assert len(proof['native_comparisons'])==80 and proof['bytes_checked']==20971520
    assert proof['auditor_sha256']==sha(HERE/'audit.py')
    report=checked('candidate-a100-20261123T162930Z.json');assert proof['report_sha256']==sha(HERE/'candidate-a100-20261123T162930Z.json')
    assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
    assert report['harness_sha256']==sha(HERE/'run_modal.py')
    continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==1598
    assert continuity['auditor_sha256']==sha(HERE/'continuity.py')
    assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
    offline=checked('candidate-offline.json');assert len(offline['results'])==80 and sum(r['executions']for r in offline['results'])==1440
    assert offline['frontend_sha256']==sha(HERE/'candidate-compile.py')and offline['fixtures_sha256']==sha(HERE/'fixtures.py')
    assert all(sha(ROOT/name)==digest for name,digest in offline['backend_hashes'].items())
    oracle=checked('oracle-audit.json');assert len(oracle['artifact_hashes'])==60
    assert oracle['script_sha256']==sha(HERE/'audit_oracle.py')and oracle['report_sha256']==sha(HERE/'oracle-probe.json')
    assert all(sha(HERE/name)==digest for name,digest in oracle['artifact_hashes'].items())
    encoding=checked('convergence-encoding.json');assert len(encoding['results'])==56
    assert encoding['script_sha256']==sha(HERE/'convergence_encoding.py')
    assert encoding['corpus_sha256']==sha(ROOT/'corpus/sm_80/pairs.jsonl')and encoding['encoder_sha256']==sha(ROOT/'openassembler/isa/encode.py')
    failed=json.loads((HERE/'failure-isolation-audit.json').read_text());assert failed['audit_completed']and len(failed['results'])==6
    assert sum(r['bytes_checked']for r in failed['results'])==75497472
    assert sum(len(r['failures'])for r in failed['results'])==192
    assert failed['script_sha256']==sha(HERE/'audit_failures.py')
    for row in failed['results']:assert row['report_sha256']==sha(HERE/row['report'])
    wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==80 and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])
    assert wheel['script_sha256']==sha(HERE/'audit_wheel.py')
    assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
    frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['results'])==14
    assert frontend['current_frontend_sha256']==sha(ROOT/'openassembler/pyfront/compile.py')and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')
    assert all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
    policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T164034Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']
    assert policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
    assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
    supported=suite(HERE/'supported-suite.log');assert supported['passed']==3176 and supported['skipped']==0
    supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');assert core['passed']==2691 and core['skipped']==15
    core.update(python='3.14.5',torch='not installed')
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    evidence=('baseline.json','oracle-probe.json','candidate-offline.json','convergence-encoding.json','candidate-a100-20261123T152334Z.json','candidate-a100-20261123T152514Z.json','candidate-a100-20261123T153328Z.json','candidate-a100-20261123T155732Z.json','candidate-a100-20261123T162930Z.json','candidate-audit.json','failure-isolation-audit.json','promotion-audit.json','application-continuity.json','frontend-continuity.json','wheel-audit.json','rowwise-policy-probe.json')
    evidence+=('oracle-audit.json',)
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
    result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='Typed Python i64/u64 shared arrays with natural alignment/padding and hardware-validated forward CTA-phase reconvergence; original 51-gap program remains incomplete',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),
                added_production_inputs=[],changed_existing_test_inputs=sorted(changed_tests),
                evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),
                # This invocation's stdout is still being written while this
                # record is constructed, so it cannot hash its own final log.
                logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},
                experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.txt')and'__pycache__'not in p.parts},
                docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(launches=1280,bytes_audited=20971520,native_comparisons=80,prior_application_native_comparisons=1598,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
