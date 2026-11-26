"""Bind wide-selection promotion to exact independent evidence and full integration."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 obj=json.loads((HERE/name).read_text());assert obj['passed'],name;return obj
def suite(path):
 text=path.read_text();m=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text)
 assert m and 'FAILED 'not in text,path
 return dict(passed=int(m[1]),skipped=int(m[2]or 0),failed=0,seconds=float(m[3]),log=path.name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 baseline=HERE.parents[1]/'wide_select/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
 assert changed=={'openassembler/backend/ra.py','openassembler/backend/contracts.py','openassembler/pyfront/compile.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 for source,name in(('openassembler/backend/ra.py','ra'),('openassembler/backend/contracts.py','contracts'),('openassembler/pyfront/compile.py','pyfront')):
  assert sha(HERE/('baseline-'+name+'.py'))==previous['source_hashes'][source]
  assert(ROOT/source).read_bytes()==(HERE/('candidate-'+name+'.py')).read_bytes()
 changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest};assert changed_tests=={'tests/test_pyfront_wide.py'},changed_tests
 old=(HERE/'baseline-test-pyfront-wide.py').read_text();new=old.replace("('x * x','64-bit multiplication')","('x ** i64(2)','operator Pow')");assert(ROOT/'tests/test_pyfront_wide.py').read_text()==new and sha(HERE/'baseline-test-pyfront-wide.py')==previous['source_hashes']['tests/test_pyfront_wide.py']
 for source,target in(('test_backend_wide_multiply.py','tests/test_backend_wide_multiply.py'),('wide_multiply_fixtures.json','tests/data/wide_multiply_fixtures.json')):assert(HERE/source).read_bytes()==(ROOT/target).read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==1560 and proof['bytes_checked']==15974400 and len(proof['native_comparisons'])==156
 assert proof['auditor_sha256']==sha(HERE/'combine_audits.py')
 for audit_name,script in(('production-base-audit.json','audit.py'),('production-squares-audit.json','audit_squares.py')):
  audit=checked(audit_name);assert audit['audited_backend']=='production'and audit['auditor_sha256']==sha(HERE/script)and sha(HERE/audit_name)==proof['individual_audits_sha256'][audit_name]
  assert sha(ROOT/audit['report'])==audit['report_sha256']
 squares=checked('candidate-squares-a100-20261123T213402Z.json');assert squares['harness_sha256']==sha(HERE/'run_squares_modal.py')and len(squares['results'])==160
 assert all(sha(ROOT/name)==digest for name,digest in squares['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 report=checked('candidate-a100-20261123T212929Z.json');assert report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 for key in('fixtures.py','python_fixtures.py','cuda.cu'):assert report['source_hashes']['compiler_exp/gap_closure/wide_multiply/promotion/'+key]==sha(HERE/key)
 offline=checked('candidate-offline.json');assert len(offline['results'])==140 and sum(r['executions']for r in offline['results'])==2520
 assert all(sha(HERE/name)==digest for name,digest in offline['source_hashes'].items())
 rejected=checked('baseline.json');assert len(rejected['results'])==35 and all(not r['passed']and ('unsupported PTX contract mul.lo.'in r['error']or'64-bit multiplication is not supported'in r['error'])for r in rejected['results'])
 square_offline=checked('candidate-squares-offline.json');assert len(square_offline['results'])==16 and sum(r['executions']for r in square_offline['results'])==288
 assert all(sha(HERE/name)==digest for name,digest in square_offline['source_hashes'].items())
 rejected_squares=checked('baseline-squares.json');assert len(rejected_squares['results'])==4 and all(not r['passed']and'64-bit multiplication is not supported'in r['error']for r in rejected_squares['results'])
 oracle=checked('oracle-audit.json');assert len(oracle['artifact_hashes'])==157 and oracle['script_sha256']==sha(HERE/'audit_oracle.py')and oracle['report_sha256']==sha(HERE/'oracle-probe.json')
 assert all(sha(HERE/name)==digest for name,digest in oracle['artifact_hashes'].items())
 square_oracle=checked('oracle-squares-audit.json');assert len(square_oracle['artifact_hashes'])==20 and square_oracle['script_sha256']==sha(HERE/'audit_oracle_squares.py')and square_oracle['report_sha256']==sha(HERE/'oracle-squares-probe.json')
 assert all(sha(HERE/name)==digest for name,digest in square_oracle['artifact_hashes'].items())
 frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==15 and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')and frontend['current_frontend_sha256']==sha(ROOT/'openassembler/pyfront/compile.py')
 assert all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==156 and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and wheel['script_sha256']==sha(HERE/'audit_wheel.py')
 assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==1934 and continuity['auditor_sha256']==sha(HERE/'continuity.py')
 assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T213840Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 supported=suite(HERE/'supported-suite.log');assert supported['passed']==3465 and supported['skipped']==0
 supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite(HERE/'compiler-only-suite.log');assert core['passed']==2980 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite(HERE/'targeted-suite.log')['passed']==140
 docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 evidence=('baseline.json','baseline-squares.json','candidate-offline.json','candidate-squares-offline.json','oracle-probe.json','oracle-squares-probe.json','oracle-audit.json','oracle-squares-audit.json','candidate-a100-20261123T212929Z.json','candidate-squares-a100-20261123T213402Z.json','candidate-audit.json','candidate-squares-audit.json','production-base-audit.json','production-squares-audit.json','promotion-audit.json','wheel-audit.json','application-continuity.json','frontend-continuity.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G03/G11 low64 products from recovered scalar IMAD words and typed Python i64/u64 multiplication, exact snapshots/guards/aliases/parameters/immediates/helpers/loops/casts/shared-operand squares with original CUDA and ptxas controls; general Python/scalar semantics and original 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),added_production_inputs=[],added_test_inputs=['tests/test_backend_wide_multiply.py','tests/data/wide_multiply_fixtures.json'],changed_existing_test_inputs=sorted(changed_tests),evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=1560,bytes_audited=15974400,native_comparisons=156,prior_application_native_comparisons=1934,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
