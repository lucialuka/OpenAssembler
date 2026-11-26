"""Bind exact promoted return paths to independent hardware and integration evidence."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 data=json.loads((HERE/name).read_text());assert data['passed'],name;return data
def suite(name):
 text=(HERE/name).read_text();m=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert m and 'FAILED 'not in text,name
 return dict(passed=int(m[1]),skipped=int(m[2]or 0),failed=0,seconds=float(m[3]),log=name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 previous_path=HERE.parents[1]/'python_comparison_chains/promotion/validation.json';previous=json.loads(previous_path.read_text());assert previous['passed']
 staged={'openassembler/backend/compile.py':'backend','openassembler/pyfront/compile.py':'pyfront','openassembler/pyfront/modules.py':'modules'}
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest};assert changed==set(staged)|{'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 for name,suffix in staged.items():
  assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 changed_tests={n for n,h in previous['source_hashes'].items()if n.startswith('tests/')and sha(ROOT/n)!=h};updates=json.loads((HERE/'test-contract-updates.json').read_text());assert changed_tests=={row['file']for row in updates['files']}
 for row in updates['files']:assert previous['source_hashes'][row['file']]==row['baseline_sha256']and sha(ROOT/row['file'])==row['current_sha256']
 assert(ROOT/'tests/test_pyfront_boolean_values.py').read_bytes()==(HERE/'production-test_pyfront_boolean_values.py').read_bytes()
 assert(ROOT/'tests/data/boolean_value_fixtures.json').read_bytes()==(HERE/'boolean_value_fixtures.json').read_bytes()
 assert(ROOT/'tests/test_native_predicate_paths.py').read_bytes()==(HERE/'test_native_predicate_paths.py').read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==2400 and proof['bytes_checked']==16957440 and len(proof['native_comparisons'])==240
 report_path=HERE/'candidate-a100-20261124T120418Z.json';report=checked(report_path.name)
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(report_path)and report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==h for name,h in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 offline=checked('candidate-offline.json');assert offline['executions']==4320 and len(offline['results'])==240 and all(sha(HERE/n)==h for n,h in offline['source_hashes'].items())
 baseline=checked('baseline.json');assert len(baseline['results'])==60 and all(row['rejected']for row in baseline['results'])and baseline['fixture_sha256']==sha(HERE/'fixtures.py')
 negative=checked('negative-controls.json');assert len(negative['results'])==16 and negative['script_sha256']==sha(HERE/'negative_controls.py')and all(sha(HERE/n)==h for n,h in negative['source_hashes'].items())
 properties=checked('control-properties.json');assert len(properties['results'])==128 and properties['executions']==1536 and all(sha(HERE/n)==h for n,h in properties['source_hashes'].items())
 oracle=checked('oracle-probe.json');assert len(oracle['results'])==3 and len(oracle['artifact_hashes'])==22 and oracle['script_sha256']==sha(HERE/'oracle_probe.py')
 assert all(sha(HERE/n)==h for n,h in oracle['source_hashes'].items())and all(sha(HERE/n)==h for n,h in oracle['artifact_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==240 and wheel['script_sha256']==sha(HERE/'audit_wheel.py')and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and all(sha(ROOT/n)==h for n,h in wheel['source_hashes'].items())
 context=checked('context-probe.json');assert context['executions']==60 and len(context['results'])==60 and all(sha(HERE/n)==h for n,h in context['source_hashes'].items())
 frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==20 and not frontend['changed_source_outputs']and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')and all(sha(ROOT/n)==h for n,h in frontend['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==3254 and len(continuity['native_comparisons'])==3254 and continuity['auditor_sha256']==sha(HERE/'continuity.py')
 assert continuity['previous_validation_sha256']==sha(previous_path)and continuity['validated_backend_report_sha256']==sha(report_path)and all(sha(ROOT/n)==h for n,h in continuity['backend_hashes'].items())
 assert continuity['native_exact_count']+continuity['stall_yield_changed_count']==3254 and all(row['passed']for row in continuity['native_comparisons'])
 transition=checked('failure-transition-audit.json');assert transition['original_failure_count']==54 and transition['repaired_launches']==2400 and transition['changed_control_outputs']==12 and transition['auditor_sha256']==sha(HERE/'audit_failure_transition.py')
 assert transition['repaired_report_sha256']==sha(report_path)and transition['original_failed_report_sha256']==sha(HERE/transition['original_failed_report'])
 for name,snapshot in(('candidate-a100-20261124T115739Z.json','baseline-backend.py'),('candidate-a100-20261124T120206Z.json','first-cfg-backend.py')):
  failure=json.loads((HERE/name).read_text());assert failure['passed']is False and failure['error']is None and len(failure['results'])==2400 and sum(not row['passed']for row in failure['results'])==54
  assert failure['source_hashes']['openassembler/backend/compile.py']==sha(HERE/snapshot)
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')and remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json')and remote['contract']['boolean_values']
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261124T121203Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)and all(sha(ROOT/n)==h for n,h in policy['compiler_inputs_sha256'].items())
 supported=suite('supported-suite.log');assert supported['passed']==4438 and supported['skipped']==0;supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite('compiler-only-suite.log');assert core['passed']==3953 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite('targeted-suite.log')['passed']==1246
 docs={n:(HERE/n).is_file()and'Documentation check passed:'in(HERE/n).read_text()for n in('docs-check.log','exported-docs-check.log')}
 evidence=('baseline.json','candidate-offline.json','negative-controls.json','context-probe.json','control-properties.json','oracle-probe.json','candidate-a100-20261124T120418Z.json','candidate-audit.json','promotion-audit.json','frontend-continuity.json','application-continuity.json','normal-remote-audit.json','wheel-audit.json','failure-transition-audit.json','test-contract-updates.json','candidate-a100-20261124T115739Z.json','candidate-a100-20261124T120206Z.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G02/G06 original fixed-kind Boolean values, operand-returning lazy scalar and/or, exact literal contexts/conversions/special-value/effect/source ownership and independently hardware-repaired native predicate readiness across CFG edges; Boolean ABI/type unions/dynamic steps/general Python effects and full 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,foundation_record=str(previous_path.relative_to(ROOT)),foundation_sha256=sha(previous_path),changed_production_inputs=sorted(changed),added_production_inputs=[],changed_existing_test_inputs=sorted(changed_tests),added_test_inputs=['tests/test_pyfront_boolean_values.py','tests/data/boolean_value_fixtures.json','tests/test_native_predicate_paths.py'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},evidence_sha256={n:sha(HERE/n)for n in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'normal-remote-build'not in p.parts and'wheel'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=2400,bytes_audited=16957440,native_comparisons=240,prior_application_native_comparisons=3254,prior_native_exact=continuity['native_exact_count'],prior_native_stall_yield_only=continuity['stall_yield_changed_count'],gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
