"""Bind logical Boolean parameter promotion to exact original-source/integration evidence."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 d=json.loads((HERE/name).read_text());assert d['passed'],name;return d
def suite(name,count,skips):
 text=(HERE/name).read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert match and'FAILED 'not in text,name
 assert int(match[1])==count and int(match[2]or 0)==skips,(name,match.groups())
 return dict(passed=count,skipped=skips,failed=0,seconds=float(match[3]),log=name)

if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 previous_path=HERE.parents[1]/'narrow_parameters/validation.json';previous=json.loads(previous_path.read_text());assert previous['passed']
 staged=json.loads((HERE/'inputs.json').read_text());policy_names={'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'}
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest};assert changed==set(staged.values())|policy_names,changed
 for suffix,name in staged.items():
  assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 updates=checked('test-contract-updates.json');changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest}
 assert changed_tests=={row['file']for row in updates['files']}and len(changed_tests)==2
 for row in updates['files']:assert previous['source_hashes'][row['file']]==row['baseline_sha256']and sha(ROOT/row['file'])==row['current_sha256']
 for destination,source in [('tests/test_pyfront_boolean_parameters.py','test_pyfront_boolean_parameters.py'),('tests/data/boolean_parameter_fixtures.json','boolean_parameter_fixtures.json')]:assert destination not in previous['source_hashes']and(ROOT/destination).read_bytes()==(HERE/source).read_bytes()
 report_path=HERE/'candidate-a100-20261125T074133Z.json';report=checked(report_path.name);assert report['error']is None and len(report['results'])==9600 and len(report['explicit_launch_results'])==960 and all(row['passed']for row in report['results']+report['explicit_launch_results'])
 assert report['archive']['sha256']==sha(HERE/report['archive']['path'])and report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in policy_names)
 for name in('candidate-audit.json','promotion-audit.json'):
  proof=checked(name);assert proof['report_sha256']==sha(report_path)and proof['auditor_sha256']==sha(HERE/'audit.py')
  assert proof['launches']==9600 and proof['explicit_launches']==960 and proof['bytes_checked']==69746688 and len(proof['native_comparisons'])==240 and proof['corrupted_output_rejected']
 assert checked('promotion-audit.json')['audited_backend']=='production'
 promotion=checked('promotion.json');assert promotion['candidate_audit_sha256']==sha(HERE/'candidate-audit.json')
 for name,count in [('baseline-replay.json',60),('original-replay.json',3),('candidate-offline.json',240),('negative-controls.json',14)]:
  data=checked(name);assert len(data['results'])==count and all(sha(HERE/path)==digest for path,digest in data['source_hashes'].items())
 assert checked('original-replay.json')['original_record_sha256']==sha(HERE/'baseline.json')
 assert checked('candidate-offline.json')['executions']==17280
 assert all(row['rejected']for name in('baseline-replay.json','original-replay.json','negative-controls.json')for row in checked(name)['results'])
 chain=checked('application-continuity.json');assert chain['native_comparison_count']==chain['native_exact_count']==len(chain['native_comparisons'])==3974 and chain['stall_yield_changed_count']==0
 assert all(row['passed']and row['native_exact']for row in chain['native_comparisons'])and chain['previous_validation_sha256']==sha(previous_path)and chain['auditor_sha256']==sha(HERE/'continuity.py')
 assert all(sha(ROOT/name)==digest for field in('candidate_source_hashes','backend_hashes','retained_hashes')for name,digest in chain[field].items())
 frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==21 and not frontend['changed_source_outputs']and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')
 assert all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')and remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json')and len(remote['logical_contracts'])==20
 wheel=checked('wheel-audit.json');assert len(wheel['source_hashes'])==132 and len(wheel['native_comparisons'])==240 and wheel['report_sha256']==sha(report_path)and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and wheel['script_sha256']==sha(HERE/'audit_wheel.py')
 assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261125T074730Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 for name in policy_names:assert json.loads((ROOT/name).read_text())['source_continuity']['sha256']==sha(policy_path)
 supported=suite('supported-suite.log',4928,0);supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite('compiler-only-suite.log',4443,15);core.update(python='3.14.5',torch='not installed');suite('candidate-tests.log',164,0)
 docs={name:'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 properties=checked('control-properties.json');assert len(properties['results'])==512 and len({row['index']for row in properties['results']})==128 and properties['executions']==6144
 assert properties['script_sha256']==sha(HERE/'control_properties.py')and all(sha(HERE/name)==digest for name,digest in properties['source_hashes'].items())
 evidence=('baseline.json','baseline-replay.json','original-replay.json','candidate-offline.json','negative-controls.json','control-properties.json',report_path.name,'candidate-audit.json','promotion-audit.json','promotion.json','application-continuity.json','frontend-continuity.json','normal-remote-audit.json','wheel-audit.json','test-contract-updates.json','boolean_parameter_fixtures.json','first-proxy-app-state.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 data=dict(schema=1,date='2026-11-25',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
  scope='G03/G10 logical bool kernel parameters on canonical u8 launch slots, exact predicate decoding/type metadata/native custody, Boolean JSON role checking and typed verification including Boolean dimension names. Original source/A100/public-run/effect/installed/integration gates; Boolean storage, arbitrary unions/CUDA narrow scalar operations and full 51-gap program remain open',
  supported_suite=supported,compiler_only_suite=core,foundation_record=str(previous_path.relative_to(ROOT)),foundation_sha256=sha(previous_path),changed_production_inputs=sorted(changed),added_production_inputs=[],changed_existing_test_inputs=sorted(changed_tests),added_test_inputs=['tests/test_pyfront_boolean_parameters.py','tests/data/boolean_parameter_fixtures.json'],
  source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},evidence_sha256={name:sha(HERE/name)for name in evidence},logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'wheel'not in p.parts and'normal-remote-build'not in p.parts},
  policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),policy_continuity_auditor_sha256=sha(ROOT/'compiler_exp/gap_closure/foundations_20261121/continuity.py'),docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=9600,explicit_launches=960,bytes_audited=69746688,native_comparisons=240,prior_application_native_comparisons=3974,prior_native_exact=3974,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(data,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',data['docs']['status'])
