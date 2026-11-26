"""Bind exact Boolean storage production to original-source/full integration evidence."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def checked(name):
 d=json.loads((HERE/name).read_text());assert d['passed'],name;return d
def suite(name,count,skips):
 text=(HERE/name).read_text();m=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert m and'FAILED 'not in text,name
 assert int(m[1])==count and int(m[2]or 0)==skips,(name,m.groups());return dict(passed=count,skipped=skips,failed=0,seconds=float(m[3]),log=name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 previous_path=HERE.parent/'scalar/validation.json';previous=json.loads(previous_path.read_text());assert previous['passed']
 staged=json.loads((HERE/'inputs.json').read_text());policy_names={'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'}
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest};assert changed==set(staged.values())|policy_names,changed
 for suffix,name in staged.items():assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]and(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 promotion=checked('promotion.json');updates=checked('test-contract-updates.json');assert len(updates['updates'])==8 and len(updates['results'])==16
 changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest};assert changed_tests=={row['file']for row in updates['files']}and len(changed_tests)==3
 for row in updates['files']:assert previous['source_hashes'][row['file']]==row['baseline_sha256']and sha(ROOT/row['file'])==row['current_sha256']
 added={'tests/test_pyfront_boolean_storage.py':'test_pyfront_boolean_storage.py','tests/test_pyfront_boolean_storage_contracts.py':'test_storage_contracts.py','tests/data/boolean_storage_fixtures.json':'boolean_storage_fixtures.json'}
 for name,source in added.items():assert name not in previous['source_hashes']and(ROOT/name).read_bytes()==(HERE/source).read_bytes()
 for field in('changed_production_inputs','changed_test_inputs','added_test_inputs'):
  assert all(sha(ROOT/name)==digest for name,digest in promotion[field].items())
 assert all(sha(HERE/name)==digest for name,digest in promotion['evidence_sha256'].items())
 for report_name,proof_names,count,explicit,byte_count,native_count,harness,auditor in [
 ('candidate-a100-20261126T065029Z.json',('candidate-audit.json','promotion-audit.json'),4320,432,18247680,216,'run_modal.py','audit.py'),
 ('shapes-a100-20261126T065356Z.json',('shapes-audit.json','promotion-shapes-audit.json'),960,96,4055040,48,'run_modal_shapes.py','audit_shapes.py')]:
  report=checked(report_name);assert report['error']is None and len(report['results'])==count and len(report['explicit_launch_results'])==explicit and all(row['passed']for row in report['results']+report['explicit_launch_results'])
  assert sha(HERE/report['archive']['path'])==report['archive']['sha256']and report['harness_sha256']==sha(HERE/harness)
  assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in policy_names)
  for proof_name in proof_names:
   proof=checked(proof_name);assert proof['report_sha256']==sha(HERE/report_name)and proof['auditor_sha256']==sha(HERE/auditor)
   assert proof['launches']==count and proof['explicit_launches']==explicit and proof['bytes_checked']==byte_count and len(proof['native_comparisons'])==native_count and proof['corrupted_output_rejected']
  assert checked(proof_names[1])['audited_backend']=='production'
 for name,rows,executions in [('candidate-offline.json',216,10368),('shapes-offline.json',48,2304)]:
  d=checked(name);assert len(d['results'])==rows and d['executions']==executions and all(sha(HERE/name)==digest for name,digest in d['source_hashes'].items())
 baseline=checked('baseline-replay.json');assert len(baseline['results'])==54 and all(row['rejected']for row in baseline['results'])and all(sha(HERE/name)==digest for name,digest in baseline['source_hashes'].items())
 chain=checked('application-continuity.json');assert chain['native_comparison_count']==chain['native_exact_count']==len(chain['native_comparisons'])==4214 and chain['stall_yield_changed_count']==0
 assert all(row['passed']and row['native_exact']for row in chain['native_comparisons'])and chain['previous_validation_sha256']==sha(previous_path)and chain['auditor_sha256']==sha(HERE/'continuity.py')
 assert all(sha(ROOT/name)==digest for field in('backend_hashes','candidate_source_hashes','retained_hashes')for name,digest in chain[field].items())
 frontend=checked('frontend-continuity.json');assert len(frontend['source_hashes'])==21 and frontend['kernels']==62 and not frontend['changed_source_outputs']and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')and all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')and remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json')and len(remote['logical_contracts'])==18
 for filename,count,report,script in [('wheel-audit.json',216,'candidate-a100-20261126T065029Z.json','audit_wheel.py'),('wheel-shapes-audit.json',48,'shapes-a100-20261126T065356Z.json','audit_wheel_shapes.py')]:
  wheel=checked(filename);assert len(wheel['source_hashes'])==132 and len(wheel['native_comparisons'])==count and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and wheel['report_sha256']==sha(HERE/report)and wheel['script_sha256']==sha(HERE/script)
  assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 tables=[json.loads((ROOT/name).read_text())for name in policy_names];policy_path=ROOT/tables[0]['source_continuity']['audit'];policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert policy['baseline_commit']=='4335157dcf30a9576718d2d008e16c8786548368'and policy['audit_script_sha256']==sha(ROOT/'compiler_exp/gap_closure/foundations_20261121/continuity.py')
 assert all(table['source_continuity']['sha256']==sha(policy_path)for table in tables)and all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 supported=suite('supported-suite.log',5136,0);supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform());core=suite('compiler-only-suite.log',4651,15);core.update(python='3.14.5',torch='not installed');suite('candidate-tests.log',208,0)
 docs={name:'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 evidence=['baseline.json','baseline-replay.json','layout-baseline.json','candidate-offline.json','shapes-offline.json','candidate-a100-20261126T065029Z.json','shapes-a100-20261126T065356Z.json','candidate-audit.json','shapes-audit.json','promotion-audit.json','promotion-shapes-audit.json','promotion.json','test-contract-updates.json','application-continuity.json','frontend-continuity.json','normal-remote-audit.json','wheel-audit.json','wheel-shapes-audit.json','boolean_storage_fixtures.json']
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-26',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G03/G07/G09/G10/G12 Boolean one-byte raw/shaped/shared storage, source/host/metadata/native custody and independent original-source/numerical/A100/public/installed/integration gates; direct predicate comparisons, Boolean atomics/arithmetic, arbitrary memory safety/Python and all 51 broad gaps remain open',supported_suite=supported,compiler_only_suite=core,foundation_record=str(previous_path.relative_to(ROOT)),foundation_sha256=sha(previous_path),changed_production_inputs=sorted(changed),changed_existing_test_inputs=sorted(changed_tests),added_test_inputs=sorted(added),source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},evidence_sha256={name:sha(HERE/name)for name in evidence},logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'wheel'not in p.parts and'normal-remote-build'not in p.parts},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=5280,explicit_launches=528,bytes_audited=22302720,native_comparisons=264,prior_application_native_comparisons=4214,prior_native_exact=4214,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
