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
 previous_path=HERE.parents[1]/'python_helper_returns/promotion/validation.json';previous=json.loads(previous_path.read_text());assert previous['passed']
 staged={'openassembler/pyfront/compile.py':'pyfront','openassembler/pyfront/return_flow.py':'return_flow'}
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest};assert changed==set(staged),changed
 for name,suffix in staged.items():
  assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 updates=json.loads((HERE/'test-contract-updates.json').read_text());assert len(updates['files'])==3
 changed_tests={n for n,h in previous['source_hashes'].items()if n.startswith('tests/')and sha(ROOT/n)!=h};assert changed_tests=={r['file']for r in updates['files']}
 for row in updates['files']:assert previous['source_hashes'][row['file']]==row['baseline_sha256']and sha(ROOT/row['file'])==row['current_sha256']
 assert(ROOT/'tests/test_pyfront_range_control.py').read_bytes()==(HERE/'production-test_pyfront_range_control.py').read_bytes()
 assert(ROOT/'tests/data/range_control_fixtures.json').read_bytes()==(HERE/'range_control_fixtures.json').read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==2400 and proof['bytes_checked']==17203200 and len(proof['native_comparisons'])==240
 report_path=HERE/'candidate-a100-20261124T111006Z.json';report=checked(report_path.name)
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(report_path)and report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==h for name,h in report['source_hashes'].items()if name.startswith(('openassembler/','specs/')))
 offline=checked('candidate-offline.json');assert offline['executions']==4320 and len(offline['results'])==240 and all(sha(HERE/n)==h for n,h in offline['source_hashes'].items())
 baseline=checked('baseline.json');assert len(baseline['results'])==60 and baseline['accepted_not_executed']==24 and baseline['rejected']==36 and sum(row['rejected'] for row in baseline['results'])==36 and baseline['fixture_sha256']==sha(HERE/'fixtures.py')
 negative=checked('negative-controls.json');assert len(negative['results'])==16 and negative['script_sha256']==sha(HERE/'negative_controls.py')and all(sha(HERE/n)==h for n,h in negative['source_hashes'].items())
 properties=checked('control-properties.json');assert len(properties['results'])==128 and properties['executions']==1536 and all(sha(HERE/n)==h for n,h in properties['source_hashes'].items())
 oracle=checked('oracle-probe.json');assert len(oracle['results'])==3 and len(oracle['artifact_hashes'])==22 and oracle['script_sha256']==sha(HERE/'oracle_probe.py')
 assert all(sha(HERE/n)==h for n,h in oracle['source_hashes'].items())and all(sha(HERE/n)==h for n,h in oracle['artifact_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==240 and wheel['script_sha256']==sha(HERE/'audit_wheel.py')and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and all(sha(ROOT/n)==h for n,h in wheel['source_hashes'].items())
 frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==18 and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')and all(sha(ROOT/n)==h for n,h in frontend['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==2774 and continuity['inherited_native_comparison_count']==2534 and continuity['replayed_native_comparison_count']==240 and continuity['auditor_sha256']==sha(HERE/'continuity.py')
 assert continuity['previous_validation_sha256']==sha(previous_path)and continuity['validated_backend_report_sha256']==sha(report_path)and all(sha(ROOT/n)==h for n,h in continuity['unchanged_backend_hashes'].items())
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')and remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json')and remote['contract']['helpers']
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261124T052003Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)and all(sha(ROOT/n)==h for n,h in policy['compiler_inputs_sha256'].items())
 research_names=('baseline.json','baseline-replay.json','candidate-offline.json','candidate-audit.json','candidate-a100-20261124T060602Z.json','minimized.json','minimized.ptx','minimized-audit.json','minimized-a100-20261124T061630Z.json','manual/source.json','manual/ptx-add-excerpt.txt')
 research={str((HERE.parent/name).relative_to(ROOT)):sha(HERE.parent/name)for name in research_names}
 minimized=json.loads((HERE.parent/'minimized-audit.json').read_text());assert minimized['passed']and minimized['observations']==40 and minimized['bytes_checked']==6080
 assert minimized['auditor_sha256']==sha(HERE.parent/'audit_minimized.py')and minimized['report_sha256']==sha(HERE.parent/minimized['report'])
 minimized_report=json.loads((HERE.parent/minimized['report']).read_text());assert minimized_report['harness_sha256']==sha(HERE.parent/'run_minimized.py')and minimized_report['controls_sha256']==sha(HERE.parent/'minimized.json')
 replay=json.loads((HERE.parent/'baseline-replay.json').read_text());assert replay['passed']and replay['executions']==72 and replay['baseline_sha256']==sha(HERE.parent/'baseline.json')and replay['script_sha256']==sha(HERE.parent/'replay_baseline.py')
 supported=suite('supported-suite.log');assert supported['passed']==4112 and supported['skipped']==0;supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite('compiler-only-suite.log');assert core['passed']==3627 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite('targeted-suite.log')['passed']==920
 docs={n:(HERE/n).is_file()and'Documentation check passed:'in(HERE/n).read_text()for n in('docs-check.log','exported-docs-check.log')}
 evidence=('baseline.json','candidate-offline.json','negative-controls.json','control-properties.json','oracle-probe.json','candidate-a100-20261124T111006Z.json','candidate-audit.json','promotion-audit.json','frontend-continuity.json','application-continuity.json','normal-remote-audit.json','wheel-audit.json','test-contract-updates.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G06 finite i32 range exhaustion without counter wrap and original runtime range/while else with nearest-loop exits, conservative live-path initialization, convergence and source/native ownership; dynamic steps/wide bounds/arbitrary iterables/objects/effects/ABI and full 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,foundation_record=str(previous_path.relative_to(ROOT)),foundation_sha256=sha(previous_path),changed_production_inputs=sorted(changed),added_production_inputs=[],changed_existing_test_inputs=sorted(changed_tests),added_test_inputs=['tests/test_pyfront_range_control.py','tests/data/range_control_fixtures.json'],source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},evidence_sha256={n:sha(HERE/n)for n in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'normal-remote-build'not in p.parts and'wheel'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=2400,bytes_audited=17203200,native_comparisons=240,prior_application_native_occurrences=2774,prior_native_directly_regenerated=240,prior_native_inherited_with_unchanged_backend=2534,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 result['research_evidence_sha256']=research
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
