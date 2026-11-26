"""Bind final local-package helper production to independent complete integration gates."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 data=json.loads((HERE/name).read_text());assert data['passed'],name;return data
def suite(name):
 text=(HERE/name).read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert match and'FAILED 'not in text,name
 return dict(passed=int(match[1]),skipped=int(match[2]or 0),failed=0,seconds=float(match[3]),log=name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 foundation=HERE.parents[1]/'wide_multiply/promotion/validation.json';previous=json.loads(foundation.read_text());assert previous['passed']
 staged={'openassembler/pyfront/compile.py':'pyfront','openassembler/pyfront/helpers.py':'helpers','openassembler/build.py':'build','openassembler/frontend.py':'frontend','openassembler/native_ownership.py':'ownership','openassembler/remote_build.py':'remote','openassembler/cli.py':'cli'}
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
 assert changed==set(staged)|{'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 for name,suffix in staged.items():
  assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 assert(ROOT/'openassembler/pyfront/modules.py').read_bytes()==(HERE/'modules.py').read_bytes()
 changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest};assert not changed_tests,changed_tests
 for source,target in(('test_pyfront_module_helpers.py','tests/test_pyfront_module_helpers.py'),('module_helper_fixtures.json','tests/data/module_helper_fixtures.json')):assert(HERE/source).read_bytes()==(ROOT/target).read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==600 and proof['bytes_checked']==5529600 and len(proof['native_comparisons'])==60
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(HERE/'candidate-a100-20261123T224420Z.json')
 report=checked('candidate-a100-20261123T224420Z.json');assert report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 assert report['source_hashes']['compiler_exp/gap_closure/python_module_helpers/promotion/fixtures.py']==sha(HERE/'fixtures.py')
 offline=checked('candidate-offline.json');assert offline['executions']==1080 and len(offline['results'])==60 and all(sha(HERE/name)==digest for name,digest in offline['source_hashes'].items())
 negatives=checked('negative-controls.json');assert len(negatives['results'])==29 and negatives['script_sha256']==sha(HERE/'negative_controls.py')and negatives['module_sha256']==sha(HERE/'modules.py')
 oracle=checked('oracle-probe.json');assert len(oracle['results'])==5 and len(oracle['artifact_hashes'])==42 and oracle['script_sha256']==sha(HERE/'oracle_probe.py')
 assert all(sha(HERE/name)==digest for name,digest in oracle['artifact_hashes'].items())and all(sha(HERE/name)==digest for name,digest in oracle['source_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==60 and wheel['script_sha256']==sha(HERE/'audit_wheel.py')and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])
 assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==16 and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')
 assert frontend['current_frontend_sha256']==sha(ROOT/'openassembler/pyfront/compile.py')and all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==2090 and continuity['auditor_sha256']==sha(HERE/'rebind_final_adapters.py')
 assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
 base=ROOT/continuity['base_audit'];assert sha(base)==continuity['base_audit_sha256']and json.loads(base.read_text())['auditor_sha256']==sha(HERE/'continuity.py')
 assert continuity['frontend_non_doc_ast_equal']and continuity['build_ast_outside_entry_adapter_equal']and continuity['remote_ast_outside_client_target_discovery_equal']
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T224235Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')
 assert sha(HERE/'normal-remote-build/manifest.json')==remote['manifest_sha256']
 assert remote['contract']['entry']=='kernels/entry.py'and len(remote['contract']['files'])==4
 supported=suite('supported-suite.log');assert supported['passed']==3538 and supported['skipped']==0;supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite('compiler-only-suite.log');assert core['passed']==3053 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite('targeted-suite.log')['passed']==73
 for name,passed in(('first-supported-suite.log',3536),('first-compiler-only-suite.log',3051)):
  text=(HERE/name).read_text();assert'1 failed'in text and str(passed)+' passed'in text and'test_failed_frontend_retains_completed_phase_time'in text
 docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 evidence=('candidate-offline.json','negative-controls.json','oracle-probe.json','candidate-a100-20261123T221805Z.json','candidate-a100-20261123T222132Z.json','candidate-a100-20261123T222916Z.json','candidate-a100-20261123T224152Z.json','candidate-a100-20261123T224420Z.json','first-candidate-audit.json','pre-doc-promotion-audit.json','documented-promotion-audit.json','promotion-audit.json','final-candidate-audit.json','frontend-continuity.json','application-continuity.json','normal-remote-audit.json','wheel-audit.json','remote-probe.json')
 sources=sorted(path for base in('openassembler','tests')for path in(ROOT/base).rglob('*')if path.suffix in('.py','.json','.ptx')and'__pycache__'not in path.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
  scope='G04/G42 static local-package Python device helpers, defining-module scalar semantics, exact source/diagnostic/graph/ownership and CLI/local/remote build integration; arbitrary packages/runtime imports/captures/host effects and original 51-gap program remain incomplete',
  supported_suite=supported,compiler_only_suite=core,source_hashes={str(path.relative_to(ROOT)):sha(path)for path in sources},foundation_record=str(foundation.relative_to(ROOT)),foundation_sha256=sha(foundation),changed_production_inputs=sorted(changed),added_production_inputs=['openassembler/pyfront/modules.py'],changed_existing_test_inputs=[],added_test_inputs=['tests/test_pyfront_module_helpers.py','tests/data/module_helper_fixtures.json'],evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={path.name:sha(path)for path in HERE.glob('*.log')if path.name!='record-validation.log'},experiment_source_hashes={str(path.relative_to(HERE)):sha(path)for path in sorted(HERE.rglob('*'))if path.suffix in('.py','.ptx','.cu')and'__pycache__'not in path.parts and'normal-build'not in path.parts and'normal-remote-build'not in path.parts and'remote-output'not in path.parts and'wheel'not in path.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=600,bytes_audited=5529600,native_comparisons=60,prior_application_native_comparisons=2090,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
