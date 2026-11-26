"""Bind original static declarations to source-exact numerical and integration proofs."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 d=json.loads((HERE/name).read_text());assert d['passed'],name;return d
def suite(name):
 text=(HERE/name).read_text();m=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert m and'FAILED 'not in text,name
 return dict(passed=int(m[1]),skipped=int(m[2]or 0),failed=0,seconds=float(m[3]),log=name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 foundation=HERE.parents[1]/'python_module_helpers/promotion/validation.json';prior=json.loads(foundation.read_text());assert prior['passed']
 staged={'openassembler/pyfront/compile.py':'pyfront','openassembler/pyfront/helpers.py':'helpers','openassembler/pyfront/modules.py':'modules'}
 changed={n for n,h in prior['source_hashes'].items()if n.startswith('openassembler/')and sha(ROOT/n)!=h}
 assert changed==set(staged)|{'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 for name,suffix in staged.items():
  assert sha(HERE/('baseline-'+suffix+'.py'))==prior['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
 assert(ROOT/'openassembler/pyfront/static_values.py').read_bytes()==(HERE/'static_values.py').read_bytes()
 assert not[n for n,h in prior['source_hashes'].items()if n.startswith('tests/')and sha(ROOT/n)!=h]
 for source,target in(('test_pyfront_static_values.py','tests/test_pyfront_static_values.py'),('static_value_fixtures.json','tests/data/static_value_fixtures.json')):assert(HERE/source).read_bytes()==(ROOT/target).read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==1440 and proof['bytes_checked']==9338880 and len(proof['native_comparisons'])==144
 report_path=HERE/'candidate-a100-20261124T021349Z.json';report=checked(report_path.name)
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(report_path)and report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/n)==h for n,h in report['source_hashes'].items()if n.startswith(('openassembler/','specs/'))and n not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 first=checked('first-hardware-audit.json');assert first['bytes_checked']==9338880 and first['script_sha256']==sha(HERE/'audit_first.py')and first['report_sha256']==sha(HERE/'candidate-a100-20261124T021108Z.json')
 offline=checked('candidate-offline.json');assert len(offline['results'])==144 and offline['executions']==2592 and all(sha(HERE/n)==h for n,h in offline['source_hashes'].items())
 baseline=checked('baseline.json');assert len(baseline['results'])==36 and all(r['rejected']for r in baseline['results'])and baseline['fixture_sha256']==sha(HERE/'fixtures.py')
 negative=checked('negative-controls.json');assert len(negative['results'])==29 and negative['script_sha256']==sha(HERE/'negative_controls.py')and negative['candidate_sha256']==sha(HERE/'static_values.py')
 property_gate=checked('property-probe.json');assert len(property_gate['results'])==2048 and property_gate['script_sha256']==sha(HERE/'property_probe.py')and property_gate['static_values_sha256']==sha(HERE/'static_values.py')
 oracle=checked('oracle-probe.json');assert len(oracle['results'])==4 and len(oracle['artifact_hashes'])==30 and oracle['script_sha256']==sha(HERE/'oracle_probe.py')
 assert all(sha(HERE/n)==h for n,h in oracle['artifact_hashes'].items())and all(sha(HERE/n)==h for n,h in oracle['source_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==144 and wheel['script_sha256']==sha(HERE/'audit_wheel.py')and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and all(sha(ROOT/n)==h for n,h in wheel['source_hashes'].items())
 frontend=checked('frontend-continuity.json');assert len(frontend['source_hashes'])==17 and frontend['kernels']==62 and frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py')and frontend['current_frontend_sha256']==sha(ROOT/'openassembler/pyfront/compile.py')
 assert all(sha(ROOT/n)==h for n,h in frontend['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==2150 and continuity['auditor_sha256']==sha(HERE/'continuity.py')and all(sha(ROOT/n)==h for n,h in continuity['backend_hashes'].items())
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261124T022017Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/n)==h for n,h in policy['compiler_inputs_sha256'].items())
 remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')and remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json')and remote['contract']['static_values']
 supported=suite('supported-suite.log');assert supported['passed']==3656 and supported['skipped']==0;supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite('compiler-only-suite.log');assert core['passed']==3171 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite('targeted-suite.log')['passed']==118
 docs={n:(HERE/n).is_file()and'Documentation check passed:'in(HERE/n).read_text()for n in('docs-check.log','exported-docs-check.log')}
 evidence=('baseline.json','candidate-offline.json','property-probe.json','negative-controls.json','oracle-probe.json','candidate-a100-20261124T021108Z.json','first-hardware-audit.json','candidate-a100-20261124T021349Z.json','candidate-audit.json','promotion-audit.json','frontend-continuity.json','application-continuity.json','normal-remote-audit.json','wheel-audit.json')
 sources=sorted(p for b in('openassembler','tests')for p in(ROOT/b).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G05 original explicit Final numeric/immutable tuple values, bounded source arithmetic/index/slice/len, per-kernel helper specialization and original dependency/type/diagnostic/native ownership; arbitrary objects/metaprogramming/host effects and full 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},foundation_record=str(foundation.relative_to(ROOT)),foundation_sha256=sha(foundation),changed_production_inputs=sorted(changed),added_production_inputs=['openassembler/pyfront/static_values.py'],changed_existing_test_inputs=[],added_test_inputs=['tests/test_pyfront_static_values.py','tests/data/static_value_fixtures.json'],evidence_sha256={n:sha(HERE/n)for n in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'normal-remote-build'not in p.parts and'wheel'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=1440,bytes_audited=9338880,native_comparisons=144,prior_application_native_comparisons=2150,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
