"""Bind direct-function promotion to independent original-source gates and prior integration."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def checked(name):
 obj=json.loads((HERE/name).read_text());assert obj['passed'],name;return obj
def suite(path):
 text=path.read_text();m=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text);assert m and 'FAILED 'not in text,path
 return dict(passed=int(m[1]),skipped=int(m[2]or 0),failed=0,seconds=float(m[3]),log=path.name)
if __name__=='__main__':
 import torch
 assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
 baseline=HERE.parent/'python_shared_wide/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
 assert changed=={'openassembler/ptx/parse.py','openassembler/build.py','openassembler/native_ownership.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 candidates={'openassembler/ptx/parse.py':('candidate-parse.py','baseline-parse.py'),'openassembler/build.py':('candidate-build.py','baseline-build.py'),'openassembler/native_ownership.py':('candidate-native-ownership.py','baseline-native-ownership.py')}
 for name,(candidate,old)in candidates.items():
  assert sha(HERE/old)==previous['source_hashes'][name]
  assert(ROOT/name).read_bytes()==(HERE/candidate).read_bytes()
 assert(ROOT/'openassembler/ptx/inlining.py').read_bytes()==(HERE/'inlining.py').read_bytes()
 changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest};assert not changed_tests,changed_tests
 for name in('test_ptx_device_functions.py','data/device_function_fixtures.json'):assert(ROOT/'tests'/name).read_bytes()==(HERE/name).read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==960 and proof['bytes_checked']==6881280 and len(proof['native_comparisons'])==60
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(HERE/'candidate-a100-20261123T181957Z.json')
 report=checked('candidate-a100-20261123T181957Z.json');assert report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 offline=checked('candidate-offline.json');assert len(offline['results'])==60 and sum(r['executions']for r in offline['results'])==1080
 assert all(sha(HERE/name)==digest for name,digest in offline['source_hashes'].items())
 oracle=checked('oracle-audit.json');assert len(oracle['artifact_hashes'])==60 and oracle['script_sha256']==sha(HERE/'audit_oracle.py')and oracle['report_sha256']==sha(HERE/'oracle-probe.json')
 assert all(sha(HERE/name)==digest for name,digest in oracle['artifact_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==60 and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and wheel['script_sha256']==sha(HERE/'audit_wheel.py')
 assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==1678 and continuity['auditor_sha256']==sha(HERE/'complete_continuity.py')
 assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T180951Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 supported=suite(HERE/'supported-suite.log');assert supported['passed']==3229 and supported['skipped']==0
 supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite(HERE/'compiler-only-suite.log');assert core['passed']==2744 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 negative=checked('void-effect-negative.json');assert 'st.global.b64 [%dst],%result'not in negative['source']and 'st.global.u64 [%p],%v'not in negative['mutant']
 first=checked('first-hardware-audit.json');assert first['launches']==512 and first['bytes_checked']==3670016 and first['script_sha256']==sha(HERE/'audit_first_hardware.py')
 evidence=('baseline.json','candidate-offline.json','oracle-probe.json','oracle-audit.json','candidate-a100-20261123T173558Z.json','candidate-a100-20261123T181957Z.json','candidate-audit.json','promotion-audit.json','wheel-audit.json','application-continuity.json')
 evidence+=('first-hardware-audit.json','base-application-continuity.json','void-effect-negative.json','candidate-a100-20261123T175233Z.json','pre-effect-promotion-audit.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G16/G10 direct scalar PTX module-local static inlining, original CUDA noinline controls and exact ABI/ownership/local resources; native stacks/linking and original 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),added_production_inputs=['openassembler/ptx/inlining.py'],changed_existing_test_inputs=[],evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=960,bytes_audited=6881280,native_comparisons=60,prior_application_native_comparisons=1678,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
