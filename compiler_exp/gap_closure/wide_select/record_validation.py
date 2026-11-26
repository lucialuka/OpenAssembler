"""Bind wide-selection promotion to exact independent evidence and full integration."""
from datetime import datetime,timezone
import hashlib,json,platform,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

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
 baseline=HERE.parent/'device_linkage/promotion/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
 changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
 assert changed=={'openassembler/backend/ra.py','openassembler/backend/contracts.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
 for name in('ra','contracts'):
  source='openassembler/backend/'+name+'.py';assert sha(HERE/('baseline-'+name+'.py'))==previous['source_hashes'][source]
  assert(ROOT/source).read_bytes()==(HERE/('candidate-'+name+'.py')).read_bytes()
 changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest};assert not changed_tests,changed_tests
 for source,target in(('test_backend_wide_select.py','tests/test_backend_wide_select.py'),('wide_select_fixtures.json','tests/data/wide_select_fixtures.json')):assert(HERE/source).read_bytes()==(ROOT/target).read_bytes()
 proof=checked('promotion-audit.json');assert proof['audited_backend']=='production'and proof['launches']==1480 and proof['bytes_checked']==15155200 and len(proof['native_comparisons'])==148
 assert proof['auditor_sha256']==sha(HERE/'audit.py')and proof['report_sha256']==sha(HERE/'candidate-a100-20261123T205126Z.json')
 report=checked('candidate-a100-20261123T205126Z.json');assert report['harness_sha256']==sha(HERE/'run_modal.py')
 assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items()if name.startswith(('openassembler/','specs/'))and name not in('openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'))
 for key in('fixtures.py','noinline.cu'):assert report['source_hashes']['compiler_exp/gap_closure/wide_select/'+key]==sha(HERE/key)
 offline=checked('candidate-offline.json');assert len(offline['results'])==144 and sum(r['executions']for r in offline['results'])==2592
 assert all(sha(HERE/name)==digest for name,digest in offline['source_hashes'].items())
 rejected=checked('baseline.json');assert len(rejected['results'])==36 and all(not r['passed']and'unsupported PTX contract selp.'in r['error']for r in rejected['results'])
 oracle=checked('oracle-audit.json');assert len(oracle['artifact_hashes'])==144 and oracle['script_sha256']==sha(HERE/'audit_oracle.py')and oracle['report_sha256']==sha(HERE/'oracle-probe.json')
 assert all(sha(HERE/name)==digest for name,digest in oracle['artifact_hashes'].items())
 wheel=checked('wheel-audit.json');assert len(wheel['native_comparisons'])==148 and wheel['wheel_sha256']==sha(HERE/wheel['wheel'])and wheel['script_sha256']==sha(HERE/'audit_wheel.py')
 assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
 continuity=checked('application-continuity.json');assert continuity['native_comparison_count']==1786 and continuity['auditor_sha256']==sha(HERE/'continuity.py')
 assert all(sha(ROOT/name)==digest for name,digest in continuity['backend_hashes'].items())
 policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261123T205620Z.json';policy=json.loads(policy_path.read_text());assert policy['passed']and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
 assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
 supported=suite(HERE/'supported-suite.log');assert supported['passed']==3361 and supported['skipped']==0
 supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
 core=suite(HERE/'compiler-only-suite.log');assert core['passed']==2876 and core['skipped']==15;core.update(python='3.14.5',torch='not installed')
 assert suite(HERE/'targeted-suite.log')['passed']==87
 docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
 evidence=('baseline.json','first-baseline.json','first-candidate-offline.json','first-oracle-probe.json','candidate-offline.json','oracle-probe.json','oracle-audit.json','candidate-a100-20261123T205126Z.json','candidate-audit.json','promotion-audit.json','wheel-audit.json','application-continuity.json','ptx-reference.json')
 sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
 result=dict(schema=1,date='2026-11-24',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,scope='G11 scalar selp.b64/u64/s64/f64 via recovered scalar SEL words, exact snapshots/guards/aliases/negation/parameter/immediate and original CUDA noinline gate; general scalar arithmetic/ABI and original 51-gap program remain incomplete',supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),added_production_inputs=[],added_test_inputs=['tests/test_backend_wide_select.py','tests/data/wide_select_fixtures.json'],changed_existing_test_inputs=[],evidence_sha256={name:sha(HERE/name)for name in evidence},policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),hardware=dict(launches=1480,bytes_audited=15155200,native_comparisons=148,prior_application_native_comparisons=1786,gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
