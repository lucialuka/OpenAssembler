"""Bind exact scalar entry ABI promotion to independent numerical/integration proofs."""
from datetime import datetime,timezone
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(name):
    value=json.loads((HERE/name).read_text())
    assert value['passed'],name
    return value


def suite(name,count,skips):
    text=(HERE/name).read_text()
    match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text)
    assert match and 'FAILED ' not in text,name
    assert int(match[1])==count and int(match[2]or 0)==skips,(name,match.groups())
    return dict(passed=count,skipped=skips,failed=0,seconds=float(match[3]),log=name)


if __name__=='__main__':
    import torch
    assert platform.python_version()=='3.12.10' and torch.__version__=='2.7.1'
    previous_path=HERE.parent/'python_boolean_abi/helpers/validation.json'
    previous=json.loads(previous_path.read_text());assert previous['passed']
    staged=json.loads((HERE/'inputs.json').read_text())
    policy_names={'openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'}
    changed={name for name,digest in previous['source_hashes'].items() if sha(ROOT/name)!=digest}
    assert changed==set(staged.values())|policy_names,changed
    for suffix,name in staged.items():
        assert sha(HERE/('baseline-'+suffix+'.py'))==previous['source_hashes'][name]
        assert (ROOT/name).read_bytes()==(HERE/('candidate-'+suffix+'.py')).read_bytes()
    for destination,source in [('tests/test_narrow_parameters.py','test_narrow_parameters.py'),('tests/data/narrow_parameter_fixtures.json','narrow_parameter_fixtures.json')]:
        assert destination not in previous['source_hashes'] and (ROOT/destination).read_bytes()==(HERE/source).read_bytes()
    report_path=HERE/'candidate-a100-20261125T011357Z.json'
    report=checked(report_path.name)
    assert report['error'] is None and len(report['results'])==14400 and len(report['explicit_launch_results'])==240
    assert all(row['passed']for row in report['results']+report['explicit_launch_results'])
    assert report['archive']['sha256']==sha(HERE/report['archive']['path']) and report['harness_sha256']==sha(HERE/'run_modal.py')
    assert all(sha(ROOT/name)==digest for name,digest in report['source_hashes'].items() if name.startswith(('openassembler/','specs/')) and name not in policy_names)
    for name in ('candidate-audit.json','promotion-audit.json'):
        proof=checked(name)
        assert proof['report_sha256']==sha(report_path) and proof['auditor_sha256']==sha(HERE/'audit.py')
        assert proof['launches']==14400 and proof['explicit_launches']==240 and proof['bytes_checked']==67461120
        assert len(proof['native_comparisons'])==240 and proof['corrupted_output_rejected']
    assert checked('promotion-audit.json')['audited_backend']=='production'
    promotion=checked('promotion.json');assert promotion['candidate_audit_sha256']==sha(HERE/'candidate-audit.json') and promotion['existing_tests_changed']is False
    for name,count in [('baseline.json',60),('candidate-offline.json',240),('negative-controls.json',14)]:
        data=checked(name);assert len(data['results'])==count
        assert all(sha(HERE/path)==digest for path,digest in data['source_hashes'].items())
    assert checked('candidate-offline.json')['executions']==34560
    assert all(row['rejected']for row in checked('baseline.json')['results'])
    assert all(row['rejected']for row in checked('negative-controls.json')['results'])
    layouts=checked('layout-properties.json')
    assert len(layouts['results'])==4096 and sum(len(row['extractions'])for row in layouts['results'])==407972
    assert layouts['script_sha256']==sha(HERE/'layout_properties.py') and layouts['parser_sha256']==sha(ROOT/'openassembler/ptx/parse.py')
    chain=checked('application-continuity.json')
    assert chain['native_comparison_count']==chain['native_exact_count']==len(chain['native_comparisons'])==3734
    assert chain['stall_yield_changed_count']==0 and all(row['passed']and row['native_exact']for row in chain['native_comparisons'])
    assert chain['previous_validation_sha256']==sha(previous_path) and chain['auditor_sha256']==sha(HERE/'continuity.py')
    assert all(sha(ROOT/name)==digest for name,digest in chain['candidate_source_hashes'].items())
    assert all(sha(ROOT/name)==digest for name,digest in chain['backend_hashes'].items())
    assert all(sha(ROOT/name)==digest for name,digest in chain['retained_hashes'].items())
    frontend=checked('frontend-continuity.json');assert frontend['kernels']==62 and len(frontend['source_hashes'])==20 and not frontend['changed_source_outputs']
    assert frontend['auditor_sha256']==sha(HERE/'frontend_continuity.py') and all(sha(ROOT/name)==digest for name,digest in frontend['source_hashes'].items())
    remote=checked('normal-remote-audit.json');assert remote['script_sha256']==sha(HERE/'normal_remote_check.py')
    assert remote['manifest_sha256']==sha(HERE/'normal-remote-build/manifest.json') and len(remote['parameters'])==60
    wheel=checked('wheel-audit.json');assert len(wheel['source_hashes'])==132 and len(wheel['native_comparisons'])==240
    assert wheel['report_sha256']==sha(report_path) and wheel['wheel_sha256']==sha(HERE/wheel['wheel']) and wheel['script_sha256']==sha(HERE/'audit_wheel.py')
    assert all(sha(ROOT/name)==digest for name,digest in wheel['source_hashes'].items())
    policy_path=ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261125T011749Z.json'
    policy=json.loads(policy_path.read_text());assert policy['passed'] and policy['counts']==dict(gemm_contracts=44,rowwise_contracts=5,retained_session_checks=98)
    assert all(sha(ROOT/name)==digest for name,digest in policy['compiler_inputs_sha256'].items())
    for name in policy_names:
        table=json.loads((ROOT/name).read_text());assert table['source_continuity']['sha256']==sha(policy_path)
    replay_path=HERE.parent/'python_boolean_abi/parameter-boundary-replay.json'
    replay=json.loads(replay_path.read_text());assert replay['passed'] and len(replay['results'])==8
    assert all(sha(ROOT/name)==digest for field in ('source_hashes','original_records_sha256') for name,digest in replay[field].items())
    for name in ('candidate-a100-20261125T010951Z.json','candidate-a100-20261125T011123Z.json'):
        failure=json.loads((HERE/name).read_text());assert failure['passed']is False and failure['error']and not failure['results']and not failure['explicit_launch_results']
    supported=suite('supported-suite.log',4764,0);supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite('compiler-only-suite.log',4279,15);core.update(python='3.14.5',torch='not installed')
    suite('candidate-tests.log',171,0)
    docs={name:'Documentation check passed:'in(HERE/name).read_text()for name in ('docs-check.log','exported-docs-check.log')}
    evidence=('baseline.json','candidate-offline.json','negative-controls.json','layout-properties.json',report_path.name,'candidate-audit.json','promotion-audit.json',
              'promotion.json','application-continuity.json','frontend-continuity.json','normal-remote-audit.json','wheel-audit.json',
              'narrow_parameter_fixtures.json','candidate-a100-20261125T010951Z.json','candidate-a100-20261125T011123Z.json')
    sources=sorted(p for base in ('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)+[ROOT/'specs/sm_80/memory_templates.json']
    data=dict(schema=1,date='2026-11-25',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
              scope='G10/G11 exact scalar b/u/s8/16 entry slots, natural layout, signed/zero extension, within-slot partial loads, guarded/loop/alias snapshots and coordinated verifier/explicit runner/batch-driver ABI. Original PTX/A100/explicit launch/native/installed/source gates; Boolean source launch/storage, aggregate/vector/call ABI and complete 51-gap program remain open',
              supported_suite=supported,compiler_only_suite=core,foundation_record=str(previous_path.relative_to(ROOT)),foundation_sha256=sha(previous_path),
              changed_production_inputs=sorted(changed),added_production_inputs=[],changed_existing_test_inputs=[],added_test_inputs=['tests/test_narrow_parameters.py','tests/data/narrow_parameter_fixtures.json'],
              source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},evidence_sha256={name:sha(HERE/name)for name in evidence},
              logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')if p.name!='record-validation.log'},
              experiment_source_hashes={str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.rglob('*'))if p.suffix in('.py','.ptx','.cu')and'__pycache__'not in p.parts and'wheel'not in p.parts and'normal-remote-build'not in p.parts},
              policy_audit=str(policy_path.relative_to(ROOT)),policy_audit_sha256=sha(policy_path),
              policy_continuity_auditor_sha256=sha(ROOT/'compiler_exp/gap_closure/foundations_20261121/continuity.py'),
              original_parameter_replay=str(replay_path.relative_to(ROOT)),original_parameter_replay_sha256=sha(replay_path),
              docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
              hardware=dict(launches=14400,explicit_launches=240,bytes_audited=67461120,native_comparisons=240,prior_application_native_comparisons=3734,prior_native_exact=3734,
                            gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',data['docs']['status'])
