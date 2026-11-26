"""Bind promoted logical helper signatures to exact independent integration evidence."""
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(name):
    data = json.loads((HERE/name).read_text())
    assert data['passed'], name
    return data


def suite(name, count, skips):
    text = (HERE/name).read_text()
    match = re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z', text)
    assert match and 'FAILED ' not in text, name
    assert int(match[1]) == count and int(match[2] or 0) == skips, (name, match.groups())
    return dict(passed=count, skipped=skips, failed=0, seconds=float(match[3]), log=name)


if __name__ == '__main__':
    import torch
    assert platform.python_version() == '3.12.10' and torch.__version__ == '2.7.1'
    previous_path = HERE.parents[1]/'python_boolean_values/promotion/validation.json'
    previous = json.loads(previous_path.read_text())
    assert previous['passed']
    staged = {'openassembler/pyfront/compile.py': 'pyfront',
              'openassembler/pyfront/helpers.py': 'helpers',
              'openassembler/pyfront/modules.py': 'modules'}
    changed = {name for name, digest in previous['source_hashes'].items() if sha(ROOT/name) != digest}
    assert changed == set(staged), changed
    for name, suffix in staged.items():
        assert sha(HERE/('baseline-'+suffix+'.py')) == previous['source_hashes'][name]
        assert (ROOT/name).read_bytes() == (HERE/('candidate-'+suffix+'.py')).read_bytes()
    for destination, source in [('tests/test_pyfront_boolean_helpers.py', 'production-test_pyfront_boolean_helpers.py'),
                                ('tests/data/boolean_helper_fixtures.json', 'boolean_helper_fixtures.json')]:
        assert (ROOT/destination).read_bytes() == (HERE/source).read_bytes()
        assert destination not in previous['source_hashes']
    proof = checked('promotion-audit.json')
    candidate = checked('candidate-audit.json')
    assert proof['audited_backend'] == 'production' and candidate['audited_backend'] == 'isolated candidate'
    for data in (proof, candidate):
        assert data['launches'] == 2400 and data['bytes_checked'] == 16220160
        assert len(data['native_comparisons']) == 240 and data['corrupted_output_rejected']
        assert data['auditor_sha256'] == sha(HERE/'audit.py')
    report_path = HERE/'candidate-a100-20261124T123953Z.json'
    report = checked(report_path.name)
    assert report['error'] is None and len(report['results']) == 2400
    assert all(row['passed'] for row in report['results'])
    assert proof['report_sha256'] == candidate['report_sha256'] == sha(report_path)
    assert report['harness_sha256'] == sha(HERE/'run_modal.py')
    assert report['archive']['sha256'] == sha(HERE/report['archive']['path'])
    assert all(sha(ROOT/name) == digest for name, digest in report['source_hashes'].items()
               if name.startswith(('openassembler/', 'specs/')))
    for name, count in [('candidate-offline.json', 240), ('baseline.json', 60),
                        ('negative-controls.json', 22), ('control-properties.json', 128)]:
        data = checked(name)
        assert len(data['results']) == count
        assert all(sha(HERE/path) == digest for path, digest in data['source_hashes'].items())
    assert checked('candidate-offline.json')['executions'] == 4320
    assert checked('control-properties.json')['executions'] == 1536
    assert all(row['rejected'] for row in checked('baseline.json')['results'])
    assert all(row['rejected'] for row in checked('negative-controls.json')['results'])
    oracle = checked('oracle-probe.json')
    assert len(oracle['results']) == 3 and len(oracle['artifact_hashes']) == 22
    assert oracle['script_sha256'] == sha(HERE/'oracle_probe.py')
    assert all(sha(HERE/path) == digest for path, digest in oracle['artifact_hashes'].items())
    assert all(sha(HERE/path) == digest for path, digest in oracle['source_hashes'].items())
    wheel = checked('wheel-audit.json')
    assert len(wheel['source_hashes']) == 132 and len(wheel['native_comparisons']) == 240
    assert wheel['script_sha256'] == sha(HERE/'audit_wheel.py') and wheel['report_sha256'] == sha(report_path)
    assert wheel['wheel_sha256'] == sha(HERE/wheel['wheel'])
    assert all(sha(ROOT/name) == digest for name, digest in wheel['source_hashes'].items())
    frontend = checked('frontend-continuity.json')
    assert frontend['kernels'] == 62 and len(frontend['source_hashes']) == 20
    assert not frontend['changed_source_outputs'] and frontend['auditor_sha256'] == sha(HERE/'frontend_continuity.py')
    assert all(sha(ROOT/name) == digest for name, digest in frontend['source_hashes'].items())
    continuity = checked('application-continuity.json')
    assert continuity['native_comparison_count'] == 3494 and continuity['inherited_native_comparison_count'] == 3254
    assert continuity['replayed_native_comparison_count'] == len(continuity['native_comparisons']) == 240
    assert all(row['passed'] for row in continuity['native_comparisons'])
    assert continuity['auditor_sha256'] == sha(HERE/'continuity.py') and continuity['previous_validation_sha256'] == sha(previous_path)
    assert continuity['validated_backend_report_sha256'] == sha(report_path)
    assert continuity['inherited_application_chain_sha256'] == sha(ROOT/continuity['inherited_application_chain'])
    assert all(sha(ROOT/name) == digest for name, digest in continuity['unchanged_backend_hashes'].items())
    remote = checked('normal-remote-audit.json')
    assert remote['script_sha256'] == sha(HERE/'normal_remote_check.py')
    assert remote['manifest_sha256'] == sha(HERE/'normal-remote-build/manifest.json')
    assert any('pred' in row['inputs'] or row['output'] == 'pred' for row in remote['contract']['helpers'].values())
    abi_path = HERE.parent/'baseline-replay.json'
    abi = json.loads(abi_path.read_text())
    assert abi['passed'] and len(abi['results']) == 3
    assert abi['original_toolkit_record_sha256'] == sha(HERE.parent/'baseline-toolkit.json')
    assert all(sha(HERE.parent/name) == digest for name, digest in abi['source_hashes'].items())
    policy_path = ROOT/'compiler_exp/gap_closure/foundations_20261121/audit-20261124T121203Z.json'
    policy = json.loads(policy_path.read_text())
    assert policy['passed'] and policy['counts'] == dict(gemm_contracts=44, rowwise_contracts=5, retained_session_checks=98)
    assert all(sha(ROOT/name) == digest for name, digest in policy['compiler_inputs_sha256'].items())
    supported = suite('supported-suite.log', 4593, 0)
    supported.update(python=sys.version, torch=torch.__version__, platform=platform.platform())
    core = suite('compiler-only-suite.log', 4108, 15)
    core.update(python='3.14.5', torch='not installed')
    assert suite('candidate-tests.log', 155, 0)['passed'] == 155
    docs = {name: 'Documentation check passed:' in (HERE/name).read_text()
            for name in ('docs-check.log', 'exported-docs-check.log')}
    evidence = ('baseline.json', 'candidate-offline.json', 'negative-controls.json', 'control-properties.json',
                'oracle-probe.json', report_path.name, 'candidate-audit.json', 'promotion-audit.json',
                'frontend-continuity.json', 'application-continuity.json', 'normal-remote-audit.json',
                'wheel-audit.json', 'promotion.json', 'boolean_helper_fixtures.json')
    sources = sorted(p for base in ('openassembler', 'tests') for p in (ROOT/base).rglob('*')
                     if p.suffix in ('.py', '.json', '.ptx') and '__pycache__' not in p.parts)
    sources.append(ROOT/'specs/sm_80/memory_templates.json')
    result = dict(schema=1, date='2026-11-24', created_utc=datetime.now(timezone.utc).isoformat(),
                  passed=all(docs.values()), all_gaps_closed=False,
                  scope='G03/G04 exact logical Boolean helper parameter/return signatures through pure static expansion, predicate copies/complete paths/caller isolation/source/native ownership and independent original-source A100/effect gates. Physical Boolean launch/storage ABI, type unions/general Python effects and full 51-gap program remain incomplete',
                  supported_suite=supported, compiler_only_suite=core,
                  foundation_record=str(previous_path.relative_to(ROOT)), foundation_sha256=sha(previous_path),
                  changed_production_inputs=sorted(changed), added_production_inputs=[],
                  changed_existing_test_inputs=[], added_test_inputs=['tests/test_pyfront_boolean_helpers.py', 'tests/data/boolean_helper_fixtures.json'],
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
                  evidence_sha256={name: sha(HERE/name) for name in evidence},
                  original_abi_replay=str(abi_path.relative_to(ROOT)), original_abi_replay_sha256=sha(abi_path),
                  policy_audit=str(policy_path.relative_to(ROOT)), policy_audit_sha256=sha(policy_path),
                  logs_sha256={p.name: sha(p) for p in HERE.glob('*.log') if p.name != 'record-validation.log'},
                  experiment_source_hashes={str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
                                           if p.suffix in ('.py', '.ptx', '.cu') and '__pycache__' not in p.parts
                                           and 'normal-remote-build' not in p.parts and 'wheel' not in p.parts},
                  docs=dict(status='passed' if all(docs.values()) else 'pending', checks=docs),
                  hardware=dict(launches=2400, bytes_audited=16220160, native_comparisons=240,
                                prior_application_native_comparisons=3494, inherited_native_comparisons=3254,
                                replayed_native_comparisons=240, gemm_policies=44, rowwise_policies=5,
                                retained_policy_sessions=98, policy_compiler_inputs_unchanged=True, performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Supported', supported['passed'], 'compiler-only', core['passed'], 'docs', result['docs']['status'])
