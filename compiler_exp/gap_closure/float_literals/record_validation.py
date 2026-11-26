"""Bind the literal promotion to the preceding complete pressure integration."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path):
    match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',path.read_text())
    assert match and 'FAILED 'not in path.read_text()
    return dict(passed=int(match[1]),skipped=int(match[2]or 0),failed=0,seconds=float(match[3]),log=path.name)


if __name__=='__main__':
    import torch
    assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1'
    baseline=HERE.parent/'pressure_schedule/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/ptx/parse.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    assert sha(HERE/'baseline-parse.py')==previous['source_hashes']['openassembler/ptx/parse.py']
    assert(ROOT/'openassembler/ptx/parse.py').read_bytes()==(HERE/'candidate-parse.py').read_bytes()
    changed_tests={name for name,digest in previous['source_hashes'].items()if name.startswith('tests/')and sha(ROOT/name)!=digest}
    assert changed_tests=={'tests/test_backend_memory_widths.py'},changed_tests
    assert sha(HERE/'baseline-test-backend-memory-widths.py')==previous['source_hashes']['tests/test_backend_memory_widths.py']
    proof=json.loads((HERE/'promotion-audit.json').read_text())
    assert proof['passed']and proof['audited_backend']=='production'and proof['launches']==320
    assert len(proof['native_comparisons'])==20 and proof['bytes_checked']==28737536
    assert proof['auditor_sha256']==sha(HERE/'audit.py')
    continuity=json.loads((HERE/'application-continuity.json').read_text());assert continuity['passed']and continuity['native_comparison_count']==1478
    for name in('candidate-offline.json','execution-offline.json'):
        check=json.loads((HERE/name).read_text());assert check['passed']and check.get('source_sha256')==sha(HERE/'candidate-parse.py')
    execution=json.loads((HERE/'execution-offline.json').read_text());assert sum(row['executions']for row in execution['results'])==360
    oracle_audit=json.loads((HERE/'oracle-evidence-audit.json').read_text())
    assert oracle_audit['passed']and len(oracle_audit['results'])==110 and oracle_audit['parser_sha256']==sha(ROOT/'openassembler/ptx/parse.py')
    assert all(sha(HERE/name)==digest for name,digest in oracle_audit['artifact_hashes'].items())
    supported=suite(HERE/'supported-suite.log');supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');core.update(python='3.14.5',torch='not installed')
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    evidence=('baseline.json','oracle-matrix.json','context-oracle.json','range-oracle.json','widen-oracle.json','integer-context-oracle.json','store64-oracle.json',
              'rounding-probe.json','oracle-evidence-audit.json','candidate-offline.json','execution-offline.json','candidate-a100-20261123T121453Z.json','promotion-audit.json','application-continuity.json','initial-suite-failure.json','expression-baseline.json','expression-semantics.json')
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)
    result=dict(schema=1,date='2026-11-23',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='Bounded G11 decimal binary64 parsing, scalar use conversion, raw payload/width contexts and exact A100 literal/arithmetic/guarded outputs; original 51-gap program remains incomplete',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),changed_existing_test_inputs=sorted(changed_tests),
                evidence_sha256={name:sha(HERE/name)for name in evidence},logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')},
                docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(launches=320,bytes_audited=28737536,native_comparisons=20,prior_application_native_comparisons=1478,
                              gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
