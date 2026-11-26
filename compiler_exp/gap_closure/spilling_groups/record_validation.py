"""Bind the bounded group/predicate promotion to its immutable scalar baseline."""
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
    text=path.read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text)
    assert match and 'FAILED 'not in text
    return dict(passed=int(match[1]),skipped=int(match[2]or 0),failed=0,seconds=float(match[3]),log=path.name)


if __name__=='__main__':
    import torch
    assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1','bind suite logs using their supported environment'
    baseline=HERE.parent/'spilling/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/backend/spill.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    assert sha(HERE/'baseline-spill.py')==previous['source_hashes']['openassembler/backend/spill.py']
    assert (ROOT/'openassembler/backend/spill.py').read_bytes()==(HERE/'spill.py').read_bytes()
    supported=suite(HERE/'supported-suite.log');supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');core.update(python='3.14.5',torch='not installed')
    proof=json.loads((HERE/'promotion-audit.json').read_text());assert proof['passed']and proof['audited_backend']=='production'
    assert proof['launches']==320 and proof['bytes_checked']==83787776 and len(proof['native_comparisons'])==20
    continuity=json.loads((HERE/'application-continuity.json').read_text());assert continuity['passed']and continuity['native_comparison_count']==1366
    controls=json.loads((HERE/'candidate-controls.json').read_text());assert controls['passed']and controls['allocator_sha256']==sha(HERE/'spill.py')
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)
    evidence=('baseline.json','baseline-audit.json','candidate-a100-20261123T102437Z.json','promotion-audit.json','application-continuity.json','candidate-offline.json','candidate-controls.json')
    result=dict(schema=1,date='2026-11-23',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='Bounded G17 aligned pair/quad and predicate spills, guarded/self-guarded/negated loop state and exact resources/ownership; original 51-gap program remains incomplete',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),
                evidence_sha256={name:sha(HERE/name)for name in evidence},logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')},
                docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(launches=320,bytes_audited=83787776,native_comparisons=20,prior_application_native_comparisons=1366,
                              gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
