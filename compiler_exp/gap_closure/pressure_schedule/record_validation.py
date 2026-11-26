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
    baseline=HERE.parent/'spilling_groups/validation.json';previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/backend/spill.py','openassembler/kernels/sm80_gemm_dispatch.json','openassembler/kernels/sm80_rowwise_dispatch.json'},changed
    assert sha(HERE/'baseline-spill.py')==previous['source_hashes']['openassembler/backend/spill.py']
    assert 'openassembler/backend/pressure.py'not in previous['source_hashes']
    for name in('spill.py','pressure.py'):assert(ROOT/'openassembler/backend'/name).read_bytes()==(HERE/name).read_bytes()
    supported=suite(HERE/'supported-suite.log');supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');core.update(python='3.14.5',torch='not installed')
    proofs=[json.loads((HERE/name).read_text())for name in('promotion-audit.json','promotion-heldout-audit.json','promotion-final-audit.json')]
    assert all(proof['passed']and proof['audited_backend']=='production'for proof in proofs)
    assert [len(proof['native_comparisons'])for proof in proofs]==[136,32,8]
    for proof,auditor in zip(proofs,('audit.py','audit.py','audit_final.py')):
        assert proof['auditor_sha256']==sha(HERE/auditor)
    assert all(proof['final_allocator_sha256']==sha(HERE/'spill.py')for proof in proofs[:2])
    auditor_inputs=json.loads((HERE/'promotion-auditor-inputs.json').read_text())
    assert all(sha(HERE/name)==digest for name,digest in auditor_inputs['sources'].items())
    rehearsal=json.loads((HERE/'candidate-audit-custody.json').read_text());assert not rehearsal['promotion_authority']
    continuity=json.loads((HERE/'application-continuity.json').read_text());assert continuity['passed']and continuity['native_comparison_count']==1386
    controls=json.loads((HERE/'final-controls.json').read_text());assert controls['passed']and controls['source_hashes']['spill.py']==sha(HERE/'spill.py')
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)
    evidence=('baseline.json','candidate-a100-20261123T105809Z.json','heldout-a100-20261123T110440Z.json','final-a100-20261123T112918Z.json','promotion-audit.json','promotion-heldout-audit.json','promotion-final-audit.json','application-continuity.json','candidate-offline.json','heldout-offline.json','final-controls.json','compile-cost.json','terminator-repair.json','heldout-freeze.json','promotion-auditor-inputs.json','candidate-audit-custody.json')
    result=dict(schema=1,date='2026-11-23',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='G17 pressure-aware virtual scheduling with bounded spill/transfer admission, paired A100 kernel/resource controls and maximum-frame recovery; original 51-gap program remains incomplete',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),added_production_inputs=['openassembler/backend/pressure.py'],
                evidence_sha256={name:sha(HERE/name)for name in evidence},logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')},
                docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(correctness_launches=2144,timed_launches=1323,warmup_launches=189,
                              bytes_audited=sum(proof['bytes_checked']for proof in proofs),native_comparisons=176,prior_application_native_comparisons=1386,
                              gemm_policies=44,rowwise_policies=5,retained_policy_sessions=98,performance_claim='paired resident kernel intervals for exact pressure fixtures only'))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
