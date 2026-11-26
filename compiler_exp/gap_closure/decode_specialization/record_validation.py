"""Bind the completed decode cycle to the earlier foundation integration."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(path):
    text=path.read_text();match=re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z',text)
    assert match and 'FAILED 'not in text
    return dict(passed=int(match[1]),skipped=int(match[2]or 0),failed=0,seconds=float(match[3]),log=path.name)


if __name__=='__main__':
    import torch
    assert platform.python_version()=='3.12.10'and torch.__version__=='2.7.1','use the tested supported environment to bind its log'
    baseline=HERE.parent/'integration_20261123/validation.json'
    previous=json.loads(baseline.read_text());assert previous['passed']
    changed={name for name,digest in previous['source_hashes'].items()if name.startswith('openassembler/')and sha(ROOT/name)!=digest}
    assert changed=={'openassembler/modelfront/decode.py'},changed
    supported=suite(HERE/'supported-suite.log');supported.update(python=sys.version,torch=torch.__version__,platform=platform.platform())
    core=suite(HERE/'compiler-only-suite.log');core.update(python='3.14.5',torch='not installed')
    proof=json.loads((HERE/'promotion-audit.json').read_text());assert proof['passed']and proof['audited_backend']=='production'
    docs={name:(HERE/name).is_file()and'Documentation check passed:'in(HERE/name).read_text()for name in('docs-check.log','exported-docs-check.log')}
    sources=sorted(p for base in('openassembler','tests')for p in(ROOT/base).rglob('*')if p.suffix in('.py','.json','.ptx')and'__pycache__'not in p.parts)
    result=dict(schema=1,date='2026-11-23',created_utc=datetime.now(timezone.utc).isoformat(),passed=all(docs.values()),all_gaps_closed=False,
                scope='Explicit bounded G39 specialization, preserving default exact precompiled decode; earlier foundations retained under their own integration record',
                supported_suite=supported,compiler_only_suite=core,source_hashes={str(p.relative_to(ROOT)):sha(p)for p in sources},
                foundation_record=str(baseline.relative_to(ROOT)),foundation_sha256=sha(baseline),changed_production_inputs=sorted(changed),
                evidence_sha256={name:sha(HERE/name)for name in('candidate-hardware-20261123T085245Z.json','promotion-audit.json','candidate-cpu-results.json')},
                logs_sha256={p.name:sha(p)for p in HERE.glob('*.log')},docs=dict(status='passed'if all(docs.values())else'pending',checks=docs),
                hardware=dict(comparisons=126,elements=9520,profiled_runs=40,profiled_oa_launches=440,build_requests=16,
                              native_bundles=25,native_comparisons=711,performance_claim=False))
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Supported',supported['passed'],'compiler-only',core['passed'],'docs',result['docs']['status'])
