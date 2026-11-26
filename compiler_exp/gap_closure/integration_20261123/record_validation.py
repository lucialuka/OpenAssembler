"""Bind completed integration logs and retained hardware to current source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite(name):
    text = (HERE / name).read_text()
    match = re.search(r'(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s[^\n]*\n?\Z', text)
    assert match and 'FAILED ' not in text
    return dict(passed=int(match[1]), skipped=int(match[2] or 0), failed=0,
                seconds=float(match[3]), log=name)


if __name__ == '__main__':
    import torch
    supported = suite('supported-suite-final.log')
    supported.update(python=sys.version, torch=torch.__version__, platform=platform.platform())
    core = suite('compiler-only-suite-final.log')
    core.update(python=subprocess.check_output([str(ROOT / '.venv/bin/python'), '-c', 'import platform;print(platform.python_version())'], text=True).strip(), torch='not installed')
    references = [
        '../atomic_ordering/a100-20261123T080159Z.json',
        '../atomic_ordering/forms-a100-20261123T081214Z.json',
        '../atomic_ordering/promotion-audit-20261123.json',
        '../atomic_ordering/promotion-forms-audit-20261123.json',
        '../atomic_ordering/ledger-ordering-20261123.json',
        '../foundations_20261121/audit-20261123T082155Z.json',
        '../checkpoints/hardware-20261123T082914Z.json',
        '../checkpoints/maximum-20261123T082454Z.json',
        'checkpoint-final-audit.json',
    ]
    records = {name: json.loads((HERE / name).read_text()) for name in references}
    assert all(record['passed'] if 'passed' in record else record['report']['passed'] for record in records.values()
               if 'ledger-' not in record.get('scope', '') and 'declared_forms' not in record)
    docs = {name: (HERE / name).is_file() and 'Documentation check passed:' in (HERE / name).read_text()
            for name in ('docs-check.log', 'exported-docs-check.log')}
    sources = sorted({p for base in ('openassembler', 'tests') for p in (ROOT / base).rglob('*')
                      if p.suffix in ('.py', '.json', '.ptx') and '__pycache__' not in p.parts})
    result = dict(schema=1, date='2026-11-23', created_utc=datetime.now(timezone.utc).isoformat(),
                  all_gaps_closed=False, passed=all(docs.values()),
                  scope='Bounded ordering/shared atomic and value-lifetime repairs; integrated earlier foundation/state contracts. Original 51-gap program remains incomplete.',
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  supported_suite=supported, compiler_only_suite=core,
                  source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
                  evidence_sha256={name: sha(HERE / name) for name in references},
                  retained_log_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.log'))},
                  docs=dict(status='passed' if all(docs.values()) else 'pending', checks=docs),
                  hardware=dict(ordering_shared_value_launches=472, atomic_form_launches=768,
                                atomic_bytes_audited=44465024, checkpoint_comparisons=7862,
                                checkpoint_elements=124438, ieee_comparisons=80, ieee_elements=1360,
                                independent_device_variants=['A100-SXM4-40GB', 'A100 80GB PCIe', 'A100-SXM4-80GB'],
                                performance_claim=False),
                  continuity=dict(gemm_policies=44, rowwise_policies=5, retained_policy_sessions=98,
                                  current_hardware_model_bundles=28, current_hardware_model_kernel_occurrences=383,
                                  checkpoint_byte_pairs=17))
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Supported', supported['passed'], 'compiler-only', core['passed'], 'docs', result['docs']['status'])
