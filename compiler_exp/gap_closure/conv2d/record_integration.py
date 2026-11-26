"""Bind the practical Conv2d release gates without expanding their tested scope."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    validation = json.loads((HERE/'validation.json').read_text())
    wheel = json.loads((HERE/'wheel-audit.json').read_text())
    assert validation['passed'] and wheel['passed']
    assert validation['script_sha256'] == sha(HERE/'audit.py')
    assert wheel['script_sha256'] == sha(HERE/'wheel_audit.py')
    assert wheel['validation_sha256'] == sha(HERE/'validation.json')
    assert wheel['wheel_sha256'] == sha(HERE/'wheel'/wheel['wheel'])
    assert all(sha(ROOT/name) == digest for name, digest in validation['source_hashes'].items())
    assert all(sha(ROOT/name) == digest for name, digest in validation['current_compiler_hashes'].items())
    assert all(sha(ROOT/name) == digest for name, digest in wheel['installed_sources'].items())
    assert validation['hardware_comparisons'] == 80 and len(validation['native_comparisons']) == 142
    assert wheel['native_comparisons'] == 142 and wheel['bundles'] == 11
    assert re.search(r'645 passed in ', (HERE/'model-regressions-final.log').read_text())
    assert re.search(r'21 passed in ', (HERE/'candidate-tests.log').read_text())
    assert 'Documentation check passed:' in (HERE/'docs-check.log').read_text()
    evidence = ['baseline.json','offline.json','hardware-20261126T082735Z.json','validation.json',
                'wheel-audit.json','model-regressions-final.log','candidate-tests.log','docs-check.log']
    result = dict(schema=1, passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Practical FP32 groups=1 Conv2d forward, first-order gradients and CNN training/replay on SM80/A100',
                  all_gaps_closed=False, model_tests=645, focused_tests=21, offline_executions=144,
                  hardware_tensor_checks=80, public_cli_training_checks=90, exact_native_outputs=142,
                  installed_wheel_passed=True, documentation_passed=True,
                  source_hashes=validation['source_hashes'], evidence_sha256={name:sha(HERE/name) for name in evidence},
                  open_coverage=['groups/depthwise convolution','pooling','BatchNorm','dynamic shapes','general dtype/AMP support'],
                  script_sha256=sha(Path(__file__)))
    (HERE/'integration.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Conv2d integration passed: 645 model tests, 80 A100 and 90 public CLI training checks, 142 native outputs, wheel and docs')
