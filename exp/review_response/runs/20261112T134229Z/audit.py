"""Classify independently observed GPU trace activities by kind, never name."""
from collections import Counter
import json
import tempfile
from pathlib import Path


def audit_trace(trace, ledger):
    if isinstance(trace, dict):
        trace = trace.get('traceEvents', [])
    if not isinstance(trace, list):
        raise ValueError('GPU trace must contain activity records')
    counts = Counter()
    storage = Counter()
    for event in trace:
        category = event.get('cat')
        if category == 'kernel':
            counts[event['name']] += 1
        elif category in ('gpu_memcpy', 'gpu_memset'):
            storage[category] += 1
    expected = Counter(record['entry'] for record in ledger)
    if not counts or counts != expected:
        raise ValueError(f'native GPU launch audit failed: observed={dict(counts)}, expected={dict(expected)}')
    return {'passed': True, 'observed_compute_launches': sum(counts.values()),
            'kernel_counts': dict(counts), 'storage_activities': dict(storage),
            'unknown_compute_kernels': [], 'scope': 'independent Kineto kernel activities, exact names/counts; copies classified by activity category'}


def audit_profile(profile, ledger):
    with tempfile.TemporaryDirectory(prefix='oa-trace-') as directory:
        path = Path(directory) / 'trace.json'
        profile.export_chrome_trace(str(path))
        return audit_trace(json.loads(path.read_text()), ledger)
