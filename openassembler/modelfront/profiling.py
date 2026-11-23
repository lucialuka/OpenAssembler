"""Classify independently observed GPU trace activities by kind, never name."""
from collections import Counter
import json
import tempfile
from pathlib import Path
import math
import hashlib
import statistics


def audit_trace(trace, ledger):
    if isinstance(trace, dict):
        trace = trace.get('traceEvents', [])
    if not isinstance(trace, list):
        raise ValueError('GPU trace must contain activity records')
    counts = Counter()
    storage = Counter()
    durations = {}
    for event in trace:
        category = event.get('cat')
        if category == 'kernel':
            counts[event['name']] += 1
            duration = event.get('dur')
            if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
                durations.setdefault(event['name'], []).append(duration)
        elif category in ('gpu_memcpy', 'gpu_memset'):
            storage[category] += 1
    expected = Counter(record['entry'] for record in ledger)
    if not counts or counts != expected:
        raise ValueError(f'native GPU launch audit failed: observed={dict(counts)}, expected={dict(expected)}')
    return {'passed': True, 'observed_compute_launches': sum(counts.values()),
            'kernel_counts': dict(counts), 'storage_activities': dict(storage),
            'kernel_duration_us': durations,
            'duration_scope': 'raw kernel activities from this profiled invocation; not a steady-state timing distribution',
            'unknown_compute_kernels': [], 'scope': 'independent Kineto kernel activities, exact names/counts; copies classified by activity category'}


def audit_profile(profile, ledger):
    with tempfile.TemporaryDirectory(prefix='oa-trace-') as directory:
        path = Path(directory) / 'trace.json'
        profile.export_chrome_trace(str(path))
        return audit_trace(json.loads(path.read_text()), ledger)


def profile_iterations(steps, repetitions=1):
    """Choose existing final iterations; profiling never adds a training step."""
    if type(steps) is not int or steps < 1 or type(repetitions) is not int or not 1 <= repetitions <= steps:
        raise ValueError('profile_iterations must be an integer in [1, steps]')
    return list(range(steps - repetitions, steps))


def kernel_distributions(audits, launches, resources):
    """Summarize repeated audited activities under exact executable contracts.

    These are profiler observations, not profiler-free kernel benchmarks. A
    launch entry must retain one binary/launch identity throughout the run.
    """
    groups = {}
    seen_iterations = set()
    for audit in audits:
        iteration = audit.get('iteration')
        if audit.get('passed') is not True or type(iteration) is not int or iteration < 0 or iteration in seen_iterations:
            raise ValueError('kernel distributions require distinct passed iteration audits')
        seen_iterations.add(iteration)
        ledger = [row for row in launches if row.get('iteration') == iteration]
        expected = Counter(row['entry'] for row in ledger)
        if not expected or dict(expected) != audit['kernel_counts']:
            raise ValueError('profile distribution launch counts differ from audit')
        for row in ledger:
            name = row['entry']
            if name not in resources:
                raise ValueError(f'missing kernel resource contract: {name}')
            identity = {key: row[key] for key in ('entry','cubin_sha256','text_sha256','launch')}
            identity['signature'] = resources[name]['signature']
            digest = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if name in groups and groups[name]['contract_sha256'] != digest:
                raise ValueError(f'kernel contract changed during profile: {name}')
            if name not in groups:
                groups[name] = {'entry':name,'contract_sha256':digest,'contract':identity,
                                'resources':resources[name], 'samples':[]}
        for name,count in expected.items():
            durations = audit['kernel_duration_us'].get(name, [])
            if len(durations) != count or any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in durations):
                raise ValueError(f'incomplete kernel activity durations: {name}')
            groups[name]['samples'].append({'iteration':iteration,'duration_us':list(durations),'launches':count})
    rows = []
    for name,row in sorted(groups.items()):
        values = [value for sample in row['samples'] for value in sample['duration_us']]
        ordered = sorted(values)
        def percentile(fraction):
            at = (len(ordered)-1)*fraction
            lower = math.floor(at); upper = math.ceil(at)
            return ordered[lower]+(ordered[upper]-ordered[lower])*(at-lower)
        rows.append({**row,'launch_count':len(values),'profiled_invocations':len(row['samples']),
                     'repeated_invocations':len(row['samples']) > 1,
                     'summary_us':{'median':statistics.median(values),'min':min(values),'max':max(values),
                                   'p10':percentile(.1),'p90':percentile(.9),'iqr':percentile(.75)-percentile(.25)}})
    return {'schema':1,'kernels':rows,'profiled_iterations':sorted(seen_iterations),
            'compute_launches':sum(row['launch_count'] for row in rows),
            'scope':'audited Kineto kernel activities on existing iterations, profiler overhead included; changing values/state may affect durations; no profiler-free per-kernel speed claim'}
