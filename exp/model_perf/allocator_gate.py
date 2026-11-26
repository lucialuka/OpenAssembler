"""Compare allocator binaries, resources and compile cost in paired samples."""
from hashlib import sha256
import json
import inspect
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from exp.model_perf.allocator_candidate import allocate_kernel, intervals, original_intervals
from openassembler import build
from openassembler.backend import ra, spill
from openassembler.backend.storage import LocalFrame
from openassembler.ptx.parse import parse_module


def impossible():
    stack = ra.V(1)
    ops = [ra.Op('MOV {d0}, c[0x0][0x28]', [stack], [], side=True)]
    groups = [tuple(ra.V(2 + 4*i + j) for j in range(4)) for i in range(4)]
    ops.extend(ra.Op('wide define {d0}', [group], [], var_lat=True) for group in groups)
    ops.append(ra.Op('wide consume', [], groups, side=True))
    selector = SimpleNamespace(k=SimpleNamespace(directives={'maxnreg': (16,)}),
                               r1=stack, local_frame=LocalFrame({}, {}, 0))
    return spill.allocate_kernel(selector, ops)


def sample(kernels, install, interval_pass, level):
    with patch.object(spill, 'allocate_kernel', install), patch.object(ra, '_intervals', interval_pass):
        started = perf_counter()
        records = []
        try:
            if kernels is None:
                impossible()
                raise AssertionError('impossible simultaneous groups were accepted')
            for kernel in kernels:
                resources = {}
                code, labels, words, exits, registers = build.generate(kernel, 'sm_80', level, resource_info=resources)
                records.append((code, labels, words, exits, registers, resources))
        except ra.NotSupported as error:
            if kernels is not None:
                raise
            records.append(str(error))
        elapsed = perf_counter() - started
    return elapsed, sha256(repr(records).encode()).hexdigest()


def run(*, samples, baseline, candidate, baseline_intervals, candidate_intervals, focused=False):
    paths = ('pressure_fixtures.json', 'spill_group_fixtures.json', 'spill_fixtures.json')
    cases = {case['name']: case['source'] for path in paths
             for case in json.loads((Path('tests/data') / path).read_text())}
    cases['no_pressure'] = ('.version 8.7\n.target sm_80\n.address_size 64\n'
                           '.entry simple(.param .u64 out) { .reg .u64 %p;'
                           'ld.param.u64 %p,[out]; st.global.u32 [%p],1; ret; }')
    cases['impossible_groups'] = None
    timed = {'no_pressure', 'scalar96', 'pred24', 'group4_auto', 'impossible_groups'}
    arms = {'baseline': (baseline, baseline_intervals), 'candidate': (candidate, candidate_intervals)}
    rng = random.Random(617)
    results = []
    build.encoder('sm_80')
    for name, source in cases.items():
        if focused and name not in timed:
            continue
        kernels = list(parse_module(source).kernels.values()) if source else None
        for level in ('opt',) if focused else ('safe', 'opt'):
            timings, hashes = {key: [] for key in arms}, set()
            orders = []
            for _ in range(samples if name in timed and level == 'opt' else 1):
                order = list(arms)
                rng.shuffle(order)
                orders.append(order)
                for key in order:
                    seconds, fingerprint = sample(kernels, *arms[key], level)
                    timings[key].append(seconds)
                    hashes.add(fingerprint)
            assert len(hashes) == 1, (name, level, hashes)
            results.append({'case': name, 'level': level, 'orders': orders, 'seconds': timings,
                            'median_seconds': {key: statistics.median(v) for key, v in timings.items()},
                            'source_sha256': sha256(source.encode()).hexdigest() if source else None,
                            'output_sha256': hashes.pop()})
            print(name, level, results[-1]['median_seconds'], file=sys.stderr, flush=True)
    return {'passed': True, 'python': sys.version, 'platform': platform.platform(), 'cases': results,
            'allocator_source_sha256': sha256(Path(ra.__file__).read_bytes()).hexdigest(),
            'baseline_intervals_sha256': sha256(inspect.getsource(baseline_intervals).encode()).hexdigest(),
            'candidate_intervals_sha256': sha256(inspect.getsource(candidate_intervals).encode()).hexdigest(),
            'scope': 'Uninstrumented generation with parsed PTX and warm encoder, host work may run concurrently'}


def cold_child(name, baseline):
    source = next(case['source'] for file in ('pressure_fixtures.json', 'spill_group_fixtures.json')
                  for case in json.loads((Path('tests/data') / file).read_text()) if case['name'] == name)
    with patch.object(ra, '_intervals', original_intervals if baseline else ra._intervals):
        started = perf_counter()
        kernel = next(iter(parse_module(source).kernels.values()))
        resources = {}
        result = build.generate(kernel, 'sm_80', 'opt', resource_info=resources)
        seconds = perf_counter() - started
    print(json.dumps({'cold_generate_seconds': seconds, 'native_sha256': sha256(repr((result, resources)).encode()).hexdigest()}))


def cold_samples():
    rows = []
    for name in ('scalar96', 'group4_auto'):
        results = {}
        for arm in ('production', 'baseline'):
            started = perf_counter()
            child = subprocess.run([sys.executable, '-m', 'exp.model_perf.allocator_gate',
                                    '--cold-child', name, '--arm', arm], capture_output=True, text=True, check=True)
            results[arm] = {**json.loads(child.stdout), 'process_seconds': perf_counter() - started}
        assert results['production']['native_sha256'] == results['baseline']['native_sha256']
        rows.append({'case': name, 'arms': results})
    return {'passed': True, 'cases': rows, 'scope': 'Fresh subprocess and encoder per arm, one sample each',
            'platform': platform.platform(), 'python': sys.version}


if __name__ == '__main__':
    from argparse import ArgumentParser
    from exp.model_perf.allocator_candidate import baseline

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--linear-intervals-only', action='store_true')
    mode.add_argument('--integrated', action='store_true')
    parser.add_argument('--focused', action='store_true')
    parser.add_argument('--cold', action='store_true')
    parser.add_argument('--cold-child')
    parser.add_argument('--arm', choices=('baseline', 'production'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cold_child:
        cold_child(args.cold_child, args.arm == 'baseline')
        sys.exit(0)
    if args.output is None:
        parser.error('--output is required')
    if args.cold:
        args.output.write_text(json.dumps(cold_samples(), indent=2))
        sys.exit(0)
    args.output.write_text(json.dumps(run(samples=args.samples, baseline=baseline,
                                         candidate=baseline if args.linear_intervals_only or args.integrated else allocate_kernel,
                                         baseline_intervals=original_intervals,
                                         candidate_intervals=(ra._intervals if args.integrated else
                                                              intervals if args.linear_intervals_only else original_intervals),
                                         focused=args.focused), indent=2))
