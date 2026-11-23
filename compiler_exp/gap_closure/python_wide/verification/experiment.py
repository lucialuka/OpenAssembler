"""Retain typed-verifier positives and independent corruption/bounds controls."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tests.test_verify_typed import compiled
import openassembler.verify as verifier


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('baseline_typed_verifier', HERE / 'baseline-verify.py')
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    text, contract, module, code, labels = compiled('u64')
    try:
        baseline.run_and_diff(module, 'typed', code, labels, n_elems=19, seed=3, parameter_contract=contract)
        before = {'status': 'unexpectedly-accepted'}
    except Exception as exc:
        before = {'status': 'rejected', 'exception': type(exc).__name__, 'reason': str(exc)}
    observed = []
    original = baseline.run_kernel
    def observe(mod, name, launch, memory):
        observed.append(dict(launch.params))
        return original(mod, name, launch, memory)
    baseline.run_kernel = observe
    legacy = baseline.run_and_diff(module, 'typed', code, labels, n_elems=19, seed=3)
    baseline.run_kernel = original
    before.update({'legacy_mismatches': [asdict(m) for m in legacy],
                   'legacy_ptx_parameter_values': observed,
                   'typed_expected_u64_scalar': verifier._typed_bits(random.Random(3), 'u64', 3),
                   'legacy_observation': 'u64 data scalar was guessed to be a pointer; no typed coverage claim'})
    positives = []
    for dtype in ('f32', 's32', 'u32', 's64', 'u64'):
        text, contract, module, code, labels = compiled(dtype)
        for seed in (0, 3, 4, 6):
            bad = verifier.run_and_diff(module, 'typed', code, labels, n_elems=19, seed=seed, parameter_contract=contract)
            positives.append({'dtype': dtype, 'seed': seed, 'passed': not bad, 'source': text,
                              'contract': contract, 'mismatches': [asdict(m) for m in bad]})
    _, contract, module, code, labels = compiled('u64')
    broken = [(text.replace('LDG.E.64', 'LDG.E'), ctl) for text, ctl in code]
    high = verifier.run_and_diff(module, 'typed', broken, labels, n_elems=19, parameter_contract=contract)
    _, contract, module, code, labels = compiled('s64', offset_store=True)
    guards = verifier.run_and_diff(module, 'typed', code, labels, n_elems=8, parameter_contract=contract)
    files = ['openassembler/verify.py', 'tests/test_verify_typed.py', 'tests/test_verify.py',
             'compiler_exp/gap_closure/python_wide/verification/experiment.py',
             'compiler_exp/gap_closure/python_wide/verification/baseline-verify.py']
    report = {'schema': 1, 'python': sys.version,
              'scope': 'Offline metadata/ABI/element-width checker experiment; no hardware or general memory-safety claim',
              'baseline': before, 'positive_results': positives,
              'high_word_corruption': {'rejected': bool(high), 'mismatches': [asdict(m) for m in high]},
              'matching_out_of_bounds_write': {'rejected': bool(guards), 'mismatches': [asdict(m) for m in guards]},
              'source_hashes': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in files}}
    report['passed'] = all(r['passed'] for r in positives) and bool(high) and bool(guards)
    output = HERE / 'typed-verifier-20261121.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output, 'passed', report['passed'])
