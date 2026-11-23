"""Offline independent Python/PTX/SASS correctness gate for G04 scalar helpers."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from candidate import compile_candidate
from bench_build import listing
from openassembler.backend import ra
from openassembler.emu.machine import Launch as SassLaunch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.isa.encode import Encoder
from openassembler.ptx.interp import Launch as PTXLaunch, run_kernel
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file, NotSupported

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'helpers_fixture.py'


def run_gate(compiler):
    text = FIXTURE.read_text()
    tree = ast.parse(text)
    # Execute only undecorated function bodies for the independent Python leg.
    # This is deliberate reference execution; the compiler never executes source.
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for fn in functions:
        fn.decorator_list = []
    ref_tree = ast.Module(body=[tree.body[1], *functions], type_ignores=[])
    env = {'tid_x': lambda: thread}
    exec(compile(ref_tree, str(FIXTURE), 'exec'), env)
    arrays = {'x': [-2.0, -0.5, 0.0, 0.25, 1.0, 2.5, 7.0, 13.0],
              'ints': [-100, -7, -1, 0, 1, 2, 7, 100],
              'uints': [0, 1, 2, 3, 0xffffffff, 0x80000000, 0xfffffffe, 0x7fffffff],
              'out': [-999.0] * 8, 'io': [-999] * 8, 'uo': [0xdeadbeef] * 8}
    expected = {k: v[:] for k, v in arrays.items()}
    for thread in range(8):
        env['evaluate'](*(expected[n] for n in arrays), 6)
    # Python integers are unbounded; the declared u32 device type wraps.
    expected['uo'] = [x & 0xffffffff for x in expected['uo']]
    ptx = compiler(FIXTURE)
    module = parse_module(ptx)
    kernel = module.kernels['evaluate']
    addresses = {name: 0x10000 + 0x1000 * i for i, name in enumerate(arrays)}
    types = {name: 'f' if name in ('x', 'out') else 'I' if name in ('uints', 'uo') else 'i'
             for name in arrays}
    def fresh():
        mem = Memory()
        for name, values in arrays.items():
            mem.write_bytes(addresses[name], struct.pack('<8' + types[name], *values))
        return mem
    def read(mem):
        return {name: list(struct.unpack('<8' + types[name], b''.join(
            struct.pack('<I', mem.read(addresses[name] + 4 * i, 4)) for i in range(8))))
                for name in arrays}
    interp = fresh()
    run_kernel(module, 'evaluate', PTXLaunch(block=(8, 1, 1), params={**addresses, 'n': 6}), interp)
    assert read(interp) == expected, (read(interp), expected)
    code, state = ra.compile_kernel(kernel)
    words = ra.encode_code(Encoder('sm_80'), code, state['labels'])
    sass = listing(code, state['labels'])
    machine_mem = fresh()
    blob = b''.join(struct.pack('<Q', addresses[name]) for name in arrays) + struct.pack('<i', 6)
    Machine(load_listing(sass), SassLaunch(block=(8, 1, 1), params=blob, param_base=0x160), machine_mem).run()
    assert read(machine_mem) == expected, (read(machine_mem), expected)
    return {'status': 'passed', 'reference': 'independent Python bodies plus explicit u32 wrap',
            'threads': 8, 'active_threads': 6, 'compared_elements_per_leg': 48,
            'legs': ['Python', 'PTX interpreter', 'encoded SASS emulator'],
            'tail_sentinels_unchanged': True,
            'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
            'inputs': arrays, 'expected_outputs': expected,
            'encoded_word_count': len(words), 'hardware': 'not_run', 'performance': 'not_measured'}


def main():
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    if '--production' in sys.argv:
        result['production'] = run_gate(lambda path: compile_file(path)[0])
        result['compiler_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (ROOT / 'openassembler/pyfront').glob('*.py')}
        (HERE / 'production-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    try:
        compile_file(FIXTURE)
    except NotSupported as exc:
        result['baseline'] = {'status': 'rejected', 'diagnostic': str(exc)}
    else:
        result['baseline'] = {'status': 'accepted'}
    result['candidate'] = run_gate(compile_candidate)
    # Preserve the original baseline rejection recorded before promotion.
    (HERE / 'candidate-rerun.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
