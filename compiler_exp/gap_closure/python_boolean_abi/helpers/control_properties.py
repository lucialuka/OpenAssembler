"""Generated pure predicate helper expressions with independent recursive evaluation."""
import hashlib
import json
import random
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from candidate import install
install()
from openassembler.pyfront import compile_file
from openassembler import build
from openassembler.ptx.parse import parse_module
from openassembler.ptx.interp import Interp, Launch as PLaunch
from openassembler.emu.machine import Memory, Machine, Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

rng = random.Random(86173)


def tree(depth):
    if depth == 0 or rng.random() < 0.25:
        return [rng.choice(('first', 'second', 'true', 'false', 'truth', 'chain'))]
    op = rng.choice(('not', 'and', 'or'))
    return [op, tree(depth-1)] if op == 'not' else [op, tree(depth-1), tree(depth-1)]


def render(node):
    op = node[0]
    if op in ('first', 'second'): return op
    if op in ('true', 'false'): return op.title()
    if op == 'truth': return 'bool(x)'
    if op == 'chain': return '(-3 < x < 7)'
    if op == 'not': return '(not '+render(node[1])+')'
    return '('+render(node[1])+' '+op+' '+render(node[2])+')'


def evaluate(node, x):
    op = node[0]
    if op in ('first', 'false'): return False
    if op in ('second', 'true'): return True
    if op == 'truth': return bool(x)
    if op == 'chain': return -3 < x < 7
    if op == 'not': return not evaluate(node[1], x)
    left = evaluate(node[1], x)
    return (left and evaluate(node[2], x)) if op == 'and' else (left or evaluate(node[2], x))


if __name__ == '__main__':
    inputs = [-2147483648, -19, -3, -1, 0, 1, 3, 7, 19, 2147483647]
    rows = []
    ap, out, cp = 0x100010000, 0x100000000, 0x100020000
    guard = b'\xa5'*64
    before = guard+struct.pack('<10i', *inputs)+guard
    paths = [HERE/n for n in ('candidate-pyfront.py', 'candidate-helpers.py', 'candidate-modules.py', 'control_properties.py')]
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tempfile.TemporaryDirectory(prefix='oa-predicate-helper-properties-') as temporary:
        for index in range(128):
            expression = tree(3)
            source = ('from openassembler.pyfront import *\n'
                      '@device\ndef h(first:bool,second:bool,x:i32)->bool:\n    return '+render(expression)+'\n'
                      '@kernel\ndef k(a:Array(i32),out:Array(i32),count:Array(i32)):\n'
                      '    i=tid_x()\n    out[i]=i32(h(bool(atomic_add(count,i,1)),bool(atomic_add(count,i,1)),a[i]))\n')
            expected = [int(evaluate(expression, x)) for x in inputs]
            path = Path(temporary)/'source.py'
            path.write_text(source)
            text, _ = compile_file(path)
            module = parse_module(text)
            wanted = guard+struct.pack('<10i', *expected)+guard
            counter = guard+struct.pack('<10i', *([2]*10))+guard
            for level in ('safe', 'mined', 'sched', 'opt'):
                code, labels, words, _, regs = build.generate(module.kernels['k'], 'sm_80', level)
                for engine in ('ptx', 'normal', 'poisoned'):
                    memory = Memory()
                    memory.write_bytes(ap-64, before)
                    memory.write_bytes(out-64, guard+bytes(40)+guard)
                    memory.write_bytes(cp-64, guard+bytes(40)+guard)
                    if engine == 'ptx':
                        Interp(module, 'k', PLaunch(block=(10, 1, 1), params={'a': ap, 'out': out, 'count': cp}), memory).run()
                    else:
                        Machine(load_listing(listing(code, labels)), Launch(block=(10, 1, 1), params=struct.pack('<QQQ', ap, out, cp), poison=0xdeadbeef if engine == 'poisoned' else None), memory).run()
                    for address, data in ((ap, before), (out, wanted), (cp, counter)):
                        assert bytes(memory.read(address-64+j, 1) for j in range(len(data))) == data, (index, level, engine, address)
            rows.append(dict(index=index, tree=expression, source=source, expected_outputs=expected, expected_atomic_counts=[2]*10, passed=True))
    assert frozen == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(schema=1, passed=True, seed=86173, inputs=inputs, results=rows, executions=1536, source_hashes=frozen,
                  scope='128 original predicate helper expressions and independent recursive truth references, eager once-only atomic arguments, full output/counter/immutable input/guards at four PTX/normal/poisoned levels; no hardware claim')
    (HERE/'control-properties.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Passed 128 predicate helper trees; 1536 executions')
