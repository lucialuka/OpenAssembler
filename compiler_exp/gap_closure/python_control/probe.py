"""Python/PTX/all native opt-levels gate for structured scalar device helpers."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from bench_build import listing
from openassembler.build import generate
from openassembler.emu.machine import Launch as SassLaunch, Machine, Memory
from openassembler.emu.program import load_listing
from openassembler.ptx.interp import Launch as PTXLaunch, run_kernel
from openassembler.ptx.parse import parse_module
from openassembler.pyfront import compile_file, NotSupported

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / 'fixture.py'


def sha(data):return hashlib.sha256(data).hexdigest()


def run(compiler):
    text = FIXTURE.read_text(); tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for fn in functions:fn.decorator_list = []
    env = {'tid_x': lambda: thread}
    exec(compile(ast.Module(body=[tree.body[0], *functions], type_ignores=[]), str(FIXTURE), 'exec'), env)
    arrays = {'a': [-20, -7, -1, 0, 1, 5, 12, 30, 100, -90, 7, 3, -4, 6, -8, 40],
              'b': [-4.0, -1.5, -0.0, 0.0, 0.25, 1.0, 4.0, 16.0, -8.0, 2.0, -2.0, 0.5, 3.0, 6.0, 8.0, 10.0],
              'out': [-999] * 16, 'fo': [-999.0] * 16}
    expected = {k: v[:] for k, v in arrays.items()}
    for thread in range(16):env['evaluate'](*(expected[n] for n in arrays), 14)
    ptx = compiler(FIXTURE); module = parse_module(ptx)
    addresses = {name: 0x10000 + 0x1000 * i for i, name in enumerate(arrays)}
    def fresh():
        memory = Memory()
        for name, values in arrays.items():
            memory.write_bytes(addresses[name], struct.pack('<16' + ('f' if name in ('b','fo') else 'i'), *values))
        return memory
    def read(memory):
        return {name: list(struct.unpack('<16' + ('f' if name in ('b','fo') else 'i'), b''.join(struct.pack('<I', memory.read(addresses[name] + 4 * i, 4)) for i in range(16)))) for name in arrays}
    interpreted = fresh(); run_kernel(module, 'evaluate', PTXLaunch(block=(16,1,1),params={**addresses,'n':14}), interpreted)
    assert read(interpreted) == expected, (read(interpreted), expected)
    blob = b''.join(struct.pack('<Q', addresses[name]) for name in arrays) + struct.pack('<i',14)
    native=[]
    for level in ('safe','mined','sched','opt'):
        code, labels, words, exits, registers = generate(module.kernels['evaluate'],'sm_80',level)
        memory=fresh(); Machine(load_listing(listing(code,labels)),SassLaunch(block=(16,1,1),params=blob,param_base=0x160),memory).run()
        assert read(memory) == expected, (level,read(memory),expected)
        native.append({'level':level,'words':len(words),'registers':registers,'native_text_sha256':sha(b''.join(w.to_bytes(16,'little') for w in words)),'passed':True})
    return {'status':'passed','python':platform.python_version(),'platform':platform.platform(),
            'source_sha256':sha(text.encode()),'ptx_sha256':sha(ptx.encode()),'inputs':arrays,'expected':expected,
            'threads':16,'active_threads':14,'compared_elements_per_leg':64,'input_and_tail_immutability':True,
            'native':native,'hardware':'not_run','performance':'not_measured'}


if __name__ == '__main__':
    if '--production' in sys.argv:
        result={'production':run(lambda path:compile_file(path)[0])}
        result['compiler_sha256']={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in (ROOT/'openassembler/pyfront').glob('*.py')}
        target=HERE/'production-result.json'
    else:
        from candidate import compile_candidate
        try:compile_file(FIXTURE)
        except NotSupported as exc:baseline={'status':'rejected','diagnostic':str(exc)}
        else:baseline={'status':'accepted'}
        result={'baseline':baseline,'candidate':run(compile_candidate)};target=HERE/'candidate-result.json'
        if target.exists():
            from datetime import datetime, timezone
            target=HERE/('candidate-rerun-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
