"""Independent PyTorch controls for emitted convolution kernels and poisoned SASS."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from openassembler import build
from openassembler.modelfront.kernels import emit_kernel
from openassembler.ptx.interp import Interp, Launch as PtxLaunch
from openassembler.ptx.parse import parse_module
from openassembler.emu.machine import Machine, Memory, Launch
from openassembler.emu.program import load_listing
from tests.backend_harness import listing

CASES = [
    ((1, 2, 4, 5), (3, 2, 3, 3), (1, 1), (1, 1), (1, 1)),
    ((2, 2, 5, 6), (2, 2, 2, 3), (2, 1), (1, 0), (1, 1)),
    ((1, 1, 5, 5), (2, 1, 2, 2), (1, 2), (1, 1), (2, 1)),
    ((1, 3, 3, 4), (2, 3, 1, 1), (1, 1), (0, 0), (1, 1)),
]

def packed(tensor):
    values = tensor.detach().flatten().tolist()
    return struct.pack('<' + 'f' * len(values), *values)


def run():
    torch.set_num_threads(1)
    torch.manual_seed(73571)
    rows = []
    guard = b'\xa5' * 64
    for case_index, (xshape, wshape, stride, padding, dilation) in enumerate(CASES):
        x, w = torch.randn(xshape) * .25, torch.randn(wshape) * .25
        bias = torch.randn(wshape[0]) * .125
        y = torch.nn.functional.conv2d(x, w, bias, stride, padding, dilation)
        dy = torch.randn_like(y) * .25
        dx = torch.nn.grad.conv2d_input(xshape, w, dy, stride, padding, dilation)
        dw = torch.nn.grad.conv2d_weight(x, wshape, dy, stride, padding, dilation)
        attrs = dict(stride=list(stride), padding=list(padding), dilation=list(dilation))
        for op, tensors, expected in [('conv2d', [x, w, bias], y), ('conv2d_dx', [dy, w], dx), ('conv2d_dw', [dy, x], dw)]:
            kernel = emit_kernel(op, [dict(shape=list(t.shape)) for t in tensors], dict(shape=list(expected.shape)), attrs)
            module = parse_module(kernel.ptx)
            arrays = [packed(t) for t in tensors]
            pointers = [0x100000000 + 0x100000 * i for i in range(len(arrays) + 1)]
            params = dict(zip([*(f'a{i}' for i in range(len(arrays))), 'output'], pointers))
            for level in ('safe', 'mined', 'sched', 'opt'):
                code, labels, words, _, registers = build.generate(module.kernels[kernel.name], 'sm_80', level)
                program = load_listing(listing(code, labels))
                for engine in ('ptx', 'normal', 'poisoned'):
                    memory = Memory()
                    for address, data in zip(pointers, arrays + [bytes([0xd3]) * (expected.numel() * 4)]):
                        memory.write_bytes(address - 64, guard + data + guard)
                    for cx in range(kernel.grid[0]):
                        if engine == 'ptx':
                            Interp(module, kernel.name, PtxLaunch(grid=kernel.grid, block=(128,1,1), ctaid=(cx,0,0), params=params), memory).run()
                        else:
                            Machine(program, Launch(grid=kernel.grid, block=(128,1,1), ctaid=(cx,0,0), params=struct.pack('<'+'Q'*len(pointers), *pointers), poison=0xdeadbeef if engine=='poisoned' else None), memory).run()
                    for address, data in zip(pointers, arrays):
                        assert bytes(memory.read(address-64+i,1) for i in range(len(data)+128)) == guard+data+guard
                    output = bytes(memory.read(pointers[-1]+i,1) for i in range(expected.numel()*4))
                    actual = torch.tensor(struct.unpack('<'+'f'*expected.numel(), output)).reshape(expected.shape)
                    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=5e-6)
                    for offset in (-64, expected.numel()*4):
                        assert bytes(memory.read(pointers[-1]+offset+i,1) for i in range(64)) == guard
                    rows.append(dict(case=case_index, op=op, level=level, engine=engine, elements=expected.numel(), registers=registers, passed=True, max_abs=float((actual-expected).abs().max()), words_sha256=hashlib.sha256(b''.join(word.to_bytes(16,'little') for word in words)).hexdigest()))
                print(case_index, op, level, 'passed', flush=True)
    sources = [ROOT / ('openassembler/modelfront/'+name) for name in ('graph.py','kernels.py','kernels_conv.py','reference.py')]
    result = dict(schema=1, passed=True, seed=73571, rtol=4e-5, atol=5e-6, results=rows, source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'offline.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Passed', len(rows), 'complete tensor executions')

if __name__ == '__main__':
    run()
