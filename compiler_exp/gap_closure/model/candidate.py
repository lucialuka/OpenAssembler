"""Independent G25 normalization candidate, before native lowering promotion.

The candidate materializes contiguous broadcast operands, flattens their batch
axes, and performs equal-batch BMM. It compares arbitrary output seeds and both
input derivatives with eager torch.matmul; it also independently enumerates the
flat source index used by the existing broadcast kernel generator.
"""
import hashlib
import json
import math
import platform
from pathlib import Path

CASES = [
    ([2, 3], [3, 5]),
    ([2, 3], [4, 3, 5]),
    ([4, 2, 3], [3, 5]),
    ([1, 2, 3], [4, 3, 5]),
    ([4, 2, 3], [1, 3, 5]),
    ([2, 1, 2, 3], [1, 4, 3, 5]),
    ([2, 4, 2, 3], [4, 3, 5]),
    ([2, 3], [2, 4, 3, 5]),
]


def normalized(a, b):
    import torch
    batches = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    left = a.expand(*batches, *a.shape[-2:]).contiguous()
    right = b.expand(*batches, *b.shape[-2:]).contiguous()
    batch = math.prod(batches)
    return torch.bmm(left.reshape(batch, *a.shape[-2:]),
                     right.reshape(batch, *b.shape[-2:])).reshape(*batches, a.shape[-2], b.shape[-1])


def index(flat, full, target):
    padded = [1] * (len(full)-len(target)) + list(target)
    result = 0
    for axis, dim in enumerate(full):
        if padded[axis] != 1:
            coordinate = flat // math.prod(full[axis+1:]) % dim
            result += coordinate * math.prod(padded[axis+1:])
    return result


def run():
    import torch
    generator = torch.Generator().manual_seed(25051)
    rows = []
    for ashape, bshape in CASES:
        a = torch.randn(ashape, generator=generator).requires_grad_()
        b = torch.randn(bshape, generator=generator).requires_grad_()
        candidate = normalized(a, b)
        control = a @ b
        seed = torch.randn(control.shape, generator=generator)
        cg = torch.autograd.grad(candidate, (a, b), seed)
        rg = torch.autograd.grad(control, (a, b), seed)
        torch.testing.assert_close(candidate, control, rtol=1e-6, atol=1e-6)
        for actual, expected in zip(cg, rg):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        batch = list(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        elements = 0
        for shape in (ashape, bshape):
            full = batch + shape[-2:]
            source = torch.arange(math.prod(shape)).reshape(shape)
            expected = source.expand(full).reshape(-1).tolist()
            actual = [index(i, full, shape) for i in range(math.prod(full))]
            assert actual == expected
            elements += len(actual)
        rows.append({'left': ashape, 'right': bshape, 'output': list(control.shape),
                     'passed': True, 'index_elements': elements})
    return {'schema': 1, 'gap': 'G25', 'stage': 'candidate CPU algebra and independent source indexing',
            'seed': 25051, 'python': platform.python_version(), 'torch': torch.__version__,
            'candidate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'passed': True, 'hardware': 'not run', 'cases': rows}


if __name__ == '__main__':
    result = run()
    path = Path(__file__).with_name('candidate-results.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
