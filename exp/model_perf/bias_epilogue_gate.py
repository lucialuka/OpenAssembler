"""Compare complete fused/unfused product outputs under direct and graph replay."""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import torch

from exp.ml_suite.common import guarded, guards, launch
from exp.model_perf.campaign import environment
from exp.model_perf.native_pack_gate import configure_device
from openassembler.gpu.cuda_driver import Driver
from openassembler.modelfront import kernels_tensor, tensor_epilogue


def run():
    configure_device()
    driver = Driver()
    root = Path('results')
    records, inventories = [], []
    for m, n, k in ((128, 128, 128), (128, 256, 128), (128, 128, 256)):
        specs = [{'shape': [m, k]}, {'shape': [n, k]}, {'shape': [n]}]
        out_spec = {'shape': [m, n]}
        fused = kernels_tensor.emit_tensor_recipe('linear', specs, out_spec, {'math_mode': kernels_tensor.MATH_MODE})
        with patch.object(tensor_epilogue, 'fuse_bias', lambda recipe: recipe):
            unfused = kernels_tensor.emit_tensor_recipe('linear', specs, out_spec, {'math_mode': kernels_tensor.MATH_MODE})
        directory = root / f'{m}-{n}-{k}'
        inventory = kernels_tensor.build_tensor_kernels(directory, [fused, unfused])
        inventories.append(inventory)
        modules, functions = [], {}
        for name, entry in inventory.items():
            module = driver.load((directory / entry['module']).read_bytes())
            modules.append(module)
            functions[name] = driver.function(module, name)
        backings, inputs = [], []
        for shape, dtype in [((m, k), torch.float16), ((n, k), torch.float16), ((n,), torch.float32),
                             ((m, n), torch.float32), ((m, n), torch.float32)]:
            backing, value = guarded(driver, shape, dtype)
            backings.append(backing)
            inputs.append(value)
        a, b, bias, output, control = inputs
        fused_stage = fused.stages[-1]
        product, separate = unfused.stages[-2:]

        def execute():
            launch(driver, functions[fused_stage.name], fused_stage.launch, [a, b, bias, output])
            launch(driver, functions[product.name], product.launch, [a, b, control])
            launch(driver, functions[separate.name], separate.launch, [control, bias, control])

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            execute()
        for pattern in ('zero', 'integer', 'normal', 'cancellation', 'offset', 'bias_edges'):
            for replay in range(3):
                generator = torch.Generator().manual_seed(921 + replay)
                left = torch.randn(m, k, generator=generator) * .2
                right = torch.randn(n, k, generator=generator) * .2
                addition = torch.randn(n, generator=generator) * .2
                if pattern == 'zero':
                    left.zero_(); right.zero_()
                elif pattern == 'integer':
                    left = torch.randint(-2, 3, (m, k), generator=generator).float()
                    right = torch.randint(-2, 3, (n, k), generator=generator).float()
                elif pattern == 'cancellation':
                    left[:, 1::2] = -left[:, ::2]
                    right[:, 1::2] = right[:, ::2]
                    addition *= .00001
                elif pattern == 'offset':
                    # Dyadic offsets admit an exact independent FP32 sum even
                    # after subtracting the large common bias.
                    left = (left*8).round()/8 + 4
                    right = (right*8).round()/8 + 4
                    addition -= 16 * k
                elif pattern == 'bias_edges':
                    addition[::4] = float('inf')
                    addition[1::4] = -float('inf')
                    addition[2::4] = float('nan')
                left, right = left.half(), right.half()
                a.copy_(left); b.copy_(right); bias.copy_(addition)
                driver.memset_async(output.data_ptr(), 0xff, output.numel()*4, torch.cuda.current_stream().cuda_stream)
                driver.memset_async(control.data_ptr(), 0xff, control.numel()*4, torch.cuda.current_stream().cuda_stream)
                if replay:
                    graph.replay()
                else:
                    execute()
                actual, prior = output.cpu(), control.cpu()
                expected = left.float() @ right.float().T + addition
                torch.testing.assert_close(actual, prior, rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(actual, expected, rtol=.002, atol=2e-5, equal_nan=True)
                assert torch.equal(a.cpu(), left) and torch.equal(b.cpu(), right)
                torch.testing.assert_close(bias.cpu(), addition, rtol=0, atol=0, equal_nan=True)
                guards(driver, backings)
                records.append({'shape': [m, n, k], 'pattern': pattern, 'replay': replay,
                                'elements': actual.numel(), 'actual_sha256': sha256(actual.numpy().tobytes()).hexdigest(),
                                'control_sha256': sha256(prior.numpy().tobytes()).hexdigest()})
                if pattern == 'normal' and replay == 1:
                    torch.save({'actual': actual, 'control': prior, 'expected': expected}, directory/'complete-state.pt')
        torch.cuda.synchronize()
        for module in modules:
            driver.unload(module)
    (root/'result.json').write_text(json.dumps({'passed': True, 'environment': environment(),
                                               'records': records, 'inventories': inventories}, indent=2))
