"""Probe default-stream memsets and native runtime buffer ownership."""
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from exp.model_perf.campaign import environment
from openassembler.gpu.cuda_driver import Driver


from openassembler.modelfront.runtime import Runtime


def legacy_poison(self):
    self.synchronize()
    with self._context_scope():
        for tensor in [*self.buffers.values(), *(v for x in self._training.values() for v in x.values())]:
            self._driver.memset(tensor.data_ptr(), 0xff, tensor.numel()*tensor.element_size())
        for tensor in self._tensor_workspaces.values():
            self._driver.memset(tensor.data_ptr(), 0xff, tensor.numel()*tensor.element_size())


def runtime_gate():
    from exp.ml_suite.tasks import create
    from openassembler.modelfront.graph import capture_model, extract_weights
    from openassembler.modelfront.kernels import build_bundle
    from openassembler.modelfront.optimization import optimize_plan
    from openassembler.modelfront.reference import CPUReference
    root = Path('results')
    torch.set_num_threads(1)
    torch.manual_seed(941)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    model = create('dense_regression_fp16')
    plan = optimize_plan(capture_model(model, [torch.randn(128, 128)*.25], math_mode='f16_rne_f32_accum'),
                         storage_views=False, reuse_storage=True)
    weights = extract_weights(model, plan)
    build_bundle(plan, weights, root/'bundle', sass_listings=True, cache=False)
    source = torch.randn(128, 128)*.25
    x = source.cuda()
    inputs = {plan['inputs'][0]['name']: x}
    expected = CPUReference(plan, weights).forward({plan['inputs'][0]['name']: source})
    records = []
    for arm in ('legacy', 'ordered'):
        runtime = Runtime.load(root/'bundle')
        runtime.forward(inputs)
        torch.cuda.synchronize()
        with ExitStack() as stack:
            if arm == 'legacy':
                stack.enter_context(patch.object(Runtime, 'poison_workspace', legacy_poison))
            for i in range(3):
                with torch.cuda.stream(torch.cuda.default_stream()):
                    torch.cuda._sleep(100000000)
                runtime.poison_workspace()
                out = runtime.forward(inputs)
                runtime.synchronize()
                torch.cuda.default_stream().synchronize()
                observed = out[plan['outputs'][0]].cpu()
                wanted = expected[plan['outputs'][0]]
                close = torch.isclose(observed, wanted, rtol=.002, atol=2e-5)
                records.append({'arm': arm, 'iteration': i, 'mismatches': int((~close).sum()),
                                'nonfinite': int((~torch.isfinite(observed)).sum())})
                if arm == 'ordered':
                    assert bool(close.all()), records[-1]
                else:
                    assert not bool(close.all()), 'legacy race did not reproduce'
        runtime.close()
    result = {'passed': True, 'environment': environment(), 'records': records,
              'scope': 'Delayed legacy stream reproduces poison-after-forward corruption, ordered poison repairs it'}
    (root/'result.json').write_text(json.dumps(result, indent=2))


def run():
    torch.empty(1, device='cuda')
    driver = Driver()
    default = torch.cuda.default_stream()
    independent = torch.cuda.Stream()
    records = []
    for size in (16384, 65536, 1048576):
        data = torch.zeros(size, dtype=torch.uint8, device='cuda')
        observed = torch.empty_like(data)
        torch.cuda.synchronize()
        with torch.cuda.stream(default):
            torch.cuda._sleep(100000000)
        started = perf_counter()
        driver.memset(data.data_ptr(), 0x7f, size)
        host_ms = 1000 * (perf_counter() - started)
        with torch.cuda.stream(independent):
            observed.copy_(data)
        independent.synchronize()
        before_default = observed.cpu()
        default.synchronize()
        records.append({'bytes': size, 'memset_host_ms': host_ms,
                        'seen_before_default_completion': int((before_default == 0x7f).sum()),
                        'final_matches': bool((data.cpu() == 0x7f).all())})
    result = {'passed': True, 'environment': environment(), 'records': records,
              'scope': 'Delayed legacy stream followed by driver memset and an independent-stream observer'}
    Path('results/result.json').write_text(json.dumps(result, indent=2))
