"""A100 convolution and CNN training checks against independent eager CPU Torch."""
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch

from openassembler.modelfront.graph import capture_model, extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.profiling import audit_trace
from openassembler.modelfront.runtime import Runtime


class ConvModel(torch.nn.Module):
    def __init__(self, channels, outputs, size, stride, padding, dilation, bias):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, outputs, size, stride=stride, padding=padding,
                                    dilation=dilation, bias=bias)

    def forward(self, x):
        return self.conv(x)


def run(root):
    root = Path(root)
    root.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(73571)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    comparisons, runs, raw, cli = [], [], {}, []
    error = None

    def check(actual, expected, label):
        actual = actual.detach().cpu().clone()
        expected = expected.detach().cpu().clone()
        close = torch.isclose(actual, expected, rtol=4e-5, atol=5e-6)
        raw[label] = dict(actual=actual, expected=expected)
        row = dict(name=label, passed=bool(close.all()), elements=actual.numel(),
                   max_abs=float((actual-expected).abs().max()), mismatches=int((~close).sum()))
        comparisons.append(row)
        assert row['passed'], row

    def test(model, shape, label, views, reuse, level='opt'):
        model = copy.deepcopy(model)
        example = torch.randn(shape) * .25
        plan = optimize_plan(capture_model(model, (example,)), storage_views=views, reuse_storage=reuse)
        folder = root / label
        build_bundle(plan, extract_weights(model, plan), folder, sass_listings=False, opt_level=level, cache=False)
        runtime = Runtime.load(folder)
        try:
            for iteration in range(2):
                x = (example + iteration * .03125).detach().requires_grad_()
                gpu = {plan['inputs'][0]['name']: x.detach().cuda()}
                before = {name: value.cpu().clone() for name, value in gpu.items()}
                weights = {name: value.cpu().clone() for name, value in runtime.weights.items()}
                model.zero_grad(set_to_none=True)
                expected = model(x)
                seed = torch.randn_like(expected) * .25
                expected.backward(seed)
                if iteration:
                    runtime.poison_workspace()
                actual = runtime.forward(gpu, require_tape=True)
                gradients = runtime.backward({plan['outputs'][0]: seed.cuda()})
                check(actual[plan['outputs'][0]], expected, f'{label}/{iteration}/output')
                check(gradients['inputs'][plan['inputs'][0]['name']], x.grad, f'{label}/{iteration}/dx')
                for spec in plan['parameters']:
                    parameter = dict(model.named_parameters())[spec['target']]
                    check(gradients['parameters'][spec['name']], parameter.grad,
                          f'{label}/{iteration}/gradient/{spec["target"]}')
                assert all(torch.equal(value.cpu(), before[name]) for name, value in gpu.items())
                assert all(torch.equal(value.cpu(), weights[name]) for name, value in runtime.weights.items())
                runs.append(dict(name=label, iteration=iteration, guards=runtime.check_canaries(),
                                 inputs_weights_immutable=True, launches=list(runtime.launch_ledger)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                runtime.forward(gpu, require_tape=True)
                runtime.backward({plan['outputs'][0]: seed.cuda()})
                runtime.synchronize()
            trace = root / (label + '-trace.json')
            profiler.export_chrome_trace(str(trace))
            runs.append(dict(name=label, profiler=audit_trace(json.loads(trace.read_text()), runtime.launch_ledger)))
        finally:
            runtime.close()

    try:
        cases = [
            ((1, 2, 5, 6), 3, 3, 1, 1, 1, True),
            ((2, 2, 7, 8), 2, (2, 3), (2, 1), (1, 0), 1, False),
            ((1, 1, 7, 6), 2, 2, (1, 2), 1, (2, 1), True),
            ((2, 3, 4, 5), 2, 1, 1, 0, 1, False),
        ]
        for index, (shape, outputs, size, stride, padding, dilation, bias) in enumerate(cases):
            for views, reuse in ((False, False), (True, True)):
                test(ConvModel(shape[1], outputs, size, stride, padding, dilation, bias), shape,
                     f'conv-{index}-{int(views)}{int(reuse)}', views, reuse)
        test(ConvModel(2, 3, 3, 1, 1, 1, True), (1, 2, 5, 6), 'conv-safe', True, True, 'safe')
        sys.path.insert(0, str(Path('repo').resolve()))
        from model import create_model
        test(create_model(), (2, 2, 8, 8), 'cnn', True, True)
        for args in [
            ['compile', 'repo', '--backend', 'local', '--verify', '--no-sass', '--out-dir', 'cli-output'],
            ['run', 'cli-output', '--backend', 'local', '--verify'],
        ]:
            done = subprocess.run([sys.executable, '-m', 'openassembler', *args], capture_output=True,
                                  text=True, timeout=600)
            row = dict(args=args, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
            cli.append(row)
            assert done.returncode == 0, row
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        torch.save(raw, root / 'full-tensors.pt')
        package = Path(__import__('openassembler').__file__).parent
        report = dict(schema=1, passed=error is None and len(comparisons)==80 and len(cli)==2,
                      error=error, comparisons=comparisons, runs=runs, cli=cli,
                      cli_reports={p.name:json.loads(p.read_text()) for p in Path('cli-output').glob('*report.json')},
                      python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      device=str(torch.cuda.get_device_properties(0)), seed=73571, rtol=4e-5, atol=5e-6,
                      source_hashes={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in package.rglob('*.py')})
        # Each of nine isolated convolutions checks output, dX, dW and optional dB
        # twice. The two-layer classifier checks its six parameter gradients too.
        expected_count = 2 * (5 * 4 + 4 * 3 + 8)
        report['expected_comparisons'] = expected_count
        report['passed'] = error is None and len(comparisons)==expected_count and len(cli)==2
        (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
