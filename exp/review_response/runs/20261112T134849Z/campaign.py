"""One isolated model per process: full-state proof before paired timing."""
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import statistics
import sys
import time

import torch

from openassembler.modelfront.graph import capture_model, extract_weights
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.runtime import Runtime, build_bundle, load_weights
from openassembler.modelfront.reference import CPUReference
from exp.review_response.replay import TrainingReplay
from exp.review_response.audit import audit_profile
if os.environ.get('OA_REVIEW_PRODUCTION') == '1':
    from openassembler.modelfront.replay import TrainingReplay
    from openassembler.modelfront.profiling import audit_profile


def main(case):
    torch.manual_seed(271)
    torch.set_num_threads(1)
    shape, math_mode = {'native_model': ((17, 8), 'fp32'),
                       'native_attention_model': ((2, 7, 8), 'fp32'),
                       'native_tensor_model': ((128, 128), 'f16_rne_f32_accum')}[case]
    model = runpy.run_path('cases/' + case + '/model.py')['create_model']().train()
    sample = torch.randn(shape) * .25
    before = time.perf_counter()
    plan = optimize_plan(capture_model(model, [sample], math_mode=math_mode), reuse_storage=True)
    capture_ms = (time.perf_counter()-before)*1000
    weights = extract_weights(model, plan)
    root = Path('results') / case
    before = time.perf_counter()
    build_bundle(plan, weights, root)
    build_ms = (time.perf_counter()-before)*1000
    numeric = CPUReference(plan, weights) if math_mode != 'fp32' else None
    inputs = {plan['inputs'][0]['name']: sample.cuda()}
    target = (torch.randn(shape) * .25).cuda()
    direct = Runtime.load(root)
    before = time.perf_counter()
    owner = None
    if os.environ.get('OA_REVIEW_PRODUCTION') == '1':
        from openassembler.modelfront.autograd import CompiledModel
        owner = CompiledModel(plan, weights, root).cuda()
        replay = owner.prepare_native_training(inputs, target)
    elif os.environ.get('OA_REVIEW_LOSS') == '1':
        from contextlib import ExitStack
        from unittest.mock import patch
        from openassembler.modelfront import kernels, runtime
        from exp.review_response.ordered_loss import ordered_mse
        original = kernels.emit_kernel
        def candidate(op, *args, **kwargs):
            kernel = original(op, *args, **kwargs)
            return ordered_mse(kernel) if op == 'mse' else kernel
        with ExitStack() as stack:
            stack.enter_context(patch.object(kernels, 'emit_kernel', candidate))
            stack.enter_context(patch.object(runtime, 'emit_kernel', candidate))
            candidate_root = root/'candidate'
            build_bundle(plan, weights, candidate_root)
            replay = TrainingReplay(plan, load_weights(candidate_root, plan, torch), candidate_root, inputs, target)
    else:
        replay = TrainingReplay(plan, load_weights(root, plan, torch), root, inputs, target)
    prepare_ms = (time.perf_counter()-before)*1000
    assert all(torch.equal(replay.runtime.weights[k].cpu(), value) for k, value in weights.items()), 'capture updated parameters'
    checks, audits = [], {}
    half = math_mode != 'fp32'

    def flatten(result, runtime):
        return {'prediction': result['prediction'], 'loss': result['loss'].reshape(()),
                **{'dx/' + n: v for n, v in result['gradients']['inputs'].items()},
                **{'dw/' + n: v for n, v in result['gradients']['parameters'].items()},
                **{'w/' + n: v for n, v in runtime.weights.items()}}

    for iteration in range(4):
        x = (torch.randn(shape) * .25).requires_grad_()
        y = torch.randn(shape) * .25
        inputs[plan['inputs'][0]['name']].copy_(x.detach())
        target.copy_(y)
        if numeric is None:
            model.zero_grad(set_to_none=True)
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            state = dict(model.named_parameters())
            expected = {'prediction': output.detach(), 'loss': loss.detach(),
                        'dx/' + plan['inputs'][0]['name']: x.grad,
                        **{'dw/' + s['name']: state[s['target']].grad for s in plan['parameters']}}
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(parameter.grad, alpha=-.01)
            expected.update({'w/' + s['name']: state[s['target']] for s in plan['parameters']})
        else:
            result = numeric.train_step({plan['inputs'][0]['name']: x}, y, .01)
            expected = {'prediction': result['outputs'][plan['outputs'][0]], 'loss': result['loss'].reshape(()),
                        **{'dx/' + n: v for n, v in result['gradients']['inputs'].items()},
                        **{'dw/' + n: v for n, v in result['gradients']['parameters'].items()},
                        **{'w/' + n: v for n, v in numeric.weights.items()}}
        baseline = None
        for name in ['direct', 'replay']:
            runtime = direct if name == 'direct' else replay.runtime
            if iteration:
                runtime.poison_workspace()
            if iteration == 3:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    with torch.no_grad():
                        result = direct.train_step(inputs, target, .01) if name == 'direct' else replay.run()
                    runtime.synchronize()
                audits[name] = audit_profile(profile, runtime.launch_ledger if name == 'direct' else replay.launch_ledger)
            else:
                with torch.no_grad():
                    result = direct.train_step(inputs, target, .01) if name == 'direct' else replay.run()
                runtime.synchronize()
            actual = {n: t.detach().cpu() for n, t in flatten(result, runtime).items()}
            for field, got in actual.items():
                reference = expected[field].detach().cpu()
                good = torch.isclose(got, reference, rtol=.002 if half else 1e-4, atol=2e-5 if half else 1e-5)
                item = {'iteration': iteration, 'method': name, 'field': field, 'passed': bool(good.all()),
                        'mismatches': int((~good).sum()), 'max_abs': float((got-reference).abs().max())}
                checks.append(item)
                assert item['passed'], item
                if baseline is not None:
                    assert torch.equal(got, baseline[field]), (field, 'paired bit mismatch')
            if baseline is None:
                baseline = actual
            assert torch.equal(inputs[plan['inputs'][0]['name']].cpu(), x.detach())
            assert torch.equal(target.cpu(), y)
            runtime.check_canaries()
    # Adversarial ownership checks remain outside timing.
    guards = {}
    version = next(iter(replay.runtime.weights.values()))
    torch.autograd.graph.increment_version(version)
    try:
        replay.run()
    except RuntimeError:
        guards['parameter_mutation_rejected'] = True
    else:
        raise AssertionError('modified parameter admitted')
    replay._versions = {n: t._version for n, t in replay.runtime.weights.items()}
    if owner is not None:
        attr = owner._state_attributes[0]
        original_parameter = getattr(owner, attr)
        setattr(owner, attr, torch.nn.Parameter(original_parameter.detach()))
        try:
            replay.run()
        except RuntimeError:
            guards['parameter_replacement_rejected'] = True
        else:
            raise AssertionError('replacement parameter admitted')
        setattr(owner, attr, original_parameter)
    with torch.cuda.stream(torch.cuda.Stream()):
        try:
            replay.run()
        except ValueError:
            guards['stream_switch_rejected'] = True
        else:
            raise AssertionError('wrong stream admitted')
    functions = {'direct': lambda: direct.train_step(inputs, target, .01), 'replay': replay.run}
    for fn in functions.values():
        with torch.no_grad():
            for _ in range(4):
                fn()
    torch.cuda.synchronize()
    rng, rounds = random.Random(9281), []
    for index in range(3):
        host, device, orders = {'direct': [], 'replay': []}, {'direct': [], 'replay': []}, []
        for _ in range(25):
            order = ['direct', 'replay']
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                before = time.perf_counter()
                with torch.no_grad():
                    start.record()
                    functions[name]()
                    end.record()
                end.synchronize()
                host[name].append((time.perf_counter()-before)*1000)
                device[name].append(start.elapsed_time(end))
        rounds.append({'round': index, 'orders': orders, 'host_ms': host, 'device_ms': device,
                       'host_speedup': statistics.median(host['direct'])/statistics.median(host['replay']),
                       'device_speedup': statistics.median(device['direct'])/statistics.median(device['replay'])})
    result = {'schema': 1, 'case': case, 'passed': True, 'checks': checks, 'paired_bitwise': True,
              'audit': audits, 'guards': guards, 'rounds': rounds, 'build_ms': build_ms,
              'cpu_capture_ms': capture_ms, 'replay_prepare_ms': prepare_ms,
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'plan': plan, 'ledger': direct.ledger, 'launch_count': len(replay.launch_ledger),
              'production_api': owner is not None,
              'scope': 'resident-input complete MSE/backward/SGD; host dispatch and synchronization; unchanged precision; no framework comparator'}
    direct.close()
    replay.close()
    try:
        replay.run()
    except RuntimeError:
        result['guards']['closed_rejected'] = True
    else:
        raise AssertionError('closed replay admitted')
    (root/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1])
