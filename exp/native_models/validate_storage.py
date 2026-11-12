'''A100 complete-state proof for guarded exact-size storage reuse.'''
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App('openassembler-native-storage-proof')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12').pip_install(
    'torch==2.7.1', index_url='https://download.pytorch.org/whl/cu128')


@app.function(image=image, gpu='A100-40GB', timeout=1200, max_containers=1, retries=0)
def validate(compiler, case_name=''):
    from contextlib import nullcontext
    import hashlib
    import runpy
    import sys
    import tempfile
    import torch

    with tempfile.TemporaryDirectory(prefix='oa-storage-') as directory:
        root = Path(directory)
        for name, raw in compiler.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / 'env/cuda.sh').chmod(0o700)
        sys.path.insert(0, directory)
        from openassembler.modelfront.graph import capture_model, extract_weights
        from openassembler.modelfront.optimization import optimize_plan
        from openassembler.modelfront.runtime import Runtime, build_bundle, audit_profiler_events
        from openassembler.modelfront.reference import CPUReference

        torch.manual_seed(97)
        torch.set_num_threads(2)
        cases = [('native_model', (17, 8), 'fp32'),
                 ('native_attention_model', (2, 7, 8), 'fp32'),
                 ('native_tensor_model', (128, 128), 'f16_rne_f32_accum')]
        if case_name:
            cases = [case for case in cases if case[0] == case_name]
            assert cases, 'unknown storage proof case'
        checks, artifacts, audits = [], {}, {}

        def check(value, expected, label, *, exact=False, half=False):
            got, expected = value.detach().cpu(), expected.detach().cpu()
            close = got == expected if exact else torch.isclose(
                got, expected, rtol=.002 if half else 1e-4, atol=2e-5 if half else 1e-5)
            item = {'label': label, 'passed': bool(close.all()), 'elements': got.numel(),
                    'mismatches': int((~close).sum()), 'max_abs': float((got-expected).abs().max()),
                    'contract': 'bitwise' if exact else ('declared FP16 RNE products' if half else 'CPU Module/autograd')}
            checks.append(item)
            assert item['passed'], item
            return got

        for case, shape, math_mode in cases:
            model = runpy.run_path(str(root/'cases'/case/'model.py'))['create_model']().train()
            captured = capture_model(model, [torch.randn(shape) * .25], math_mode=math_mode)
            weights = extract_weights(model, captured)
            numeric = CPUReference(captured, weights) if math_mode != 'fp32' else None
            runs = {}
            for name, reuse in [('views', False), ('reuse', True)]:
                plan = optimize_plan(captured, reuse_storage=reuse)
                output = root / case / name
                build_bundle(plan, weights, output)
                runs[name] = Runtime.load(output)
                artifacts[case + '/' + name] = {
                    'plan': plan, 'ledger': json.loads((output/'native-ledger.json').read_text())}
            for iteration in range(4):
                cpu = (torch.randn(shape) * .25).requires_grad_()
                target = torch.randn(shape) * .25
                gpu_input, gpu_target = cpu.detach().cuda(), target.cuda()
                if numeric is None:
                    model.zero_grad(set_to_none=True)
                    prediction = model(cpu)
                    loss = torch.nn.functional.mse_loss(prediction, target)
                    loss.backward()
                    state = dict(model.named_parameters())
                    expected = {'prediction': prediction, 'loss': loss, 'input_gradient': cpu.grad,
                                'gradients': {p['name']: state[p['target']].grad for p in captured['parameters']}}
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.add_(parameter.grad, alpha=-.01)
                    expected['weights'] = {p['name']: state[p['target']] for p in captured['parameters']}
                else:
                    step = numeric.train_step({captured['inputs'][0]['name']: cpu}, target, .01)
                    expected = {'prediction': step['outputs'][captured['outputs'][0]], 'loss': step['loss'],
                                'input_gradient': step['gradients']['inputs'][captured['inputs'][0]['name']],
                                'gradients': step['gradients']['parameters'], 'weights': numeric.weights}
                baseline = {}
                for name, runtime in runs.items():
                    if iteration:
                        runtime.poison_workspace()
                    print(f'validate {case}/{iteration}/{name}', flush=True)
                    # Audit a steady-state invocation after module loading and
                    # warm replay, exactly as the production worker does. The
                    # initial profiler trace did not observe two pack activities and
                    # failed the unchanged exact-count gate; retain its log.
                    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) if iteration == 3 else nullcontext()
                    with profile as prof:
                        result = runtime.train_step({captured['inputs'][0]['name']: gpu_input}, gpu_target, .01)
                        torch.cuda.synchronize()
                    if prof is not None:
                        try:
                            audit = audit_profiler_events(prof.events(), runtime.launch_ledger)
                        except ValueError as exc:
                            audit = {'passed': False, 'error': str(exc), 'ledger': runtime.launch_ledger,
                                     'events': [{'name': event.name, 'device_type': str(event.device_type),
                                                 'user_annotation': bool(getattr(event, 'is_user_annotation', False)),
                                                 'activity_type': str(getattr(event, 'activity_type', ''))}
                                                for event in prof.events()]}
                            trace = root / 'failed-profiler.json'
                            prof.export_chrome_trace(str(trace))
                            audit['chrome_trace'] = json.loads(trace.read_text())
                        audits[f'{case}/{iteration}/{name}'] = audit
                    actual = {'prediction': result['prediction'], 'loss': result['loss'].reshape(()),
                              'input_gradient': result['gradients']['inputs'][captured['inputs'][0]['name']],
                              **{'gradient/' + k: v for k, v in result['gradients']['parameters'].items()},
                              **{'weight/' + k: v for k, v in runtime.weights.items()}}
                    reference = {'prediction': expected['prediction'], 'loss': expected['loss'].reshape(()),
                                 'input_gradient': expected['input_gradient'],
                                 **{'gradient/' + k: v for k, v in expected['gradients'].items()},
                                 **{'weight/' + k: v for k, v in expected['weights'].items()}}
                    for field, value in actual.items():
                        got = check(value, reference[field], f'{case}/{iteration}/{name}/{field}', half=numeric is not None)
                        if name == 'views':
                            baseline[field] = got
                        else:
                            check(got, baseline[field], f'{case}/{iteration}/paired/{field}', exact=True)
                    assert torch.equal(gpu_input.cpu(), cpu.detach())
                    assert torch.equal(gpu_target.cpu(), target)
                    record = artifacts[case + '/' + name]
                    record['canaries'] = runtime.check_canaries()
                    record['allocated_fp32_payload_bytes'] = sum((tensor.numel()-8)*4 for tensor in runtime._allocations
                                                                 if tensor.dtype == torch.float32)
                    record['actual_launch_count'] = len(runtime.launch_ledger)
            for runtime in runs.values():
                runtime.close()
        return {'schema': 1, 'passed': all(audit['passed'] for audit in audits.values()),
                'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'checks': checks, 'artifacts': artifacts, 'profiler_audits': audits,
                'source_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in compiler.items()},
                'scope': 'four complete MSE/backward/SGD updates each for FP32 MLP, FP32 attention and explicit FP16-RNE-product TensorCore MLP; full states/guards/poison/immutable inputs; bitwise paired views/reuse; no timing claim'}


@app.local_entrypoint()
def main(case: str = 'native_attention_model'):
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    requested_case = case
    source = Path(__file__).read_bytes()
    compiler = runtime_files(native=True)
    root = Path(__file__).resolve().parents[2]
    for model_case in ['native_model', 'native_attention_model', 'native_tensor_model']:
        compiler[f'cases/{model_case}/model.py'] = (root/'examples'/model_case/'model.py').read_bytes()
    result = validate.remote(compiler, requested_case)
    assert {name.split('/')[0] for name in result['artifacts']} == {requested_case}
    result['requested_case'] = requested_case
    import hashlib
    identity = hashlib.sha256(source).hexdigest()
    snapshot = Path(__file__).parent / 'runs' / ('storage-driver-' + identity + '.py')
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(source)
    result['experiment_source_sha256'] = identity
    result['experiment_source_snapshot'] = snapshot.name
    path = Path(__file__).parent / 'runs' / ('storage-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + chr(10))
    print(json.dumps({'path': str(path), 'passed': result['passed'], 'checks': len(result['checks'])}))
    assert result['passed'], 'exact profiler launch gate failed; raw failure retained'
