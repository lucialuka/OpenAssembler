"""Native model capture/replay worker, shared by Linux, SSH and Modal.

PyTorch is used for CPU capture/reference calculations, tensor storage and DMA.
All workload GPU arithmetic is launched from the OA runtime's native ledger.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager

from openassembler.torchfront.archive import pack, unpack


@contextmanager
def _cpu_capture():
    """Fail before CUDA initialization during user factory/CPU capture work."""
    import torch
    from unittest.mock import patch
    def reject(*args, **kwargs):
        raise ValueError('native model factories and capture must run on CPU; GPU work starts in the OA runtime')
    with patch.object(torch.cuda, '_lazy_init', reject), patch.object(torch.cuda, 'init', reject):
        yield


def _model(root: Path, config: dict):
    import torch
    torch.manual_seed(config['seed'])
    path = root / config['entrypoint']
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_oa_model_' + hashlib.sha256(str(path).encode()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec)
    with _cpu_capture():
        spec.loader.exec_module(module)
        factory = getattr(module, config['factory'], None)
        if not callable(factory):
            raise ValueError(f'model factory {config["factory"]!r} is missing or not callable')
        model = factory()
    if not isinstance(model, torch.nn.Module):
        raise ValueError('model factory must return torch.nn.Module')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('model factory must create FP32 CPU parameters; GPU initialization is outside capture')
    return model.train(config['mode'] == 'training')


def _examples(config: dict, iteration: int = 0):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(config['seed'] + 103 * iteration)
    return [torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
            for shape in config['input_shapes']]


def _check(got, expected, label: str, config: dict) -> dict:
    import torch
    actual = got.detach().cpu()
    expected = expected.detach().cpu()
    if actual.shape != expected.shape:
        return {'name': label, 'passed': False, 'actual_shape': list(actual.shape),
                'expected_shape': list(expected.shape)}
    good = torch.isclose(actual, expected, rtol=config['rtol'], atol=config['atol'])
    return {'name': label, 'passed': bool(good.all()), 'elements': actual.numel(),
            'mismatches': int((~good).sum()),
            'max_abs': float((actual - expected).abs().max()) if actual.numel() else 0.0}


def _run(output: Path, config: dict, verify: bool, source_root: Path | None = None) -> dict:
    import torch
    from openassembler.modelfront.runtime import Runtime
    plan = json.loads((output / 'plan.json').read_text())
    reference = None
    numeric_reference = None
    if verify:
        if source_root is None:
            source_root = output / '_verification_source'
            unpack((output / 'model-source.zip').read_bytes(), source_root)
        reference = _model(source_root, config)
        # Build always snapshots the factory's actual weights. Restore them for
        # replay rather than assuming random initialization remains equivalent.
        saved = json.loads((output / 'weights.json').read_text())
        entries = saved.get('weights', saved) if isinstance(saved, dict) else saved
        if isinstance(entries, list):
            entries = {x['name']: x for x in entries}
        state = {**dict(reference.named_buffers()), **dict(reference.named_parameters())}
        with torch.no_grad():
            for parameter in plan['parameters']:
                item = entries[parameter['name']]
                data = (output / item['path']).read_bytes()
                tensor = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(parameter['shape'])
                state[parameter['target']].copy_(tensor)
        if config['math'] == 'f16_rne_f32_accum':
            from openassembler.modelfront.reference import CPUReference
            from openassembler.modelfront.graph import extract_weights
            numeric_reference = CPUReference(plan, extract_weights(reference, plan))
    runtime = Runtime.load(output)
    from .profiling import audit_profile
    checks, times, launches, guards, audits, device_times = [], [], [], [], [], []
    replay = None
    stable_inputs, stable_target = None, None
    replay_prepare_ms = 0.0
    try:
        for iteration in range(config['steps']):
            if iteration and verify:
                runtime.poison_workspace()
            cpu = [x.requires_grad_(config['mode'] == 'training' and verify) for x in _examples(config, iteration)]
            tensors = {spec['name']: value.to('cuda') for spec, value in zip(plan['inputs'], cpu)}
            if config.get('execution', 'direct') == 'cuda_graph':
                if stable_inputs is None:
                    stable_inputs = tensors
                else:
                    for name, value in tensors.items():
                        stable_inputs[name].copy_(value)
                tensors = stable_inputs
            if reference is not None:
                reference.zero_grad(set_to_none=True)
                if numeric_reference is not None:
                    expected_outputs = numeric_reference.forward(dict(zip((s['name'] for s in plan['inputs']), cpu)))
                else:
                    with _cpu_capture():
                        expected = reference(*cpu)
                    from openassembler.modelfront.graph import flatten_outputs
                    expected_outputs = flatten_outputs(plan, expected)
                expected_values = [expected_outputs[name] for name in plan['outputs']]
            if config['mode'] == 'training':
                if len(plan['outputs']) != 1:
                    raise ValueError('MSE training requires a single model output')
                shape = plan['buffers'][plan['outputs'][0]]['shape']
                target = torch.randn(shape, generator=torch.Generator().manual_seed(
                    config['seed'] + 997 + iteration), dtype=torch.float32) * 0.25
                gpu_target = target.to('cuda')
                if config.get('execution', 'direct') == 'cuda_graph':
                    if stable_target is None:
                        stable_target = gpu_target
                        from .replay import TrainingReplay
                        before = time.perf_counter()
                        replay = TrainingReplay(plan, runtime.weights, output, tensors, stable_target,
                                                learning_rate=config['learning_rate'])
                        runtime.close()
                        runtime = replay.runtime
                        replay_prepare_ms = (time.perf_counter()-before)*1000
                    else:
                        stable_target.copy_(gpu_target)
                    gpu_target = stable_target
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                gpu_start.record()
                if verify and iteration == config['steps'] - 1:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        result = replay.run() if replay else runtime.train_step(tensors, gpu_target, config['learning_rate'])
                        gpu_end.record()
                        torch.cuda.synchronize()
                    audits.append(audit_profile(profile, replay.launch_ledger if replay else runtime.launch_ledger))
                else:
                    result = replay.run() if replay else runtime.train_step(tensors, gpu_target, config['learning_rate'])
                    gpu_end.record()
                    torch.cuda.synchronize()
                gpu_end.synchronize()
                device_times.append(gpu_start.elapsed_time(gpu_end))
                times.append((time.perf_counter() - start) * 1000)
                if reference is not None:
                    if numeric_reference is not None:
                        expected_step = numeric_reference.train_step(dict(zip(
                            (s['name'] for s in plan['inputs']), cpu)), target, config['learning_rate'])
                        for name in plan['outputs']:
                            checks.append(_check(result['outputs'][name], expected_step['outputs'][name],
                                                 f'{iteration}/output/{name}', config))
                        checks.append(_check(result['loss'].reshape(()), expected_step['loss'].reshape(()),
                                             f'{iteration}/loss', config))
                        for group in ('inputs', 'parameters'):
                            for name, value in expected_step['gradients'][group].items():
                                checks.append(_check(result['gradients'][group][name], value,
                                                     f'{iteration}/gradient/{group}/{name}', config))
                        for name, value in numeric_reference.weights.items():
                            checks.append(_check(runtime.weights[name], value, f'{iteration}/parameter/{name}', config))
                        launches.extend({**entry, 'iteration': iteration} for entry in runtime.launch_ledger)
                        guards.append(runtime.check_canaries())
                        continue
                    loss = torch.nn.functional.mse_loss(expected_values[0], target)
                    loss.backward()
                    outputs = result['outputs']
                    for name, value in zip(plan['outputs'], expected_values):
                        checks.append(_check(outputs[name], value, f'{iteration}/output/{name}', config))
                    checks.append(_check(result['loss'].reshape(()), loss, f'{iteration}/loss', config))
                    for spec, value in zip(plan['inputs'], cpu):
                        if value.grad is not None:
                            checks.append(_check(result['gradients']['inputs'][spec['name']], value.grad,
                                                 f'{iteration}/input_gradient/{spec["name"]}', config))
                    for parameter in plan['parameters']:
                        name, original = parameter['name'], state[parameter['target']]
                        if isinstance(original, torch.nn.Parameter) and original.grad is not None:
                            checks.append(_check(result['gradients']['parameters'][name], original.grad,
                                                 f'{iteration}/gradient/{name}', config))
                    with torch.no_grad():
                        for p in reference.parameters():
                            if p.grad is not None:
                                p.add_(p.grad, alpha=-config['learning_rate'])
                    for parameter in plan['parameters']:
                        checks.append(_check(runtime.weights[parameter['name']], state[parameter['target']],
                                             f'{iteration}/parameter/{parameter["name"]}', config))
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                if verify and iteration == config['steps'] - 1:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        outputs = runtime.forward(tensors)
                        torch.cuda.synchronize()
                    audits.append(audit_profile(profile, runtime.launch_ledger))
                else:
                    outputs = runtime.forward(tensors)
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                if reference is not None:
                    for name, value in zip(plan['outputs'], expected_values):
                        checks.append(_check(outputs[name], value, f'{iteration}/output/{name}', config))
            launches.extend({**entry, 'iteration': iteration} for entry in runtime.launch_ledger)
            if verify:
                guards.append(runtime.check_canaries())
        device = torch.cuda.get_device_properties(0)
        ledger = launches
        if not ledger:
            raise ValueError('native runtime returned no executable launch ledger')
        return {'schema': 1, 'backend': 'openassembler', 'mode': config['mode'],
                'passed': all(c['passed'] for c in checks), 'verified': verify,
                'checks': checks, 'iterations': config['steps'], 'host_step_ms': times,
                'execution': config.get('execution', 'direct'), 'replay_prepare_ms': replay_prepare_ms,
                'cuda_stream_elapsed_ms': device_times,
                'profiled_iterations': [config['steps']-1] if verify else [],
                'timing_scope': 'resident input native model dispatch, synchronization included; last verified iteration profiled; no performance comparator',
                'device': {'name': device.name, 'cc': f'{device.major}.{device.minor}'},
                'launch_count': len(ledger), 'launch_ledger': ledger,
                'canaries': guards, 'changed_inputs': config['steps'] > 1,
                'profiler_native_audits': audits,
                'gpu_codegen_during_replay': 0, 'vendor_compute_launches': 0}
    finally:
        if replay:
            replay.close()
        else:
            runtime.close()


def process(request: dict) -> dict:
    try:
        config = request['config']
        with tempfile.TemporaryDirectory(prefix='oa-native-model-worker-') as d:
            root = Path(d)
            output = root / 'output'
            if request['action'] == 'compile':
                source = root / 'source'
                blob = base64.b64decode(request['source'], validate=True)
                unpack(blob, source)
                from openassembler.modelfront.graph import capture_model, extract_weights
                from openassembler.modelfront.runtime import build_bundle
                started = time.perf_counter()
                model = _model(source, config)
                factory_ms = (time.perf_counter()-started)*1000
                examples = _examples(config)
                started = time.perf_counter()
                with _cpu_capture():
                    plan = capture_model(model, examples, math_mode=config['math'])
                capture_ms = (time.perf_counter()-started)*1000
                from .optimization import optimize_plan
                started = time.perf_counter()
                plan = optimize_plan(plan, storage_views=config.get('storage_views', True),
                                     reuse_storage=config.get('reuse_storage', False))
                planning_ms = (time.perf_counter()-started)*1000
                started = time.perf_counter()
                build_bundle(plan, extract_weights(model, plan), output,
                             sass_listings=config.get('sass_listings', True))
                bundle_ms = (time.perf_counter()-started)*1000
                (output / 'model-source.zip').write_bytes(blob)
                (output / 'model.json').write_text(json.dumps({'schema': 1, 'backend': 'openassembler',
                    'config': config, 'source_root': request.get('source_root'),
                    'source_digest': request['source_digest']}, indent=2))
                # Native source discovery remains independent from the model
                # graph. CUDA/PTX/@kernel translation units are compiled by OA,
                # while host files stay in the retained source bundle.
                from openassembler.repository import native_candidates
                native = native_candidates(source, output)
                if native:
                    from openassembler.build import Options, compile_paths
                    compiled = compile_paths(source, output / 'repository-native', Options(fallback=None))
                    if compiled.rc:
                        raise ValueError('repository contains unsupported native kernels; no fallback is allowed')
                report = _run(output, config, True, source) if request.get('verify') else {
                    'passed': True, 'verified': False, 'backend': 'openassembler',
                    'native_kernels': len(plan['steps']), 'source_native_files': len(native)}
                report['compile_stage_ms'] = {'cpu_factory': factory_ms, 'cpu_capture_autodiff': capture_ms,
                    'storage_planning': planning_ms, 'native_bundle_and_weights': bundle_ms}
                report['compile_stage_scope'] = 'worker-local non-overlapping phases; excludes transport, independent repository source builds and validation'
                (output / 'model-build-report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
                files = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*')
                         if p.is_file() and p.name != 'writer.lock'}
                return {'artifacts': base64.b64encode(pack(files)).decode(), 'report': report}
            if request['action'] == 'run':
                unpack(base64.b64decode(request['artifacts'], validate=True), output)
                return {'report': _run(output, config, bool(request.get('verify')))}
            raise ValueError('unknown native model request')
    except Exception as exc:
        traceback.print_exc()
        return {'error': f'native model {request.get("action", "request")} failed: {type(exc).__name__}: {exc}'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    wrapper = Path(__file__).resolve().parents[2] / 'env/cuda.sh'
    if wrapper.is_file():
        wrapper.chmod(wrapper.stat().st_mode | 0o100)
    args.result.write_text(json.dumps(process(json.loads(args.request.read_text())), allow_nan=False))


if __name__ == '__main__':
    main()
