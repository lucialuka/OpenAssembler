"""Byte-exact pack checks and paired full-workload A100 measurements."""
from contextlib import ExitStack
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import torch

from exp.ml_suite.common import guarded, guards, launch, paired
from exp.ml_suite.engines import Engine
from exp.ml_suite.tasks import TASKS, create
from exp.model_perf.campaign import environment, fields
from exp.model_perf.native_pack import enable
from exp.model_perf.pack_reuse import bind_candidate
from openassembler.gpu.cuda_driver import Driver
from openassembler.modelfront import kernels_tensor
from openassembler.modelfront.graph import capture_model, extract_weights
from openassembler.modelfront.kernels import build_bundle
from openassembler.modelfront.optimization import optimize_plan
from openassembler.modelfront.profiling import audit_profile
from openassembler.modelfront.reference import CPUReference
from openassembler.modelfront.runtime import Runtime


class PackedPTXEngine(Engine):
    """Exact OA PTX under ptxas with the production pack reuse schedule."""

    def __init__(self, plan, weights, root):
        super().__init__(plan, weights, root, 'nvidia_same_ptx')
        from openassembler.modelfront.tensor_schedule import workspace_plan
        self.schedule = workspace_plan(plan, self.steps, self.recipes)
        if self.schedule is None:
            return
        storage = {name: self.allocate(spec['shape'],
                                       torch.float16 if spec['dtype'] == 'f16' else torch.float32,
                                       label='scheduled:' + name)
                   for name, spec in self.schedule['allocations'].items()}
        self.bindings = {out: {name: storage[spec['allocation']].view(-1).narrow(
            0, spec['offset'], math.prod(spec['shape'])).view(spec['shape'])
                              for name, spec in bindings.items()}
                         for out, bindings in self.schedule['bindings'].items()}
        self.ready = set()

    def forward(self):
        if self.schedule is not None:
            self.ready.clear()
        return super().forward()

    def run_phase(self, phase):
        if self.schedule is None:
            return super().run_phase(phase)
        for step in self.steps:
            if step['phase'] != phase:
                continue
            if step.get('alias'):
                self.state[step['output']] = self.state[step['inputs'][0]].view(self.specs[step['output']]['shape'])
                continue
            xs = [self.state[name] for name in step['inputs']]
            out = self.state[step['output']]
            if step.get('tensor'):
                recipe = self.recipes[step['entry']]
                bindings = {f'a{i}': value for i, value in enumerate(xs)}
                bindings.update(self.bindings[step['output']])
                bindings['output'] = out
                for stage in recipe.stages:
                    slot = self.schedule['pack_slots'][step['output']].get(stage.output)
                    if slot is not None:
                        if slot in self.ready:
                            continue
                        self.ready.add(slot)
                    launch(self.driver, self.functions[stage.name], stage.launch,
                           [*[bindings[name] for name in stage.inputs], bindings[stage.output]])
            else:
                self.op(step['kernel'], xs, out, step.get('attrs'))


def gate_packs(root):
    torch.empty(1, device="cuda")
    driver = Driver()
    rng = random.Random(3081)
    special = [0, 0x80000000, 1, 0x80000001, 0x33000000, 0x33000001,
               0x33800000, 0x387FFFFF, 0x38800000, 0x3F801000, 0x3F803000,
               0xBF801000, 0x477FE000, 0x477FF000, 0x7F800000, 0xFF800000,
               0x7F800001, 0xFF800001, 0x7FC01234, 0xFFC01234]
    results = []
    for rows, cols in [(128, 128), (128, 256), (256, 128)]:
        count = rows * cols
        for transpose in (False, True):
            stage = kernels_tensor._pack((rows, cols), transpose, "a0", "packed")
            recipe = kernels_tensor.TensorRecipe("probe", "probe", {}, {}, (stage,))
            directory = root / stage.name
            try:
                record = kernels_tensor.build_tensor_kernels(directory, [recipe])[stage.name]
            except subprocess.CalledProcessError as error:
                raise RuntimeError(error.stderr) from error
            for arm in ("native", "reference"):
                binary = (directory / record['module']).with_name(arm + '.cubin').read_bytes()
                module = driver.load(binary)
                fn = driver.function(module, stage.name)
                for pattern in ("edges", "bits"):
                    bits = [special[i % len(special)] if pattern == "edges" else rng.getrandbits(32)
                            for i in range(count)]
                    source_backing, source = guarded(driver, (rows, cols), torch.float32)
                    source_bits = torch.frombuffer(bytearray(struct.pack(f"<{count}I", *bits)),
                                                   dtype=torch.int32).reshape(rows, cols)
                    source.view(torch.int32).copy_(source_bits)
                    out_backing, output = guarded(driver, (count,), torch.float16)
                    expected = []
                    for raw in bits:
                        value = struct.unpack("<f", struct.pack("<I", raw))[0]
                        sign = (raw >> 16) & 0x8000
                        if (raw & 0x7fffffff) > 0x7f800000:
                            half = struct.pack("<H", sign | 0x7e00)
                        else:
                            try:
                                half = struct.pack("<e", value)
                            except OverflowError:
                                half = struct.pack("<H", sign | 0x7c00)
                        expected.append(struct.unpack("<h", half)[0])
                    expect = torch.tensor(expected, dtype=torch.int16).reshape(rows, cols)
                    launch(driver, fn, stage.launch, [source, output])
                    actual = output.view(torch.int16).cpu()
                    assert torch.equal(actual, (expect.T if transpose else expect).reshape(-1)), (
                        arm, pattern, rows, cols, transpose, int((actual != (expect.T if transpose else expect).reshape(-1)).sum()))
                    assert torch.equal(source.view(torch.int32).cpu(), source_bits)
                    guards(driver, [source_backing, out_backing])
                driver.unload(module)
                results.append({"shape": [rows, cols], "transpose": transpose, "arm": arm,
                                "patterns": 2, "module_sha256": sha256(binary).hexdigest(), **record})
    return results


def model_case(name, root, phase, storage_views, reuse_storage):
    torch.manual_seed(941 if phase.endswith('integrated') else 818)
    spec = TASKS[name]
    shape, training = spec['shape'], spec['training']
    model = create(name)
    plan = optimize_plan(capture_model(model, [torch.randn(shape) * .25], math_mode=spec['math']),
                         storage_views=storage_views, reuse_storage=reuse_storage)
    weights = extract_weights(model, plan)
    input_name, output = plan['inputs'][0]['name'], plan['outputs'][0]
    out_shape = plan['buffers'][output]['shape']
    runtimes, engines, build_seconds = {}, {}, {}
    reuse = phase.startswith('pack-reuse')
    for label in ('incumbent', 'candidate'):
        with ExitStack() as stack:
            if label == 'candidate' and not reuse:
                enable(stack)
            if label == 'incumbent' and reuse:
                from openassembler.modelfront import tensor_schedule
                stack.enter_context(patch.object(tensor_schedule, 'workspace_plan', return_value=None))
            bundle = root / label
            if label == 'candidate' and reuse:
                bundle = root / 'incumbent'
                runtimes[label] = Runtime.load(bundle)
                if not phase.endswith('integrated'):
                    bind_candidate(runtimes[label])
                continue
            start = perf_counter()
            build_bundle(plan, weights, bundle, sass_listings=True, cache=False)
            build_seconds[label] = perf_counter() - start
            runtimes[label] = Runtime.load(bundle)
            engines[label + '_ptxas'] = Engine(plan, weights, bundle, 'nvidia_same_ptx')
    engines['triton'] = Engine(plan, weights, root / 'incumbent', 'triton')
    if phase.endswith('integrated'):
        engines['candidate_ptxas'] = PackedPTXEngine(plan, weights, root / 'incumbent')
    x, target = torch.empty(shape, device='cuda'), torch.empty(out_shape, device='cuda')
    inputs = {input_name: x}
    functions = {key: (lambda r=r: r.train_step(inputs, target, .01) if training else r.forward(inputs))
                 for key, r in runtimes.items()}
    functions.update({key: (e.train if training else e.forward) for key, e in engines.items()})

    def update(cx, cy):
        x.copy_(cx)
        target.copy_(cy)
        for e in engines.values():
            e.inputs[input_name].copy_(cx)
            e.target.copy_(cy)

    validation = []

    def check(key, result, truth, reference):
        if key in runtimes:
            r = runtimes[key]
            actual = fields(r, result) if training else {'prediction': result[output]}
            r.check_canaries()
        else:
            e = engines[key]
            actual = e.fields() if training else {'prediction': e.state[output]}
            e.check_guards()
        expected = fields(reference, truth) if training else {'prediction': truth[output]}
        assert actual.keys() == expected.keys(), (key, actual.keys(), expected.keys())
        snapshot = {'input': cx.clone(), 'target': cy.clone(), 'actual': {}, 'expected': {}}
        for field, value in actual.items():
            observed, wanted = value.detach().cpu(), expected[field].detach()
            close = torch.isclose(observed, wanted, rtol=.002, atol=2e-5)
            validation.append({'iteration': i, 'arm': key, 'field': field, 'elements': observed.numel(),
                               'mismatches': int((~close).sum()), 'max_abs': float((observed - wanted).abs().max()),
                               'actual_sha256': sha256(observed.numpy().tobytes()).hexdigest(),
                               'expected_sha256': sha256(wanted.numpy().tobytes()).hexdigest()})
            torch.testing.assert_close(observed, wanted, rtol=.002, atol=2e-5,
                                       msg=lambda err: f'{key}/{field}: {err}')
            if i in (0, 6):
                snapshot['actual'][field] = observed.clone()
                snapshot['expected'][field] = wanted.clone()
        if i in (0, 6):
            torch.save(snapshot, root / f'complete-state-{i}-{key}.pt')

    reference = CPUReference(plan, weights)
    graphs, captured = {}, {}
    for i in range(12):
        cx, cy = torch.randn(shape) * .25, torch.randn(out_shape) * .25
        update(cx, cy)
        if i == 6:
            reference = CPUReference(plan, weights)
            for obj in [*runtimes.values(), *engines.values()]:
                for parameter, value in obj.weights.items():
                    value.copy_(weights[parameter])
            for key, fn in functions.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                    captured[key] = fn()
                graphs[key] = graph
        if not training:
            # Each direct iteration and replay changes every inference weight
            # in place after capture, exercising pack invalidation explicitly.
            for parameter in reference.weights:
                changed = torch.randn_like(reference.weights[parameter]) * .05
                reference.weights[parameter].copy_(changed)
                for obj in [*runtimes.values(), *engines.values()]:
                    obj.weights[parameter].copy_(changed)
        truth = reference.train_step({input_name: cx}, cy, .01) if training else reference.forward({input_name: cx})
        for key, fn in functions.items():
            if i and key in runtimes:
                runtimes[key].poison_workspace()
            if i >= 6:
                graphs[key].replay()
                result = captured[key]
            else:
                result = fn()
            try:
                check(key, result, truth, reference)
            except AssertionError as error:
                torch.save({'iteration': i, 'arm': key, 'plan': plan,
                            'input': cx, 'target': cy, 'truth': truth,
                            'actual': fields(runtimes[key], result) if training and key in runtimes else result,
                            'initial_weights': weights}, root / 'failure.pt')
                raise AssertionError(f'iteration={i}, arm={key}, views={storage_views}, reuse={reuse_storage}: {error}') from error
            assert torch.equal(x.cpu(), cx) and torch.equal(target.cpu(), cy)
            if key in engines:
                assert torch.equal(engines[key].inputs[input_name].cpu(), cx)
                assert torch.equal(engines[key].target.cpu(), cy)

    audits = {}
    audit_failures = []
    for key, r in runtimes.items():
        for attempt in range(3):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                torch.cuda.synchronize()
                functions[key]()
                r.synchronize()
            try:
                audits[key] = audit_profile(prof, r.launch_ledger)
                break
            except ValueError as error:
                audit_failures.append({'arm': key, 'attempt': attempt, 'error': str(error)})
                if attempt == 2:
                    raise
    if phase.endswith('integrated'):
        engine = engines['candidate_ptxas']
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            torch.cuda.synchronize()
            functions['candidate_ptxas']()
            torch.cuda.synchronize()
        audits['candidate_ptxas'] = audit_profile(prof, runtimes['candidate'].launch_ledger)
    total = paired(functions, repeats=4, bound_runtimes=runtimes)
    parts = {}
    for phase in ('forward', 'backward') if training else ('forward',):
        region = {key: getattr(engine, phase) for key, engine in engines.items()}
        for key, r in runtimes.items():
            def region_call(r=r):
                r._packed_ready.clear()
                values = dict(r._last)
                if training:
                    values[plan['backward']['seed_grads'][output]] = r._training[output]['seed']
                r._run(phase, values)
            region[key] = region_call
        parts[phase] = paired(region, repeats=4, bound_runtimes=runtimes)
    record = {"task": name, "storage_views": storage_views, "reuse_storage": reuse_storage,
              "plan": plan, "build_seconds": build_seconds, "direct_steps": 6,
              "graph_steps": 6, "total": total, "parts": parts, "audits": audits,
              "validation": validation,
              "audit_failures": audit_failures,
              "allocation_bytes": {key: sum(t.numel() * t.element_size() for t in r._allocations)
                                   for key, r in runtimes.items()},
              "ledgers": {key: r.ledger for key, r in runtimes.items()}}
    for obj in [*runtimes.values(), *engines.values()]:
        obj.close()
    return record


def configure_device():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)


def run(phase):
    root = Path('results')
    probes = []
    if phase == 'native-pack':
        configure_device()
        with ExitStack() as stack:
            enable(stack)
            probes = gate_packs(root / 'packs')
    import triton
    meta = {"environment": environment(), "torch": torch.__version__, "triton": triton.__version__, "phase": phase,
            "packs": probes, "passed": False, "models": []}
    layouts = [(True, True)] if not phase.endswith('integrated') else [(v, r) for v in (True, False) for r in (True, False)]
    if 'debug' in phase:
        layouts = [(False, True)]
    for views, storage in layouts:
        for name in ('dense_regression_fp16', 'projection_inference'):
            if 'debug' in phase and name != 'dense_regression_fp16':
                continue
            directory = root / f'{name}-{views}-{storage}'
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'worker.log').open('w') as log:
                worker = subprocess.run([sys.executable, '-m', 'exp.model_perf.native_pack_gate',
                                         phase, name, str(directory), str(int(views)), str(int(storage))],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=160)
            if worker.returncode:
                raise RuntimeError((directory / 'worker.log').read_text()[-8000:])
            meta['models'].append(json.loads((directory / 'result.json').read_text()))
            (root / 'result.json').write_text(json.dumps(meta, indent=2))
    meta['passed'] = True
    (root / 'result.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    phase, name, directory, views, storage = sys.argv[1:]
    configure_device()
    directory = Path(directory)
    record = model_case(name, directory, phase, bool(int(views)), bool(int(storage)))
    (directory / 'result.json').write_text(json.dumps(record, indent=2))
