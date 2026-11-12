"""Paired complete native attention graphs: materializations versus storage views."""
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App('openassembler-native-view-ablation')
image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12').pip_install(
    'torch==2.7.1', index_url='https://download.pytorch.org/whl/cu128')


@app.function(image=image, gpu='A100-40GB', timeout=900, max_containers=1, retries=0)
def measure(compiler):
    import hashlib
    import random
    import statistics
    import sys
    import tempfile
    import torch
    with tempfile.TemporaryDirectory(prefix='oa-views-') as directory:
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

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(8, 8)
                self.k = torch.nn.Linear(8, 8)
                self.v = torch.nn.Linear(8, 8)
                self.o = torch.nn.Linear(8, 8)
                self.norm = torch.nn.LayerNorm(8)
                self.up = torch.nn.Linear(8, 16)
                self.down = torch.nn.Linear(16, 8)
                self.gelu = torch.nn.GELU(approximate='tanh')

            def forward(self, x):
                q = self.q(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
                k = self.k(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
                v = self.v(x).reshape(2, 7, 2, 4).permute(0, 2, 1, 3)
                p = torch.softmax((q @ k.transpose(-2, -1)) * .5, dim=-1)
                y = (p @ v).permute(0, 2, 1, 3).reshape(2, 7, 8)
                z = self.norm(self.o(y) + x)
                return self.down(self.gelu(self.up(z))) + z

        torch.manual_seed(83)
        torch.set_num_threads(2)
        model = Attention().eval()
        x = torch.randn(2, 7, 8) * .25
        captured = capture_model(model, [x])
        weights = extract_weights(model, captured)
        runs, checks, artifacts = {}, [], {}
        inputs = {captured['inputs'][0]['name']: x.cuda()}
        for name, enabled in [('copies', False), ('views', True)]:
            plan = optimize_plan(captured, storage_views=enabled)
            out = root / name
            build_bundle(plan, weights, out)
            runtime = Runtime.load(out)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    runtime.forward(inputs)
            stream.synchronize()
            graphs = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs, stream=stream):
                for _ in range(16):
                    outputs = runtime.forward(inputs)
            stream.synchronize()
            runs[name] = (runtime, stream, graphs, outputs)
            artifacts[name] = {'plan_sha256': hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
                               'native_ledger': json.loads((out/'native-ledger.json').read_text()),
                               'view_count': len(plan.get('aliases', {})), 'allocations': len(runtime._guards),
                               'forward_launch_count': len(runtime.launch_ledger)}
        # Full changed-input/poison proofs precede all timing. Host-to-device
        # copies are outside graphs; graph arithmetic is exclusively OA.
        for iteration in range(3):
            value = torch.randn(2, 7, 8) * .25
            inputs[captured['inputs'][0]['name']].copy_(value)
            torch.cuda.synchronize()
            expected = model(value).detach()
            actuals = {}
            for name, (runtime, stream, graph, outputs) in runs.items():
                runtime.poison_workspace()
                graph.replay()
                torch.cuda.synchronize()
                actual = outputs[captured['outputs'][0]].cpu()
                close = torch.isclose(actual, expected, rtol=1e-4, atol=1e-5)
                canaries = runtime.check_canaries()
                checks.append({'method': name, 'input': iteration, 'passed': bool(close.all()),
                               'max_abs': float((actual-expected).abs().max()), 'canaries': canaries})
                assert bool(close.all())
                actuals[name] = actual
                assert torch.equal(inputs[captured['inputs'][0]['name']].cpu(), value)
                assert all(torch.equal(runtime.weights[k].cpu(), v) for k, v in weights.items())
            assert torch.equal(actuals['copies'], actuals['views'])
        audits = {}
        for name, (runtime, stream, _, _) in runs.items():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.cuda.stream(stream):
                    runtime.forward(inputs)
                stream.synchronize()
            audits[name] = audit_profiler_events(prof.events(), runtime.launch_ledger)
        rng = random.Random(1729)
        rounds = []
        for round_index in range(3):
            samples = {'copies': [], 'views': []}
            orders = []
            for _ in range(15):
                order = ['copies', 'views']
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    _, stream, graph, _ = runs[name]
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        a.record(stream)
                        graph.replay()
                        b.record(stream)
                    b.synchronize()
                    samples[name].append(a.elapsed_time(b) / 16)
            rounds.append({'round': round_index, 'samples_ms': samples, 'paired_order': orders,
                           'copies_over_views': statistics.median(samples['copies']) / statistics.median(samples['views'])})
        for runtime, _, _, _ in runs.values():
            runtime.close()
        return {'schema': 1, 'passed': True, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'checks': checks, 'artifacts': artifacts, 'profiler_audits': audits,
                'rounds': rounds, 'source_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in compiler.items()},
                'scope': 'complete resident-input FP32 attention+LayerNorm+GELU inference; 16-invocation graphs; paired same-worker copies/views; excludes capture, compile, allocation and transfer; no vendor-performance comparison'}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    result = measure.remote(runtime_files(native=True))
    path = Path(__file__).parent / 'runs' / ('views-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'path': str(path), 'passed': result['passed'], 'ratios': [r['copies_over_views'] for r in result['rounds']]}))
