"""Exact precompiled cache-length buckets for new-token-only native decoding."""
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App("openassembler-native-cached-decode")
image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")


@app.function(image=image, gpu="A100-40GB", timeout=900, max_containers=1, retries=0)
def validate(compiler):
    import hashlib
    import os
    import sys
    import tempfile
    import torch

    with tempfile.TemporaryDirectory(prefix="oa-native-cached-decode-") as directory:
        root = Path(directory)
        for name, raw in compiler.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "env/cuda.sh").chmod(0o700)
        sys.path.insert(0, directory)
        os.environ["PYTHONPATH"] = directory
        from openassembler.modelfront import capture_model, extract_weights, flatten_outputs
        from openassembler.modelfront.runtime import build_bundle, Runtime
        from torch.profiler import profile, ProfilerActivity

        torch.manual_seed(73)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(8, 8)
                self.k = torch.nn.Linear(8, 8)
                self.v = torch.nn.Linear(8, 8)
                self.out = torch.nn.Linear(8, 8)

            def forward(self, token, past_k, past_v):
                q = self.q(token).reshape(2, 1, 2, 4).permute(0, 2, 1, 3)
                k = self.k(token).reshape(2, 1, 2, 4).permute(0, 2, 1, 3)
                v = self.v(token).reshape(2, 1, 2, 4).permute(0, 2, 1, 3)
                keys = torch.cat((past_k, k), dim=-2)
                values = torch.cat((past_v, v), dim=-2)
                probabilities = torch.softmax((q @ keys.transpose(-2, -1))*0.5, dim=-1)
                output = (probabilities @ values).permute(0, 2, 1, 3).reshape(2, 1, 8)
                return self.out(output) + token, keys, values

        model = Decoder()
        plans, runtimes = {}, {}
        for previous_length in (1, 2, 3, 4):
            examples = (torch.randn(2, 1, 8), torch.randn(2, 2, previous_length, 4),
                        torch.randn(2, 2, previous_length, 4))
            plan = capture_model(model, examples)
            output = root / f"bucket-{previous_length}"
            build_bundle(plan, extract_weights(model, plan), output)
            plans[previous_length], runtimes[previous_length] = plan, Runtime.load(output)
        # One CPU-generated prefix key/value token initializes the cache. The
        # timed/profiled chain consumes one new token per precompiled bucket.
        prefix = torch.randn(2, 1, 8)
        with torch.no_grad():
            past_k = model.k(prefix).reshape(2, 1, 2, 4).permute(0, 2, 1, 3).contiguous()
            past_v = model.v(prefix).reshape(2, 1, 2, 4).permute(0, 2, 1, 3).contiguous()
        gpu_k, gpu_v = past_k.cuda(), past_v.cuda()
        checks, bucket_checks, all_ledger, compute = [], [], [], []
        for previous_length in (1, 2, 3, 4):
            token = torch.randn(2, 1, 8)
            with torch.no_grad():
                expected = model(token, past_k, past_v)
            plan, runtime = plans[previous_length], runtimes[previous_length]
            gpu_token = token.cuda()
            inputs = dict(zip((x["name"] for x in plan["inputs"]), (gpu_token, gpu_k, gpu_v)))
            before_k, before_v = gpu_k.cpu().clone(), gpu_v.cpu().clone()
            runtime.forward(inputs)
            runtime.poison_workspace()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                actual = runtime.forward(inputs)
                torch.cuda.synchronize()
            named = flatten_outputs(plan, expected)
            for name, value in named.items():
                got, expected_cpu = actual[name].cpu(), value.detach()
                close = torch.isclose(got, expected_cpu, rtol=2e-4, atol=2e-5)
                row = {"past_length": previous_length, "output": name, "passed": bool(close.all()),
                       "elements": got.numel(), "max_abs": float((got-expected_cpu).abs().max())}
                checks.append(row)
                assert row["passed"], row
            assert torch.equal(gpu_k.cpu(), before_k) and torch.equal(gpu_v.cpu(), before_v)
            assert torch.equal(gpu_token.cpu(), token)
            bucket_checks.append(runtime.check_canaries())
            events = [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]
            events = [name for name in events if not any(s in name.lower() for s in ("memcpy", "memset"))]
            assert events and all("oa_" in name for name in events), events
            compute.append(events)
            all_ledger.append(runtime.launch_ledger)
            # Feed device cache outputs directly to the next exact-shape plan.
            names = plan["output_structure"]["tuple"]
            gpu_k, gpu_v = actual[names[1]["tensor"]], actual[names[2]["tensor"]]
            past_k, past_v = expected[1].contiguous(), expected[2].contiguous()
        rejected = False
        try:
            runtimes[1].forward(dict(zip((x["name"] for x in plans[1]["inputs"]), (gpu_token, gpu_k, gpu_v))))
        except ValueError:
            rejected = True
        assert rejected
        for runtime in runtimes.values():
            runtime.close()
        return {"schema": 1, "passed": True, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "checks": checks, "canaries": bucket_checks,
                "unknown_shape_rejected": rejected, "profile_compute_kernels": compute, "launch_ledger": all_ledger,
                "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in compiler.items()
                                  if name.startswith("openassembler/") and name.endswith(".py")},
                "scope": "four new-token decode calls with exact past-length buckets1..4, two heads, batch2, width8; cache outputs feed next bucket; concat copies existing cache, no in-place cache or unbounded dynamic shape support; prefix initialized on CPU outside execution; no performance claim"}


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    result = validate.remote(runtime_files(native=True))
    path = Path(__file__).parent / ("a100-cached-decode-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "passed": True, "checks": len(result["checks"])}))
