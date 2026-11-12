"""Pinned A100 Python attention-region native composition and state checks."""
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App("openassembler-native-attention-region")
image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")


@app.function(image=image, gpu="A100-40GB", timeout=900, max_containers=1, retries=0)
def validate(compiler):
    import hashlib
    import os
    import sys
    import tempfile
    import torch

    with tempfile.TemporaryDirectory(prefix="oa-attention-region-") as directory:
        root = Path(directory)
        for name, raw in compiler.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "env/cuda.sh").chmod(0o700)
        sys.path.insert(0, directory)
        os.environ["PYTHONPATH"] = directory
        from openassembler.modelfront import compile_model
        from torch.profiler import profile, ProfilerActivity

        torch.manual_seed(31)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

        class AttentionRegion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(4, 4)
                self.k = torch.nn.Linear(4, 4)
                self.v = torch.nn.Linear(4, 4)
                self.output = torch.nn.Linear(4, 4)
                self.norm = torch.nn.LayerNorm(4, eps=1e-5)

            def forward(self, x):
                q, k, v = self.q(x), self.k(x), self.v(x)
                scores = (q @ k.transpose(-2, -1)) * 0.5
                weights = torch.softmax(scores, dim=-1)
                mixed = weights @ v
                return x + torch.nn.functional.gelu(self.norm(self.output(mixed)), approximate="tanh")

        model = AttentionRegion()
        compiled = compile_model(model, (torch.randn(2, 3, 5, 4),), root / "bundle").cuda()
        checks = []

        def check(got, expected, label):
            got, expected = got.detach().cpu(), expected.detach().cpu()
            close = torch.isclose(got, expected, rtol=2e-4, atol=2e-5)
            result = {"label": label, "passed": bool(close.all()), "elements": got.numel(),
                      "mismatches": int((~close).sum()), "max_abs": float((got-expected).abs().max())}
            checks.append(result)
            assert result["passed"], result

        for iteration in range(4):
            x, target = torch.randn(2, 3, 5, 4).requires_grad_(), torch.randn(2, 3, 5, 4)
            model.zero_grad(set_to_none=True)
            expected = model(x)
            loss = torch.nn.functional.mse_loss(expected, target)
            loss.backward()
            gpu_x, gpu_target = x.detach().cuda(), target.cuda()
            if compiled._native_runtime is not None:
                compiled._native_runtime.poison_workspace()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                actual = compiled.native_train_step(gpu_x, gpu_target, learning_rate=0.01)
                torch.cuda.synchronize()
            check(actual["prediction"], expected, f"{iteration}/output")
            check(actual["loss"].reshape(()), loss, f"{iteration}/loss")
            name = compiled.plan["inputs"][0]["name"]
            check(actual["gradients"]["inputs"][name], x.grad, f"{iteration}/input_gradient")
            weights = dict(model.named_parameters())
            for spec in compiled.plan["parameters"]:
                check(actual["gradients"]["parameters"][spec["name"]], weights[spec["target"]].grad,
                      f"{iteration}/gradient/{spec['target']}")
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(parameter.grad, alpha=-0.01)
            for spec in compiled.plan["parameters"]:
                check(compiled.native_parameters()[spec["name"]], weights[spec["target"]],
                      f"{iteration}/state/{spec['target']}")
            assert torch.equal(gpu_x.cpu(), x.detach()) and torch.equal(gpu_target.cpu(), target)
            compiled._native_runtime.check_canaries()
        kernels = [event.name for event in prof.events() if str(event.device_type).endswith("CUDA")]
        compute = [name for name in kernels if not any(s in name.lower() for s in ("memcpy", "memset"))]
        assert compute and all("oa_" in name for name in compute), compute
        result = {"schema": 1, "passed": True, "device": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "cuda": torch.version.cuda, "checks": checks,
                  "canaries": compiled._native_runtime.check_canaries(), "compute_kernels": compute,
                  "ledger": compiled._native_runtime.launch_ledger,
                  "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in compiler.items()
                                    if name.startswith("openassembler/") and name.endswith(".py")},
                  "scope": "four updates bounded rank4 attention region with q/k/v/output Linear, transpose, two batched products, softmax, affine LayerNorm, tanh GELU and residual; no causal mask, embedding, dropout, KV cache or performance claim"}
        compiled.close()
        return result


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    result = validate.remote(runtime_files(native=True))
    path = Path(__file__).parent / ("a100-attention-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "passed": True, "checks": len(result["checks"])}))
