"""Bounded pinned-A100 validation of the public Python native-model API."""
from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App("openassembler-native-python-api")
image = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").pip_install(
    "torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")


@app.function(image=image, gpu="A100-40GB", timeout=900, max_containers=1, retries=0)
def validate(compiler):
    import hashlib
    import os
    import sys
    import tempfile
    import torch

    with tempfile.TemporaryDirectory(prefix="oa-native-api-") as directory:
        root = Path(directory)
        for name, raw in compiler.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "env/cuda.sh").chmod(0o700)
        sys.path.insert(0, directory)
        os.environ["PYTHONPATH"] = directory
        from openassembler.modelfront import compile_model, ModelGraphError

        torch.manual_seed(17)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(5, 7)
                self.last = torch.nn.Linear(7, 5)

            def forward(self, x):
                y = self.last(torch.relu(self.first(x)))
                return x + y * 0.25

        model = Residual()
        compiled = compile_model(model, (torch.randn(4, 5),), root / "bundle").cuda()
        checks, guards = [], []

        def check(got, expected, label):
            actual = got.detach().cpu()
            expected = expected.detach().cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            checks.append({"label": label, "elements": actual.numel(),
                           "max_abs": float((actual-expected).abs().max()), "passed": True})

        for iteration in range(4):
            x = torch.randn(4, 5).requires_grad_()
            target = torch.randn(4, 5)
            model.zero_grad(set_to_none=True)
            expected = model(x)
            loss = torch.nn.functional.mse_loss(expected, target)
            loss.backward()
            gpu_x, gpu_target = x.detach().cuda(), target.cuda()
            before_input = gpu_x.cpu().clone()
            if compiled._native_runtime is not None:
                compiled._native_runtime.poison_workspace()
            result = compiled.native_train_step(gpu_x, gpu_target, learning_rate=0.025)
            check(result["prediction"], expected, f"{iteration}/output")
            check(result["loss"].reshape(()), loss, f"{iteration}/loss")
            input_name = compiled.plan["inputs"][0]["name"]
            check(result["gradients"]["inputs"][input_name], x.grad, f"{iteration}/input_grad")
            originals = dict(model.named_parameters())
            for spec in compiled.plan["parameters"]:
                check(result["gradients"]["parameters"][spec["name"]], originals[spec["target"]].grad,
                      f"{iteration}/parameter_grad/{spec['target']}")
            with torch.no_grad():
                for value in model.parameters():
                    value.add_(value.grad, alpha=-0.025)
            for spec in compiled.plan["parameters"]:
                check(compiled.native_parameters()[spec["name"]], originals[spec["target"]],
                      f"{iteration}/parameter/{spec['target']}")
            assert torch.equal(gpu_x.cpu(), before_input)
            guards.append(compiled._native_runtime.check_canaries())
            if iteration == 1:
                # Versioned external mutation is observed on the next native
                # step; current values are rebound and matrix packs are rebuilt.
                with torch.no_grad():
                    delta = torch.full_like(model.first.bias, 0.01)
                    model.first.bias.add_(delta)
                    spec = next(s for s in compiled.plan["parameters"] if s["target"] == "first.bias")
                    compiled.native_parameters()[spec["name"]].copy_(model.first.bias.cuda())
        shape_rejected = False
        try:
            compiled.native_train_step(torch.zeros(3, 5, device="cuda"), gpu_target)
        except ValueError:
            shape_rejected = True
        assert shape_rejected

        # Interoperability is a separate check. Explicit contiguous output
        # derivatives avoid implicit vendor reduction/expanded-gradient work.
        compiled.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        x = torch.randn(4, 5).requires_grad_()
        gpu_x = x.detach().cuda().requires_grad_()
        seed = torch.randn(4, 5)
        expected = model(x)
        expected.backward(seed)
        actual = compiled(gpu_x)
        torch.autograd.backward((actual,), (seed.cuda(),))
        check(actual, expected, "autograd/output")
        check(gpu_x.grad, x.grad, "autograd/input_grad")
        for spec in compiled.plan["parameters"]:
            check(compiled.native_parameters()[spec["name"]].grad,
                  dict(model.named_parameters())[spec["target"]].grad, f"autograd/{spec['target']}")

        inference_pointers = []
        for iteration in range(3):
            cpu_input = torch.randn(4, 5)
            gpu_input = cpu_input.cuda()
            if iteration == 1:
                with torch.no_grad():
                    model.first.bias.add_(0.03)
                    spec = next(s for s in compiled.plan["parameters"] if s["target"] == "first.bias")
                    attr = compiled._state_attributes[compiled._state_names.index(spec["name"])]
                    setattr(compiled, attr, torch.nn.Parameter(model.first.bias.detach().cuda()))
            if compiled._native_runtime is not None:
                compiled._native_runtime.poison_workspace()
            actual = compiled.native_forward(gpu_input)
            assert actual.grad_fn is None and not actual.requires_grad
            inference_pointers.append(actual.data_ptr())
            with torch.no_grad():
                expected = model(cpu_input)
            check(actual, expected, f"native_forward/{iteration}")
            assert torch.equal(gpu_input.cpu(), cpu_input)
            compiled._native_runtime.check_canaries()
        assert len(set(inference_pointers)) == 1

        from torch.profiler import profile, ProfilerActivity
        # Storage transfers and independent CPU reference are outside this
        # profile. Every computational CUDA launch must be an OA entry.
        inputs = torch.randn(4, 5).cuda()
        target = torch.randn(4, 5).cuda()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            compiled.native_train_step(inputs, target, learning_rate=0.01)
            torch.cuda.synchronize()
        gpu_events = [event.name for event in prof.events() if str(event.device_type).endswith("CUDA")]
        compute = [name for name in gpu_events if not any(t in name.lower() for t in ("memcpy", "memset"))]
        assert compute and all("oa_" in name for name in compute), compute
        from openassembler.modelfront.runtime import audit_profiler_events
        training_audit = audit_profiler_events(prof.events(), compiled._native_runtime.launch_ledger)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as inference_prof:
            compiled.native_forward(inputs)
            torch.cuda.synchronize()
        inference_audit = audit_profiler_events(inference_prof.events(), compiled._native_runtime.launch_ledger)
        runtime = compiled._native_runtime
        result = {"schema": 1, "passed": True, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "device": torch.cuda.get_device_name(), "checks": checks, "guards": guards,
                  "shape_rejected": shape_rejected, "profile_compute_kernels": compute,
                  "native_forward_reuses_output_storage": True,
                  "training_profile_audit": training_audit, "inference_profile_audit": inference_audit,
                  "vendor_compute_kernels": [], "launch_ledger": runtime.launch_ledger,
                  "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in compiler.items()
                                    if name.startswith("openassembler/") and name.endswith(".py")},
                  "scope": "raw Python compile_model/native_forward/native_train_step FP32 API; inference storage reuse and live Parameter replacement; conventional autograd checked separately with contiguous seeds; no performance claim"}
        compiled.close()
        return result


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone
    from openassembler.transport import runtime_files
    result = validate.remote(runtime_files(native=True))
    destination = Path(__file__).parent / ("a100-api-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination), "passed": result["passed"], "checks": len(result["checks"])}))
