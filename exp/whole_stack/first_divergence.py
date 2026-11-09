"""Untimed eager/Inductor BF16 training trajectory, with the original epsilon.

This is an application-control diagnostic, NOT native OA-generated execution.
It observes every named boundary without changing arithmetic or admission
tolerances. The first 29 batches match bf16_diagnostic.py; update 30 repeats
the last batch to add a final checkpoint without changing those 29 updates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = ROOT / "exp/whole_stack" if (ROOT / "exp/whole_stack").is_dir() else _SCRIPT.parent
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/openassembler", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(HERE), "/root/whole_stack", copy=True,
                        ignore=["**/__pycache__", "**/*.pyc", "runs/**", "native_builds/**"]))


APP = modal.App("openassembler-bf16-first-observed-divergence")


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=600, memory=16384,
              cpu=4, max_containers=1, retries=0)
def execute(steps: int = 30):
    import copy
    import sys
    import time

    import torch
    import torch._inductor.config as inductor_config
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    sys.path.insert(0, "/root/whole_stack")
    sys.path.insert(0, "/root")
    from model import TinyTransformer
    from trajectory_gate import TrajectoryGate, application_comparator, capture_training_state

    if not 1 <= steps <= 30:
        raise ValueError("bounded diagnostic supports 1 through 30 updates")
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    inductor_config.fallback_random = True
    inductor_config.compile_threads = 1

    width, batch, seq = 512, 8, 128
    initial = TinyTransformer(512, width, 8, 2, seq).cuda()
    token_pool = torch.randint(512, (batch, seq + 1), device="cuda")
    indices = [torch.randint(0, batch, (batch,), device="cuda") for _ in range(28)]
    batches = [token_pool[index] for index in indices] + [token_pool, token_pool]
    models = [copy.deepcopy(initial), copy.deepcopy(initial)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, eps=1e-8)
                  for model in models]
    graphs = []

    def compile_graph(graph, inputs):
        graphs.append({"nodes": len(list(graph.graph.nodes)),
                       "operations": [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]})
        return make_boxed_func(torch._inductor.compile(graph, inputs))

    backend = aot_autograd(fw_compiler=compile_graph, bw_compiler=compile_graph)
    compiled_forward = torch.compile(TinyTransformer.forward, backend=backend, dynamic=False, fullgraph=True)

    def full_logits(model, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = model.token(tokens) + model.position(positions).unsqueeze(0)
        for block in model.blocks:
            x = block(x)
        return model.lm_head(model.norm(x))

    compiled_full_logits = torch.compile(full_logits, backend=backend, dynamic=False, fullgraph=True)
    phases = ("pre_update_logits", "forward_loss", "raw_backward", "clipped_backward", "post_update")
    order = [(0, "initial")] + [(step, phase) for step in range(1, steps + 1) for phase in phases]
    comparator = application_comparator(rtol=.05, atol=.005, state_rtol=.05, state_atol=.005)
    ledger = TrajectoryGate(order, comparator)

    def record(step, phase, outputs, gradient_phase):
        states = [capture_training_state(model, optimizer, outputs=output,
                                        gradient_phase=gradient_phase)
                  for model, optimizer, output in zip(models, optimizers, outputs)]
        row = ledger.observe(step, phase, states[0], states[1])
        print(json.dumps({"step": step, "phase": phase, "passed": row["passed"]}), flush=True)

    record(0, "initial", [{}, {}], "missing")
    component_rows = []
    input_checks = []
    for step, tokens in enumerate(batches[:steps], start=1):
        saved_tokens = tokens.clone()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = [full_logits(models[0], tokens[:, :seq]),
                      compiled_full_logits(models[1], tokens[:, :seq])]
        record(step, "pre_update_logits", [{"full_logits": value} for value in logits], "missing")
        del logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = [models[0](tokens[:, :seq], tokens[:, 1:]),
                      compiled_forward(models[1], tokens[:, :seq], tokens[:, 1:])]
        record(step, "forward_loss", [{"loss": value} for value in losses], "missing")
        for loss in losses:
            loss.backward()
        record(step, "raw_backward", [{"loss": value} for value in losses], "raw")
        for (name, eager), (_, compiled) in zip(models[0].named_parameters(), models[1].named_parameters()):
            if name.endswith("qkv.bias"):
                section = slice(width, 2 * width)
                a, b = eager.grad[section].double(), compiled.grad[section].double()
                component_rows.append({
                    "step": step, "parameter": name, "component": "key",
                    "eager_gradient_l2": float(a.norm()), "compiled_gradient_l2": float(b.norm()),
                    "gradient_difference_l2": float((a - b).norm()),
                    "gradient_sign_disagreements": int(((a.sign() != b.sign()) & ((a != 0) | (b != 0))).sum()),
                })
        norms = [torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) for model in models]
        record(step, "clipped_backward", [{"loss": loss, "preclip_norm": norm}
                                           for loss, norm in zip(losses, norms)], "clipped")
        for optimizer in optimizers:
            optimizer.step()
        record(step, "post_update", [{"loss": loss} for loss in losses], "clipped")
        input_checks.append({"step": step, "unchanged": bool(torch.equal(tokens, saved_tokens))})
    torch.cuda.synchronize()
    report = ledger.summary()
    return {
        "schema": 1,
        "comparison": "eager PyTorch versus AOTAutograd/Inductor; no native OA-generated kernels",
        "scope": "untimed first observed failing update boundary; not instruction/operation-level causal localization",
        "policy": {"adam_epsilon": 1e-8, "lr": 3e-4, "weight_decay": .01,
                   "precision": "FP32 master parameters, BF16 autocast, TF32 disabled",
                   "gate": {"rtol": .05, "atol": .005, "state_rtol": .05, "state_atol": .005},
                   "seed": 1729, "updates": steps,
                   "batches": "first 29 match bf16_diagnostic.py; update30 repeats the final batch"},
        "shape": {"batch": batch, "sequence": seq, "width": width, "heads": 8, "layers": 2, "vocab": 512},
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
        "state_scope": "all named parameters/buffers, explicit missing gradients, raw/clipped gradients, optimizer tensors and group settings; full logits at a separate untimed pre-update boundary",
        "trajectory": report, "key_bias_gradient_diagnostics": component_rows,
        "input_immutability": input_checks, "graphs": graphs,
        "elapsed_s": time.monotonic() - started,
    }


@APP.local_entrypoint()
def main(steps: int = 30, out: str = ""):
    paths = [Path(__file__), ROOT / "exp/whole_stack/trajectory_gate.py",
             ROOT / "exp/whole_stack/model.py", ROOT / "exp/whole_stack/bf16_diagnostic.py",
             ROOT / "openassembler/torchfront/worker.py"]
    identity = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout.strip(),
                "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in paths}}
    result = execute.remote(steps)
    result["source_identity"] = identity
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path = Path(out) if out else ROOT / "exp/whole_stack/runs" / (
        "bf16-first-observed-divergence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path)
