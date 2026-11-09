"""Source-inclusive A100 validation for the CUTLASS-style fragment candidate.

The runner compiles the baseline and load-next/compute-current PTX through the
OpenAssembler CLI inside Modal, keeps a matching ptxas reference, and checks
full outputs, canaries, poisoned graph replay, disassembly and paired timing.
It is deliberately experiment-only; no production dispatch policy is changed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/repo/openassembler")
         .add_local_dir(str(ROOT / "specs"), "/root/repo/specs")
         .add_local_dir(str(ROOT / "corpus"), "/root/repo/corpus")
         .add_local_dir(str(ROOT / "env"), "/root/repo/env")
         .add_local_dir(str(ROOT / "tools"), "/root/repo/tools",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "exp" / "tensor_scheduler"),
                        "/root/repo/exp/tensor_scheduler",
                        ignore=["**/__pycache__", "**/*.pyc"]))
APP = modal.App("openassembler-load-next-compute-current-a100")


def _cases(*, heldout: bool = False) -> list[dict]:
    cases = [
        {"m": 128, "n": 128, "k": 128, "dtype": "f16"},
        {"m": 512, "n": 512, "k": 2048, "dtype": "f16"},
        {"m": 1024, "n": 4096, "k": 1024, "dtype": "f16"},
        {"m": 2048, "n": 512, "k": 512, "dtype": "f16"},
        {"m": 4096, "n": 1024, "k": 1024, "dtype": "f16"},
        {"m": 4096, "n": 4096, "k": 4096, "dtype": "f16"},
    ]
    if heldout:
        cases.extend([
            {"m": 2048, "n": 8192, "k": 1024, "dtype": "f16"},
            {"m": 8192, "n": 2048, "k": 1024, "dtype": "f16"},
            {"m": 8192, "n": 8192, "k": 1024, "dtype": "f16"},
            {"m": 1024, "n": 4096, "k": 1024, "dtype": "bf16"},
            {"m": 4096, "n": 1024, "k": 1024, "dtype": "bf16"},
            {"m": 2048, "n": 8192, "k": 1024, "dtype": "bf16"},
        ])
    return cases


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(cases: list[dict], rounds: int = 2, samples: int = 12) -> dict:
    import ctypes as C
    import random
    import statistics
    import shutil
    import subprocess
    import sys
    import tempfile

    import torch

    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/tools")
    os.environ.pop("OPENASSEMBLER_CUDA_HOME", None)
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(
        list(args), cwd="/root/repo", capture_output=True, text=True, check=check
    )

    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import regcount
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.kernels.gemm import GemmPlan, generate as baseline_source

    def compile_one(source: str, kernel: str, root: str) -> tuple[bytes, bytes, dict]:
        os.makedirs(root, exist_ok=True)
        ptx = os.path.join(root, kernel + ".ptx")
        out = os.path.join(root, "out")
        with open(ptx, "w") as f:
            f.write(source)
        rc = cli_main(["compile", ptx, "-o", kernel, "--out-dir", out])
        if rc:
            raise RuntimeError({"compile_rc": rc, "kernel": kernel})
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        entry = manifest["entries"][0]
        return (open(os.path.join(out, entry["cubin"]), "rb").read(),
                open(os.path.join(out, entry["reference"]), "rb").read(), manifest)

    def static(cubin: bytes, kernel: str, root: str) -> dict:
        path = os.path.join(root, kernel + ".cubin")
        open(path, "wb").write(cubin)
        proc = subprocess.run(["nvdisasm", "-c", path], capture_output=True,
                              text=True, check=True)
        lines = [line for line in proc.stdout.splitlines() if ";" in line and "/*" in line]
        return {"instructions": len(lines),
                "non_nop": sum(" NOP" not in line for line in lines),
                "registers": regcount(cubin, "gemm"),
                "sass_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                "hmma": proc.stdout.count("HMMA"),
                "ldsm": proc.stdout.count("LDSM"),
                "bar": proc.stdout.count("BAR")}

    torch.set_num_threads(2)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    driver = Driver()
    rng = random.Random(1701)
    all_rows = []
    for spec in cases:
        plan = GemmPlan(spec["m"], spec["n"], spec["k"], dtype=spec["dtype"], schedule="stagepair")
        base = baseline_source(plan)
        if (spec["m"], spec["n"], spec["k"]) in (
                (1024, 4096, 1024), (4096, 1024, 1024)):
            cand = baseline_source(GemmPlan(
                spec["m"], spec["n"], spec["k"], dtype=spec["dtype"],
                schedule="stagepair", fragment_schedule="load_next"))
            implementation = "production"
        else:
            from exp.tensor_scheduler.load_next_compute_current import candidate_source
            cand = candidate_source(plan)
            implementation = "experiment"
        with tempfile.TemporaryDirectory() as td:
            # `GemmPlan.generate` already returns a complete PTX module.
            base_oa, ptxas, _ = compile_one(base, "gemm_base", os.path.join(td, "base"))
            cand_oa, cand_ptxas, _ = compile_one(cand, "gemm_candidate", os.path.join(td, "candidate"))
            # The candidate's ptxas artifact is the independent semantic control
            # for the transformed source; baseline ptxas is the timing control.
            artifacts = {"oa_baseline": base_oa, "oa_candidate": cand_oa,
                         "ptxas_baseline": ptxas, "ptxas_candidate": cand_ptxas}
            resources = {}
            modules, funcs = {}, {}
            for name, blob in artifacts.items():
                mod = driver.load(blob)
                fn = driver.function(mod, "gemm")
                driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "shared opt-in")
                modules[name], funcs[name] = mod, fn
                resources[name] = driver.func_attrs(fn)

            m, n, k = spec["m"], spec["n"], spec["k"]
            dtype = torch.float16 if spec["dtype"] == "f16" else torch.bfloat16

            def launch(name, a, b, out):
                args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                driver.ck(driver.cu.cuLaunchKernel(
                    funcs[name], m // 128, n // 128, 1, 128, 1, 1, 65536,
                    C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), name)

            checks = []
            torch.manual_seed(90210 + m + n + k)
            reference = None
            for family in ("zero", "integer", "normal"):
                if family == "zero":
                    a = torch.zeros((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((n, k), device="cuda", dtype=dtype)
                elif family == "integer":
                    a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
                    b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype)
                else:
                    a = torch.randn((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((n, k), device="cuda", dtype=dtype)
                truth = a.float() @ b.float().T
                baseline_ref = None
                for name in ("ptxas_baseline", "oa_baseline", "oa_candidate", "ptxas_candidate"):
                    storage = torch.full((m * n + 512,), -12345.0, device="cuda")
                    out = storage[256:-256].view(m, n)
                    out.fill_(float("nan"))
                    launch(name, a, b, out)
                    torch.cuda.synchronize()
                    rel = float((out - truth).norm() / truth.norm().clamp_min(1e-30))
                    canary = bool((storage[:256] == -12345).all() and (storage[-256:] == -12345).all())
                    if name == "ptxas_baseline":
                        baseline_ref = out.clone()
                        reference = baseline_ref
                    bitwise = bool(torch.equal(out.view(torch.int32), baseline_ref.view(torch.int32)))
                    passed = canary and bool(torch.isfinite(out).all()) and rel < 5e-5 and bitwise
                    checks.append({"input": family, "method": name, "relative_l2": rel,
                                   "same_ptx_baseline_bitwise": bitwise, "canaries": canary,
                                   "passed": passed})
                    if not passed:
                        raise AssertionError({"case": spec, "check": checks[-1]})

            # Capture and replay one graph per method, poisoning outputs after capture.
            outputs = {name: torch.empty((m, n), device="cuda") for name in funcs}
            graphs = {}
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((n, k), device="cuda", dtype=dtype)
            graph_reference = torch.empty((m, n), device="cuda")
            launch("ptxas_baseline", a, b, graph_reference)
            torch.cuda.synchronize()
            for name in funcs:
                for _ in range(3):
                    launch(name, a, b, outputs[name])
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        launch(name, a, b, outputs[name])
                outputs[name].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(outputs[name].view(torch.int32), graph_reference.view(torch.int32)):
                    raise AssertionError(f"poisoned graph mismatch: {spec}/{name}")
                graphs[name] = graph

            measurements = []
            for repeat in range(rounds):
                samples_ms = {name: [] for name in funcs}
                for _ in range(samples):
                    order = list(funcs)
                    rng.shuffle(order)
                    for name in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[name].replay(); end.record(); end.synchronize()
                        samples_ms[name].append(start.elapsed_time(end) / 8)
                medians = {name: statistics.median(values) for name, values in samples_ms.items()}
                measurements.append({"round": repeat, "samples_ms": samples_ms,
                                     "median_ms": medians})

            all_rows.append({"case": spec, "checks": checks, "resources": resources,
                             "static": {name: static(blob, name, td) for name, blob in artifacts.items()},
                             "timings": measurements,
                             "candidate": "load_next_compute_current", "split_n": 4,
                             "implementation": implementation})
            for graph in graphs.values():
                del graph
            for mod in modules.values():
                driver.unload(mod)
            torch.cuda.synchronize()

    driver.close()
    return {"device": torch.cuda.get_device_name(),
            "runtime": {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                        "ptxas": subprocess.run(["ptxas", "--version"], capture_output=True,
                                                text=True, check=True).stdout.strip(),
                        "nvdisasm": subprocess.run(["nvdisasm", "--version"], capture_output=True,
                                                    text=True, check=True).stdout.strip()},
            "cases": all_rows,
            "methodology": {"full_output": True, "canaries": True,
                             "poisoned_graph": True, "ptxas_bitwise": True,
                             "rounds": rounds, "samples": samples}}


@APP.local_entrypoint()
def main(subset: str = "", rounds: int = 2, samples: int = 12,
         heldout: bool = False):
    cases = _cases(heldout=heldout)
    if subset:
        cases = [case for case in cases if subset in f"{case['m']}x{case['n']}x{case['k']}_{case['dtype']}"]
    result = execute.remote(cases, rounds, samples)
    source_paths = [path for path in (ROOT / "openassembler").rglob("*")
                    if path.is_file() and path.suffix != ".pyc"]
    source_paths += sorted((ROOT / "specs" / "sm_80").glob("*.json"))
    source_paths += [ROOT / "corpus" / "sm_80" / name
                     for name in ("pairs.jsonl", "machine_model.json")]
    source_paths += sorted(HERE.glob("*.py"))
    result["source_identity"] = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip(),
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in source_paths},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = HERE / f"load-next-compute-current_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)

