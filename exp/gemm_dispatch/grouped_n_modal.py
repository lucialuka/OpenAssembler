"""A100 validation of CUTLASS-style grouped-N CTA traversal.

This experiment compares the current m-fast OA source, a grouped-N prologue
rewrite, and matching ptxas controls. The candidate changes only logical CTA
traversal and launch geometry; all numerical and resource gates are retained.
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
         .add_local_dir(str(ROOT / "exp" / "gemm_dispatch"), "/root/repo/exp/gemm_dispatch",
                        ignore=["**/__pycache__", "**/*.pyc"]))
APP = modal.App("openassembler-grouped-n-a100")


def _cases(*, heldout: bool = False) -> list[dict]:
    cases = [
        {"m": 1024, "n": 4096, "k": 1024, "dtype": "f16", "width": 8},
        {"m": 4096, "n": 4096, "k": 4096, "dtype": "f16", "width": 8},
        {"m": 16384, "n": 4096, "k": 4096, "dtype": "f16", "width": 8},
    ]
    if heldout:
        cases.extend([
            {"m": 8192, "n": 4096, "k": 4096, "dtype": "f16", "width": 8},
            {"m": 32768, "n": 4096, "k": 4096, "dtype": "f16", "width": 8},
            {"m": 16384, "n": 2048, "k": 2048, "dtype": "f16", "width": 8},
            {"m": 16384, "n": 4096, "k": 4096, "dtype": "bf16", "width": 8},
        ])
    return cases


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(cases: list[dict], rounds: int = 2, samples: int = 10) -> dict:
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
    from openassembler.kernels.gemm import GemmPlan, generate

    def compile_one(source: str, kernel: str, root: str) -> tuple[bytes, bytes]:
        os.makedirs(root, exist_ok=True)
        ptx = os.path.join(root, kernel + ".ptx")
        out = os.path.join(root, "out")
        open(ptx, "w").write(source.replace(".entry gemm(", f".entry {kernel}(", 1))
        rc = cli_main(["compile", ptx, "-o", kernel, "--out-dir", out])
        if rc:
            raise RuntimeError({"compile_rc": rc, "kernel": kernel})
        entry = json.load(open(os.path.join(out, "manifest.json")))["entries"][0]
        return (open(os.path.join(out, entry["cubin"]), "rb").read(),
                open(os.path.join(out, entry["reference"]), "rb").read())

    def static(blob: bytes, kernel: str, root: str) -> dict:
        path = os.path.join(root, kernel + ".cubin")
        open(path, "wb").write(blob)
        sass = subprocess.run(["nvdisasm", "-c", path], capture_output=True,
                              text=True, check=True).stdout
        lines = [line for line in sass.splitlines() if ";" in line and "/*" in line]
        return {"instructions": len(lines), "non_nop": sum(" NOP" not in x for x in lines),
                "registers": regcount(blob, kernel),
                "hmma": sass.count("HMMA"), "ldsm": sass.count("LDSM"),
                "sass_sha256": hashlib.sha256(sass.encode()).hexdigest()}

    torch.set_num_threads(2)
    torch.cuda.init(); _ = torch.empty(1, device="cuda")
    driver = Driver(); rng = random.Random(2108)
    rows = []
    for spec in cases:
        plan = GemmPlan(spec["m"], spec["n"], spec["k"], dtype=spec["dtype"], schedule="stagepair")
        baseline = generate(plan)
        morton = generate(GemmPlan(spec["m"], spec["n"], spec["k"],
                                   dtype=spec["dtype"], schedule="stagepair",
                                   cta_order="morton8"))
        grouped_plan = GemmPlan(spec["m"], spec["n"], spec["k"],
                                dtype=spec["dtype"], schedule="stagepair",
                                cta_order="grouped_n8")
        grouped = generate(grouped_plan)
        mapping = {"cta_order": "grouped_n8", "width": 8,
                   "logical_tiles": [spec["m"] // 128, spec["n"] // 128],
                   "launch_grid": grouped_plan.launch["grid"],
                   "mapping": "logical_m=blockIdx.x>>3; logical_n=(blockIdx.y<<3)|(blockIdx.x&7)"}
        with tempfile.TemporaryDirectory() as td:
            artifacts = {}
            for label, source in (("oa_m_fast", baseline), ("oa_morton", morton),
                                  ("oa_grouped_n", grouped)):
                artifacts[label], _ = compile_one(source, label, os.path.join(td, label))
            # ptxas references are built from the exact source variants.
            for label, source in (("ptxas_m_fast", baseline), ("ptxas_morton", morton),
                                  ("ptxas_grouped_n", grouped)):
                _, artifacts[label] = compile_one(source, label, os.path.join(td, label))
            funcs, modules, resources = {}, {}, {}
            for label, blob in artifacts.items():
                mod = driver.load(blob); fn = driver.function(mod, label)
                driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "shared opt-in")
                modules[label], funcs[label] = mod, fn; resources[label] = driver.func_attrs(fn)
            m, n, k = spec["m"], spec["n"], spec["k"]
            dtype = torch.float16 if spec["dtype"] == "f16" else torch.bfloat16
            grid = {"oa_m_fast": (m // 128, n // 128), "ptxas_m_fast": (m // 128, n // 128),
                    "oa_morton": (m // 128, n // 128), "ptxas_morton": (m // 128, n // 128),
                    "oa_grouped_n": (grouped_plan.launch["grid"][0], grouped_plan.launch["grid"][1]),
                    "ptxas_grouped_n": (grouped_plan.launch["grid"][0], grouped_plan.launch["grid"][1])}

            def launch(label, a, b, out):
                args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                gx, gy = grid[label]
                driver.ck(driver.cu.cuLaunchKernel(
                    funcs[label], gx, gy, 1, 128, 1, 1, 65536,
                    C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None), label)

            torch.manual_seed(901 + m + n + k)
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((n, k), device="cuda", dtype=dtype)
            truth = a.float() @ b.float().T
            checks = []
            references = {}
            for label in ("ptxas_m_fast", "ptxas_morton", "ptxas_grouped_n",
                          "oa_m_fast", "oa_morton", "oa_grouped_n"):
                storage = torch.full((m * n + 512,), -12345.0, device="cuda")
                out = storage[256:-256].view(m, n); out.fill_(float("nan"))
                launch(label, a, b, out); torch.cuda.synchronize()
                if label == "ptxas_m_fast": references["m_fast"] = out.clone()
                if label == "ptxas_morton": references["morton"] = out.clone()
                if label == "ptxas_grouped_n": references["grouped_n"] = out.clone()
                layout = ("grouped_n" if "grouped_n" in label else
                           "morton" if "morton" in label else "m_fast")
                ref = references[layout]
                rel = float((out - truth).norm() / truth.norm().clamp_min(1e-30))
                canary = bool((storage[:256] == -12345).all() and (storage[-256:] == -12345).all())
                bitwise = bool(torch.equal(out.view(torch.int32), ref.view(torch.int32)))
                passed = canary and bool(torch.isfinite(out).all()) and rel < 5e-5 and bitwise
                checks.append({"method": label, "relative_l2": rel,
                               "same_layout_ptxas_bitwise": bitwise, "canaries": canary,
                               "passed": passed})
                if not passed: raise AssertionError({"case": spec, "check": checks[-1]})

            outputs = {label: torch.empty((m, n), device="cuda") for label in funcs}
            graph_ref = {"m_fast": torch.empty((m, n), device="cuda"),
                         "morton": torch.empty((m, n), device="cuda"),
                         "grouped_n": torch.empty((m, n), device="cuda")}
            launch("ptxas_m_fast", a, b, graph_ref["m_fast"])
            launch("ptxas_morton", a, b, graph_ref["morton"])
            launch("ptxas_grouped_n", a, b, graph_ref["grouped_n"])
            torch.cuda.synchronize()
            graphs = {}
            for label in funcs:
                for _ in range(3): launch(label, a, b, outputs[label])
                torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8): launch(label, a, b, outputs[label])
                outputs[label].fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                layout = ("grouped_n" if "grouped_n" in label else
                           "morton" if "morton" in label else "m_fast")
                ref = graph_ref[layout]
                if not torch.equal(outputs[label].view(torch.int32), ref.view(torch.int32)):
                    raise AssertionError(f"poisoned graph mismatch: {spec}/{label}")
                graphs[label] = graph
            timing = []
            for repeat in range(rounds):
                samples_ms = {label: [] for label in funcs}
                for _ in range(samples):
                    order = list(funcs); rng.shuffle(order)
                    for label in order:
                        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[label].replay(); end.record(); end.synchronize()
                        samples_ms[label].append(start.elapsed_time(end) / 8)
                timing.append({"round": repeat, "median_ms": {k: statistics.median(v) for k, v in samples_ms.items()},
                               "samples_ms": samples_ms})
            rows.append({"case": spec, "mapping": mapping, "checks": checks, "resources": resources,
                         "static": {label: static(blob, label, td) for label, blob in artifacts.items()},
                         "timings": timing})
            for mod in modules.values(): driver.unload(mod)
            torch.cuda.synchronize()
    driver.close()
    return {"device": torch.cuda.get_device_name(),
            "runtime": {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                        "ptxas": subprocess.run(["ptxas", "--version"], capture_output=True,
                                                text=True, check=True).stdout.strip(),
                        "nvdisasm": subprocess.run(["nvdisasm", "--version"], capture_output=True,
                                                    text=True, check=True).stdout.strip()},
            "cases": rows,
            "methodology": {"full_output": True, "canaries": True,
                             "poisoned_graph": True, "layout_matched_ptxas": True,
                             "rounds": rounds, "samples": samples}}


@APP.local_entrypoint()
def main(subset: str = "", rounds: int = 2, samples: int = 10,
         heldout: bool = False):
    cases = _cases(heldout=heldout)
    if subset:
        cases = [case for case in cases if subset in f"{case['m']}x{case['n']}x{case['k']}"]
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
    out = HERE / f"grouped-n_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)

