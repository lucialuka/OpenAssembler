"""Experiment-only non-tensor instruction reduction candidate.

The first candidate removes a proven no-op address instruction emitted by the
GEMM PTX generator (`add.u32 %r30,%r30,0`). It compares the ordinary OA build,
the transformed OA build, and the matching `ptxas` controls on A100. This is a
small instruction-selection experiment, not a production rewrite.
"""

from __future__ import annotations

import base64
import json
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/repo/openassembler")
         .add_local_dir(str(ROOT / "specs"), "/root/repo/specs")
         .add_local_dir(str(ROOT / "corpus"), "/root/repo/corpus")
         .add_local_dir(str(ROOT / "env"), "/root/repo/env"))
APP = modal.App("openassembler-a100-nontensor-overhead")
SHAPES = ((512, 2048, 512), (2048, 512, 512), (1024, 4096, 1024))


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1200, max_containers=1, retries=0)
def run_remote(shapes: list[tuple[int, int, int]]) -> dict:
    import ctypes as C
    import hashlib
    import os
    import subprocess
    import torch

    import sys
    sys.path.insert(0, "/root/repo")
    os.environ.pop("OPENASSEMBLER_CUDA_HOME", None)
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    import shutil
    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(
        list(args), cwd="/root/repo", capture_output=True, text=True, check=check
    )
    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import text_words
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.kernels.gemm import GemmPlan, generate

    def build(source: str, root: str) -> tuple[bytes, bytes, dict]:
        src = os.path.join(root, "gemm.ptx")
        out = os.path.join(root, "out")
        os.makedirs(out, exist_ok=True)
        with open(src, "w") as f:
            f.write(source)
        rc = cli_main(["compile", src, "-o", "gemm", "--out-dir", out])
        if rc:
            raise RuntimeError({"compile_rc": rc, "source": src})
        manifest = json.load(open(os.path.join(out, "manifest.json")))
        entry = manifest["entries"][0]
        cubin = open(os.path.join(out, entry["cubin"]), "rb").read()
        reference = open(os.path.join(out, entry["reference"]), "rb").read()
        return cubin, reference, manifest

    def static_counts(cubin: bytes, root: str) -> dict:
        path = os.path.join(root, "x.cubin")
        open(path, "wb").write(cubin)
        proc = subprocess.run(["nvdisasm", "-c", path], capture_output=True, text=True, check=True)
        lines = [line for line in proc.stdout.splitlines() if ";" in line and "/*" in line]
        return {"instructions": len(lines), "non_nop": sum(" NOP" not in line for line in lines),
                "sass_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                "words": len(text_words(cubin, "gemm"))}

    drv = Driver(); drv.open(); torch.cuda.init()
    result = {"device": drv.info().__dict__, "cases": [],
              "candidate": "remove add.u32 %r30,%r30,0"}
    for shape in shapes:
        m, n, k = shape
        plan = GemmPlan(m, n, k, dtype="f16")
        baseline_source = generate(plan)
        candidate_source = baseline_source.replace("add.u32 %r30,%r30,0;", "")
        removed = baseline_source.count("add.u32 %r30,%r30,0;")
        with tempfile.TemporaryDirectory() as td:
            variants = {}
            for label, source in (("baseline", baseline_source), ("candidate", candidate_source)):
                root = os.path.join(td, label); os.makedirs(root)
                cubin, reference, manifest = build(source, root)
                variants[label] = {"cubin": cubin, "ptxas": reference,
                                   "static": static_counts(cubin, root),
                                   "ptxas_static": static_counts(reference, root)}

            dtype = torch.float16
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((n, k), device="cuda", dtype=dtype)
            truth = a.float() @ b.float().T
            modules, funcs = {}, {}
            for label in ("baseline", "candidate"):
                for role in ("cubin", "ptxas"):
                    name = label + ":" + role
                    module = drv.load(variants[label][role])
                    fn = drv.function(module, "gemm")
                    drv.ck(drv.cu.cuFuncSetAttribute(fn, 8, 65536), "shared memory")
                    modules[name] = module; funcs[name] = fn

            def launch(name, out):
                args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
                drv.ck(drv.cu.cuLaunchKernel(funcs[name], m // 128, n // 128, 1,
                                             128, 1, 1, 65536,
                                             C.c_void_p(torch.cuda.current_stream().cuda_stream),
                                             params, None), name)

            outputs = {}
            checks = {}
            for name in funcs:
                out = torch.full((m, n), float("nan"), device="cuda")
                launch(name, out); torch.cuda.synchronize()
                checks[name] = {
                    "relative_l2": float((out - truth).norm() / truth.norm().clamp_min(1e-30)),
                    "finite": bool(torch.isfinite(out).all()),
                }
                outputs[name] = out
            reference = outputs["baseline:ptxas"].view(torch.int32)
            for name, out in outputs.items():
                checks[name]["same_baseline_ptxas"] = bool(torch.equal(out.view(torch.int32), reference))

            timings = {}
            for name in funcs:
                out = outputs[name]
                for _ in range(3): launch(name, out)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10): launch(name, out)
                samples = []
                for _ in range(9):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); graph.replay(); end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end) / 10.0)
                timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}

            result["cases"].append({"shape": shape, "removed_ptx_noops": removed,
                                    "static": {k: v["static"] for k, v in variants.items()},
                                    "ptxas_static": {k: v["ptxas_static"] for k, v in variants.items()},
                                    "checks": checks, "timings": timings})
            for module in modules.values(): drv.unload(module)
    drv.close()
    return result


@APP.local_entrypoint()
def main():
    result = run_remote.remote(list(SHAPES))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = HERE / f"non_tensor_overhead_{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)

