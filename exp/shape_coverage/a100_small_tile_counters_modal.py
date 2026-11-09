"""Bounded A100 counter follow-up for the small output-projection GEMM.

Experiment only: 128x512x512 FP16 inputs, FP32 accumulation/output. Validate
native 128/64 CTA tiles and their matched ptxas cubins against complete outputs
before fresh-context NCU replay. No profiler duration is a timing result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import modal

if modal.is_local():
    # Reuse the authoritative upload image locally. Container import happens
    # before the function can add /root/repo to sys.path, so it must not try to
    # construct the local filesystem image again during hydration.
    from exp.shape_coverage.a100_role_gemm_modal import IMAGE
else:
    IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")


ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
APP = modal.App("openassembler-small-tile-counter-followup")


def source_identity() -> dict:
    paths = [path for path in (ROOT / "openassembler").rglob("*")
             if path.is_file() and path.suffix != ".pyc"]
    paths += list((ROOT / "specs/sm_80").glob("*.json"))
    paths += [ROOT / "corpus/sm_80/pairs.jsonl", ROOT / "corpus/sm_80/machine_model.json",
              Path(__file__), HERE / "a100_role_gemm_modal.py", HERE / "gemm_policy.py",
              HERE / "small_tile_gemm.py", HERE / "profile_worker.py",
              ROOT / "exp/a100_counters/run.py"]
    return {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        check=True, capture_output=True, text=True).stdout.strip(),
            "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(set(paths))}}


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute() -> dict:
    import ctypes as C
    import glob
    import os
    import shutil
    import sys
    import tempfile

    import torch

    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/tools")
    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(
        list(args), cwd="/root/repo", capture_output=True, text=True, check=check)
    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import regcount
    from openassembler.gemm import source_for
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.isa.word import control
    from openassembler.kernels.gemm import GemmPlan
    from exp.shape_coverage.gemm_policy import verify_compile_record
    from exp.shape_coverage.profile_worker import profile_artifacts
    from exp.shape_coverage.small_tile_gemm import source_for_tile64

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("This experiment requires SM80")
    m, n, k = 128, 512, 512
    plan = GemmPlan(m, n, k, dtype="f16")
    small_source, certificate = source_for_tile64(plan)
    launches = {"tile128": plan.launch, "tile64": certificate["manual_launch"]}
    payload = {}
    statics, resources, contracts, functions, modules = {}, {}, {}, {}, {}
    checks = []
    blas = None
    handle = C.c_void_p()
    destroy = None
    try:
        with tempfile.TemporaryDirectory(prefix="oa-small-tile-counters-") as temporary:
            root = Path(temporary)
            for policy, source, cert in (("tile128", source_for(plan), None),
                                         ("tile64", small_source, certificate)):
                ptx = root / (policy + ".ptx")
                ptx.write_text(source)
                out = root / policy
                status = cli_main(["compile", str(ptx), "-o", policy, "--out-dir", str(out)])
                if status:
                    raise RuntimeError(f"compile {policy} returned {status}")
                entry = json.loads((out / "manifest.json").read_text())["entries"][0]
                record = entry["kernels"][0]
                if record.get("backend") != "openassembler":
                    raise AssertionError({"not_native": record})
                contracts[policy] = verify_compile_record(
                    plan, "m_fast" if policy == "tile128" else "tile64", source, record, cert)
                for method, artifact_key in (("oa", "cubin"), ("ptxas", "reference")):
                    label = method + "_" + policy
                    blob = (out / entry[artifact_key]).read_bytes()
                    payload[label] = blob
                    module = driver.load(blob)
                    modules[label] = module
                    function = driver.function(module, "gemm")
                    driver.ck(driver.cu.cuFuncSetAttribute(function, 8, launches[policy]["smem"]),
                              "shared-memory opt-in")
                    functions[label] = function
                    resources[label] = driver.func_attrs(function)
                    binary = root / (label + ".cubin")
                    binary.write_bytes(blob)
                    sass = subprocess.run(["nvdisasm", "-hex", "-c", str(binary)],
                                          capture_output=True, text=True, check=True).stdout
                    listing = oracle.parse_listing(sass)
                    hmma = [control(ins.word).stall for ins in listing if ins.text.startswith("HMMA")]
                    drains = [control(ins.word) for i, ins in enumerate(listing)
                              if i and ins.text.startswith("NOP")
                              and listing[i - 1].text.startswith("HMMA")]
                    if method == "oa" and (set(hmma) != {8} or not drains
                                            or any(ctl.stall != 15 or not ctl.yield_ for ctl in drains)):
                        raise AssertionError("Unexpected native HMMA8/drain15 control")
                    statics[label] = {"registers": regcount(blob, "gemm"),
                                      "instructions": len(listing), "hmma_instructions": len(hmma),
                                      "ldsm_instructions": sum(ins.text.startswith("LDSM") for ins in listing),
                                      "hmma_stall_controls": sorted(set(hmma)),
                                      "post_hmma_drains": [{"stall": ctl.stall, "yield": ctl.yield_} for ctl in drains],
                                      "cubin_sha256": hashlib.sha256(blob).hexdigest(),
                                      "sass_sha256": hashlib.sha256(sass.encode()).hexdigest()}
            paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
            blas = C.CDLL(paths[0] if paths else "libcublas.so.12")
            create, destroy = blas.cublasCreate_v2, blas.cublasDestroy_v2
            create.argtypes, create.restype = [C.POINTER(C.c_void_p)], C.c_int
            destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
            if create(C.byref(handle)):
                raise RuntimeError("cuBLAS creation failed")
            setmath = blas.cublasSetMathMode
            setmath.argtypes, setmath.restype = [C.c_void_p, C.c_int], C.c_int
            setstream = blas.cublasSetStream_v2
            setstream.argtypes, setstream.restype = [C.c_void_p, C.c_void_p], C.c_int
            if setmath(handle, 16):
                raise RuntimeError("cuBLAS precision policy failed")
            gemm = blas.cublasGemmEx
            gemm.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                            C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int,
                            C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
            gemm.restype = C.c_int
            alpha, beta = C.c_float(1), C.c_float(0)
            methods = list(payload) + ["cublas"]
            def launch(label, output):
                stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
                if label == "cublas":
                    if setstream(handle, stream):
                        raise RuntimeError("cuBLAS stream binding failed")
                    status = gemm(handle, 1, 0, n, m, k, C.byref(alpha),
                                  C.c_void_p(b.data_ptr()), 2, k, C.c_void_p(a.data_ptr()), 2, k,
                                  C.byref(beta), C.c_void_p(output.data_ptr()), 0, n, 68, 99)
                    if status:
                        raise RuntimeError(f"cuBLAS GEMM returned {status}")
                    return
                policy = label.split("_", 1)[1]
                config = launches[policy]
                args = [C.c_uint64(tensor.data_ptr()) for tensor in (a, b, output)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                driver.ck(driver.cu.cuLaunchKernel(functions[label], *config["grid"],
                                                    config["threads"], 1, 1, config["smem"],
                                                    stream, params, None), label)
            for family in ("zero", "integer", "normal"):
                torch.manual_seed(20261109)
                if family == "zero":
                    a = torch.zeros((m, k), device="cuda", dtype=torch.float16)
                    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
                elif family == "integer":
                    a = torch.randint(-2, 3, (m, k), device="cuda").half()
                    b = torch.randint(-2, 3, (n, k), device="cuda").half()
                else:
                    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
                    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
                saved_a, saved_b = a.clone(), b.clone()
                truth = a.float() @ b.float().T
                outputs = {}
                for label in methods:
                    storage = torch.full((m * n + 512,), -9876.5, device="cuda")
                    output = storage[256:-256].view(m, n)
                    output.fill_(float("nan"))
                    launch(label, output)
                    torch.cuda.synchronize()
                    outputs[label] = output.clone()
                    rel = float((output - truth).norm() / truth.norm().clamp_min(1e-30))
                    check = {"input": family, "method": label,
                             "relative_l2_fp32": rel, "poisoned_output": True,
                             "canaries": bool((storage[:256] == -9876.5).all()
                                              and (storage[-256:] == -9876.5).all()),
                             "input_immutability": torch.equal(a, saved_a) and torch.equal(b, saved_b),
                             "finite": bool(torch.isfinite(output).all())}
                    checks.append(check)
                for check in checks[-len(methods):]:
                    label = check["method"]
                    bitwise = (torch.equal(outputs[label].view(torch.int32),
                                            outputs["ptxas_tile128"].view(torch.int32))
                               if label != "cublas" else None)
                    check["baseline_ptxas_bitwise"] = bitwise
                    check["passed"] = (check["canaries"] and check["input_immutability"]
                                       and check["finite"] and check["relative_l2_fp32"] < 5e-5
                                       and bitwise is not False)
                    if not check["passed"]:
                        raise AssertionError(check)
            groups = {}
            for policy in ("tile128", "tile64"):
                selected = {label: blob for label, blob in payload.items() if label.endswith(policy)}
                grids = {label: launches[policy]["grid"] for label in selected}
                groups[policy] = profile_artifacts(selected, "gemm", grids, (m, n, k), "f16",
                                                   root / (policy + "-ncu"), smem=launches[policy]["smem"])
            return {"schema": 1, "device": driver.info().__dict__,
                    "case": {"role": "attention_output_projection", "model_case": "small_fp32_context",
                             "m": m, "n": n, "k": k, "input_dtype": "f16",
                             "accumulation_dtype": "f32", "output_dtype": "f32", "layout": "rcr"},
                    "launches": launches, "verified_compile_contracts": contracts,
                    "static": statics, "resources": resources, "checks": checks,
                    "counters": groups,
                    "methodology": {"all_output_elements": m * n, "input_families": ["zero", "integer", "normal"],
                                    "poison_before_every_validation_launch": True,
                                    "all_native_outputs_bitwise_baseline_ptxas": True,
                                    "cuBLAS": "GemmEx FP16 inputs/FP32 accumulation and output; alpha1beta0; no reduced precision reduction",
                                    "counter_inputs": "Helper's deterministic finite repeating binary fractions",
                                    "scope": "Counter replay after gates; no NCU replay-duration performance claim"},
                    "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
                    "production_promoted": False}
    finally:
        if destroy is not None and handle.value:
            destroy(handle)
        for module in modules.values():
            driver.unload(module)
        driver.close()


@APP.local_entrypoint()
def main():
    identity = source_identity()
    result = execute.remote()
    after = source_identity()
    result["source_identity"] = identity
    result["source_unchanged_during_run"] = identity == after
    if identity != after:
        result["promotion_blocker"] = "Local compiler/harness sources changed during this counter run"
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    output = HERE / ("a100-small-tile-counters_"
                     + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
