"""Durable per-case selection and frozen held-out A100 GEMM campaign.

The production tuner owns compilation, candidate admission, numerical checks,
and native paired timing. This wrapper adds checkpoint retention, independent
session provenance, and a matched selected-native/PTXAS/cuBLAS comparison.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_id(plan: dict) -> str:
    return f"{plan['m']}x{plan['n']}x{plan['k']}_{plan['dtype']}"


def cases() -> list[dict]:
    from exp.shape_coverage.role_coverage import forward_gemm_cases
    from openassembler.kernels.gemm import GemmPlan
    rows = []
    for case in forward_gemm_cases(heldout=True):
        rows.append({"id": case_id(case), "role_context": case,
                     "plan": asdict(GemmPlan(case["m"], case["n"], case["k"], dtype=case["dtype"]))})
    for m, n, k, label in (
        (16384, 4096, 4096, "legacy_tall"),
        (4096, 4096, 4096, "legacy_square"),
        (4096, 1024, 1024, "legacy_load_next_tall"),
        (64, 192, 256, "tile64_only_narrow"),
        (192, 320, 384, "tile64_only_ragged_tile_count"),
    ):
        for dtype in ("f16", "bf16"):
            plan = GemmPlan(m, n, k, dtype=dtype,
                            tile="128x128" if m % 128 == n % 128 == 0 else "64x64")
            rows.append({"id": case_id(asdict(plan)), "role_context": {"role": label},
                         "plan": asdict(plan)})
    seen = set()
    result = []
    for row in rows:
        if row["id"] not in seen:
            result.append(row)
            seen.add(row["id"])
    return result


def local_identity() -> dict:
    from openassembler.gemm_dispatch import compiler_identity
    from openassembler.transport import runtime_files
    payload = runtime_files(native=True)
    return {"compiler_identity": compiler_identity(),
            "source_sha256": {name: hashlib.sha256(data).hexdigest()
                              for name, data in sorted(payload.items())},
            "harness_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                               for name in ("campaign.py", "modal_campaign.py")},
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        check=True, capture_output=True, text=True).stdout.strip()}


def protocol(campaign: str, session: str, rows: list[dict], *, frozen: dict | None = None) -> dict:
    from openassembler.gemm_dispatch import candidate_plans
    from openassembler.kernels.gemm import GemmPlan
    identity = local_identity()
    return {"schema": 1, "campaign": campaign, "session": session,
            "created_utc": now(), "source_identity": identity,
            "rounds": 3, "samples": 8, "graph_launches": 16,
            "cases": [{**row, "candidate_names": list(candidate_plans(GemmPlan(**row["plan"])))}
                      for row in rows],
            "frozen_selection": frozen,
            "promotion_scope": "exact unbatched aligned RCR native GEMM contract on measured A100 worker",
            "not_claimed": ["whole-model speedup", "unmeasured shapes", "global fastest implementation",
                            "ptxas or tuned cuBLASLt superiority"]}


def supplemental_comparison(record: dict, archive: Path, rounds: int, samples: int) -> dict:
    """Paired selected OA / same-PTX ptxas / GemmEx controls on fresh buffers."""
    import ctypes as C
    import glob
    import random
    import zipfile
    import torch
    from openassembler.gpu.cuda_driver import Driver

    selected = record["selected"]
    candidate = record["candidates"][selected]
    plan = candidate["plan"]
    m, n, k = plan["m"], plan["n"], plan["k"]
    dtype = torch.float16 if plan["dtype"] == "f16" else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    driver = Driver()
    modules, functions = {}, {}
    from openassembler.kernels.gemm import GemmPlan
    launch_config = GemmPlan(**plan).launch
    with zipfile.ZipFile(archive) as z:
        for kind in ("native", "ptxas"):
            modules[kind] = driver.load(z.read(f"{selected}/{kind}.cubin"))
            functions[kind] = driver.function(modules[kind], "gemm")
            if launch_config["smem"] > 49152:
                driver.ck(driver.cu.cuFuncSetAttribute(functions[kind], 8, launch_config["smem"]), "shared opt-in")
    paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    blas = C.CDLL(paths[0] if paths else "libcublas.so.12")
    handle = C.c_void_p()
    create, destroy = blas.cublasCreate_v2, blas.cublasDestroy_v2
    create.argtypes, create.restype = [C.POINTER(C.c_void_p)], C.c_int
    destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
    setmath, setstream = blas.cublasSetMathMode, blas.cublasSetStream_v2
    setmath.argtypes, setmath.restype = [C.c_void_p, C.c_int], C.c_int
    setstream.argtypes, setstream.restype = [C.c_void_p, C.c_void_p], C.c_int
    gemm = blas.cublasGemmEx
    gemm.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                    C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    gemm.restype = C.c_int
    if create(C.byref(handle)) or setmath(handle, 16):
        raise ValueError("cuBLAS FP32 accumulation policy initialization failed")
    alpha, beta = C.c_float(1), C.c_float(0)
    graph_outputs, graphs = {}, {}
    methods = ["native", "ptxas", "cublas"]

    def launch(method, output):
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        if method == "cublas":
            if setstream(handle, stream):
                raise ValueError("cuBLAS stream failed")
            operand_type = 2 if dtype == torch.float16 else 14
            code = gemm(handle, 1, 0, n, m, k, C.byref(alpha),
                        C.c_void_p(b.data_ptr()), operand_type, k,
                        C.c_void_p(a.data_ptr()), operand_type, k, C.byref(beta),
                        C.c_void_p(output.data_ptr()), 0, n, 68, 99)
            if code:
                raise ValueError(f"cuBLAS GemmEx failed: {code}")
        else:
            args = [C.c_uint64(x.data_ptr()) for x in (a, b, output)]
            pointers = (C.c_void_p * 3)(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
            driver.ck(driver.cu.cuLaunchKernel(functions[method], *launch_config["grid"],
                      128, 1, 1, launch_config["smem"], stream, pointers, None), "control launch")

    checks = []
    try:
        for family_index, family in enumerate(("zero", "integer", "normal")):
            torch.manual_seed(83711 + family_index)
            if family == "zero":
                a = torch.zeros((m, k), device="cuda", dtype=dtype)
                b = torch.ones((n, k), device="cuda", dtype=dtype)
            elif family == "integer":
                a = torch.randint(-3, 4, (m, k), device="cuda").to(dtype)
                b = torch.randint(-3, 4, (n, k), device="cuda").to(dtype)
            else:
                a = torch.randn((m, k), device="cuda", dtype=dtype)
                b = torch.randn((n, k), device="cuda", dtype=dtype)
            saved_a, saved_b = a.clone(), b.clone()
            truth = a.float() @ b.float().T
            ref = torch.empty((m, n), device="cuda", dtype=torch.float32)
            launch("ptxas", ref)
            torch.cuda.synchronize()
            for method in methods:
                guarded = torch.full((m * n + 512,), -9876.5, device="cuda", dtype=torch.float32)
                out = guarded[256:-256].view(m, n)
                out.fill_(float("nan"))
                launch(method, out)
                torch.cuda.synchronize()
                error = float((out - truth).norm() / truth.norm().clamp_min(1e-30))
                bitwise = None if method == "cublas" else bool(torch.equal(out.view(torch.int32), ref.view(torch.int32)))
                check = {"input": family, "method": method, "relative_l2_fp32": error,
                         "full_output": error < 5e-5, "finite_output": bool(torch.isfinite(out).all()),
                         "canaries": bool((guarded[:256] == -9876.5).all() and (guarded[-256:] == -9876.5).all()),
                         "inputs_unchanged": bool(torch.equal(a, saved_a) and torch.equal(b, saved_b)),
                         "ptxas_bitwise": bitwise}
                check["passed"] = all(check[name] for name in ("full_output", "finite_output", "canaries", "inputs_unchanged")) and bitwise is not False
                checks.append(check)
                if not check["passed"]:
                    raise ValueError(f"matched comparison failed: {check}")
        graph_checks = []
        for method in methods:
            output = torch.empty_like(ref)
            graph_outputs[method] = output
            for _ in range(3):
                launch(method, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(16):
                    launch(method, output)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            error = float((output - truth).norm() / truth.norm().clamp_min(1e-30))
            passed = bool(torch.isfinite(output).all()) and error < 5e-5
            if method != "cublas":
                passed &= bool(torch.equal(output.view(torch.int32), ref.view(torch.int32)))
            graph_checks.append({"method": method, "passed": passed, "relative_l2_fp32": error})
            if not passed:
                raise ValueError(f"matched graph comparison failed: {method}")
            graphs[method] = graph
        rng = random.Random(73119)
        timings = []
        for round_index in range(rounds):
            values = {method: [] for method in methods}
            orders = []
            for _ in range(samples):
                order = methods[:]
                rng.shuffle(order)
                orders.append(order)
                for method in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[method].replay()
                    end.record()
                    end.synchronize()
                    values[method].append(start.elapsed_time(end) / 16)
            timings.append({"round": round_index, "paired_orders": orders, "samples_ms": values,
                            "median_ms": {method: statistics.median(v) for method, v in values.items()}})
        return {"selected_policy": selected, "checks": checks, "poisoned_graph_checks": graph_checks,
                "timings": timings, "matched_contract": True,
                "library": "cuBLAS GemmEx FP16/BF16 inputs, FP32 output/compute, alpha1 beta0, no reduced precision reduction; not tuned cuBLASLt",
                "scope": "fresh resident-input selected-kernel paired controls after candidate tuning; no packing or whole training"}
    finally:
        graphs.clear()
        torch.cuda.synchronize()
        destroy(handle)
        for module in modules.values():
            driver.unload(module)


def worker(request: Path, result: Path) -> None:
    """Fresh process per case; writes current production artifacts before exit."""
    from openassembler.gemm_tune import tune_or_load
    from openassembler.kernels.gemm import GemmPlan
    request_value = json.loads(request.read_text())
    target = result.parent / "tuning.json"
    plan = GemmPlan(**request_value["case"]["plan"])
    _, decision = tune_or_load(plan, backend="local", record=str(target),
                              rounds=request_value["rounds"], samples=request_value["samples"])
    tuned = json.loads(target.read_text())
    tuned["campaign"] = request_value["campaign"]
    tuned["session"] = request_value["session_provenance"]
    tuned["case"] = request_value["case"]
    tuned["decision"] = decision
    tuned["protocol_sha256"] = request_value["protocol_sha256"]
    tuned["comparison"] = supplemental_comparison(tuned, target.with_suffix(".artifacts.zip"),
                                                  request_value["rounds"], request_value["samples"])
    result.write_text(json.dumps(tuned, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    worker(args.request, args.result)
