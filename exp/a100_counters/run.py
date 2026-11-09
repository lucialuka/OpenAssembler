"""A100 counter preflight and matched current-kernel validation harness.

The local entrypoint only prepares payloads and performs cheap tool checks.
The remote function validates full outputs before optional NCU replay. It is
kept under exp/ until the measurements justify a compiler change.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone

import modal

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if len(_SCRIPT.parents) > 2 else Path.cwd()
HERE = _SCRIPT.parent
EVIDENCE = ROOT / "docs/evidence/a100-mma/current-validation"
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
APP = modal.App("openassembler-a100-current-counters")

METRIC_HINTS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "dram__bytes.sum",
    "lts__t_sector_hit_rate.pct",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_payload(name: str, build_root: Path | None = None) -> dict:
    root = build_root / name if build_root is not None else EVIDENCE / name
    if build_root is not None and not (root / "manifest.json").is_file():
        aliases = (name.replace("mfast", "m_fast"), name.replace("morton", "morton8"))
        root = next((build_root / alias for alias in aliases
                     if (build_root / alias / "manifest.json").is_file()), root)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["entries"][0]
    return {
        "name": name,
        "shape": [16384, 4096, 4096],
        "dtype": entry["kernels"][0]["gemm"].get("dtype", "f16"),
        "grid": [128, 32, 1],
        "block": [128, 1, 1],
        "smem": 65536,
        "cubin": base64.b64encode((root / entry["cubin"]).read_bytes()).decode(),
        "reference": base64.b64encode((root / entry["reference"]).read_bytes()).decode(),
        "cubin_sha256": sha256(root / entry["cubin"]),
        "reference_sha256": sha256(root / entry["reference"]),
    }


def local_preflight() -> dict:
    return {
        "host": {tool: shutil.which(tool) for tool in ("ncu", "nvdisasm", "ptxas")},
        "modal_client": shutil.which("modal"),
        "image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
        "metric_hints": list(METRIC_HINTS),
    }


def build_current(out_root: Path) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype in ("f16", "bf16"):
        for order in ("m_fast", "morton8"):
            name = f"tall_{order}_{dtype}"
            out = out_root / name
            cmd = [sys.executable, "-m", "openassembler", "gemm", "--m", "16384",
                   "--n", "4096", "--k", "4096", "--dtype", dtype,
                   "--cta-order", order, "--out-dir", str(out)]
            subprocess.run(cmd, cwd=ROOT, check=True)
            rows.append({"name": name, "out": str(out),
                         "manifest_sha256": sha256(out / "manifest.json")})
    return rows


def _remote_source():
    return modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel").add_local_python_source(
        "openassembler", copy=False)


def _profile_worker() -> str:
    return textwrap.dedent(r'''
        import base64, ctypes as C, json, sys
        from openassembler.gpu.cuda_driver import Driver

        spec = json.load(open(sys.argv[1]))
        drv = Driver()
        drv.open()
        m, n, k = spec["shape"]
        item = 2 if spec.get("dtype") == "f16" else 2
        a = drv.alloc(m * k * item)
        b = drv.alloc(n * k * item)
        c = drv.alloc(m * n * 4)
        drv.memset(a, 0, m * k * item)
        drv.memset(b, 0, n * k * item)
        drv.memset(c, 0, m * n * 4)
        if spec["kind"] == "cublas":
            import glob, os
            paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
            lib = C.CDLL(paths[0] if paths else "libcublas.so.12")
            handle = C.c_void_p()
            status = lib.cublasCreate_v2(C.byref(handle))
            if status: raise RuntimeError(f"cublasCreate_v2: {status}")
            fn = lib.cublasGemmEx
            fn.restype = C.c_int
            fn.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                           C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                           C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                           C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
            alpha, beta = C.c_float(1.0), C.c_float(0.0)
            # Row-major C=A@B.T is column-major C.T=B@A.T.
            operand_type = 2 if spec.get("dtype") == "f16" else 14
            status = fn(handle, 1, 0, n, m, k, C.byref(alpha),
                        C.c_void_p(b), operand_type, k, C.c_void_p(a), operand_type, k,
                        C.byref(beta), C.c_void_p(c), 0, n, 68, 99)
            if status: raise RuntimeError(f"cublasGemmEx: {status}")
        else:
            mod = drv.load(base64.b64decode(spec["cubin"]))
            fn = drv.function(mod, "gemm")
            drv.ck(drv.cu.cuFuncSetAttribute(fn, 8, spec["smem"]), "shared memory")
            args = [C.c_uint64(a), C.c_uint64(b), C.c_uint64(c)]
            params = (C.c_void_p * len(args))(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
            drv.ck(drv.cu.cuLaunchKernel(fn, *spec["grid"], 128, 1, 1,
                                         spec["smem"], None, params, None), "profiled GEMM")
        drv.ck(drv.cu.cuCtxSynchronize(), "profile synchronization")
        print("OA_COUNTER_WORKER_DONE")
        drv.free(a); drv.free(b); drv.free(c); drv.close()
    ''')


def _query_ncu(ncu: str) -> dict:
    version = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    outputs = {}
    for name in METRIC_HINTS:
        query = subprocess.run([ncu, "--query-metrics", "--metrics", name],
                               capture_output=True, text=True)
        outputs[name] = {"returncode": query.returncode,
                         "text": (query.stdout + "\n" + query.stderr).strip()[:1200]}
    # NCU's query output prints only the device header in this build; a zero
    # return code is the reliable acceptance signal for a metric spelling.
    metrics = [name for name, result in outputs.items()
               if result["returncode"] == 0]
    return {"version": (version.stdout + version.stderr).strip(),
            "query_returncode": 0 if metrics else 1,
            "available_metrics": metrics,
            "query_results": outputs}


@APP.function(image=_remote_source(), gpu="A100-40GB", timeout=180,
              max_containers=1, retries=0)
def remote_preflight():
    import ctypes as C
    import glob
    import os
    import shutil
    import subprocess
    import torch
    from openassembler.gpu.cuda_driver import Driver

    torch.cuda.init()
    driver = Driver()
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    info = {"device": driver.info().__dict__, "ncu": ncu}
    if ncu:
        version = subprocess.run([ncu, "--version"], capture_output=True, text=True)
        query_results = {}
        for metric in METRIC_HINTS:
            query = subprocess.run([ncu, "--query-metrics", "--metrics", metric],
                                   capture_output=True, text=True, timeout=30)
            query_results[metric] = {"returncode": query.returncode,
                                     "text": (query.stdout + "\n" + query.stderr).strip()[:1200]}
        available = "\n".join(r["text"] for r in query_results.values())
        info["ncu_version"] = (version.stdout + version.stderr).strip()
        info["ncu_query_returncode"] = 0 if any(r["returncode"] == 0 for r in query_results.values()) else 1
        info["available_metric_hints"] = [m for m, r in query_results.items()
                                           if r["returncode"] == 0]
        info["ncu_query_results"] = query_results
    else:
        info["ncu_status"] = "unavailable_in_modal_image"
    cublas_paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    try:
        library = C.CDLL(cublas_paths[0] if cublas_paths else "libcublas.so.12")
        info["cublas_status"] = "loadable"
        info["cublas_path"] = cublas_paths[0] if cublas_paths else "libcublas.so.12"
        info["cublas_gemmex"] = hasattr(library, "cublasGemmEx")
    except Exception as exc:
        info["cublas_status"] = f"unavailable:{type(exc).__name__}:{exc}"
    driver.close()
    return info


@APP.function(image=_remote_source(), gpu="A100-40GB", timeout=900,
              max_containers=1, retries=0)
def execute(cases, collect_ncu: bool = True, include_cublas: bool = False):
    import os
    import ctypes as C
    import json as _json
    import csv as _csv
    import io as _io
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile
    import statistics as _statistics
    import torch
    from openassembler.gpu.cuda_driver import Driver

    torch.backends.cuda.matmul.allow_tf32 = False
    driver = Driver()
    torch.cuda.init()
    ncu = _shutil.which("ncu") or _shutil.which("nv-nsight-cu-cli")
    ncu_info = _query_ncu(ncu) if ncu else None
    results = {"device": driver.info().__dict__, "ncu": ncu,
               "ncu_info": ncu_info, "cases": [], "metric_hints": list(METRIC_HINTS)}

    def load(binary, args, label):
        module = driver.load(base64.b64decode(binary))
        fn = driver.function(module, "gemm")
        driver.ck(driver.cu.cuFuncSetAttribute(fn, 8, 65536), "shared memory")
        ptrs = [C.c_uint64(t.data_ptr()) for t in args]
        params = (C.c_void_p * len(ptrs))(*[C.cast(C.pointer(x), C.c_void_p) for x in ptrs])
        def launch():
            stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
            driver.ck(driver.cu.cuLaunchKernel(fn, 128, 32, 1, 128, 1, 1, 65536,
                                               stream, params, None), label)
        return module, launch

    for case in cases:
        dtype = torch.float16 if case["dtype"] == "f16" else torch.bfloat16
        m, n, k = case["shape"]
        a = torch.empty((m, k), device="cuda", dtype=dtype)
        b = torch.empty((n, k), device="cuda", dtype=dtype)
        out_store = torch.full((m * n + 2 * 256,), -12345.0, device="cuda")
        ref_store = torch.full_like(out_store, -12345.0)
        out = out_store[256:-256].view(m, n)
        ref = ref_store[256:-256].view(m, n)
        cublas_out = torch.empty_like(out)
        modules = []
        ours_module, ours = load(case["cubin"], [a, b, out], case["name"])
        ref_module, reference = load(case["reference"], [a, b, ref], case["name"] + ":ptxas")
        modules.extend((ours_module, ref_module))
        checks = []
        for pattern, seed in (("zero", 13), ("integer", 17), ("normal", 71)):
            torch.manual_seed(seed)
            if pattern == "zero":
                a.zero_(); b.zero_()
            elif pattern == "integer":
                a.copy_(torch.randint(-2, 3, a.shape, device="cuda").to(dtype))
                b.copy_(torch.randint(-2, 3, b.shape, device="cuda").to(dtype))
            else:
                a.normal_(); b.normal_()
            out_store.fill_(-12345.0); ref_store.fill_(-12345.0)
            out.fill_(float("nan")); ref.fill_(float("nan"))
            ours(); reference(); torch.cuda.synchronize()
            truth = a.float() @ b.float().T
            relative = float(torch.linalg.vector_norm(out - truth) /
                             torch.linalg.vector_norm(truth).clamp_min(1e-30))
            exact = bool(torch.equal(out.view(torch.int32), ref.view(torch.int32)))
            canaries = bool((out_store[:256] == -12345.0).all() and
                            (out_store[-256:] == -12345.0).all() and
                            (ref_store[:256] == -12345.0).all() and
                            (ref_store[-256:] == -12345.0).all())
            passed = (exact and canaries and bool(torch.isfinite(out).all())
                      and (pattern != "normal" or relative < 2e-5))
            checks.append({"pattern": pattern, "passed": passed, "relative_l2": relative,
                           "ptxas_bitwise_equal": exact, "canaries": canaries,
                           "elements_checked": m * n})
            if not passed:
                raise AssertionError({"case": case["name"], "check": checks[-1]})
        cublas_status = "unavailable"
        cublas_relative = None
        cublas_launch = None
        cublas_check = None
        cublas_handle = None
        cublas_destroy = None
        try:
            if not include_cublas:
                raise RuntimeError("disabled_for_matched_cubin_profile")
            handle = C.c_void_p()
            cublas_paths = __import__("glob").glob(
                "/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*"
            )
            library = C.CDLL(cublas_paths[0] if cublas_paths else "libcublas.so.12")
            create = library.cublasCreate_v2
            create.argtypes = [C.POINTER(C.c_void_p)]
            create.restype = C.c_int
            destroy = library.cublasDestroy_v2
            destroy.argtypes = [C.c_void_p]
            destroy.restype = C.c_int
            create_status = create(C.byref(handle))
            if create_status:
                raise RuntimeError(f"cublasCreate_v2: {create_status}")
            if not handle.value:
                raise RuntimeError("cublasCreate_v2 returned a null handle")
            try:
                gemmex = library.cublasGemmEx
                gemmex.restype = C.c_int
                gemmex.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                                   C.c_void_p, C.c_void_p, C.c_int, C.c_int,
                                   C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                                   C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
                alpha, beta = C.c_float(1.0), C.c_float(0.0)
                set_stream = library.cublasSetStream_v2
                set_stream.argtypes = [C.c_void_p, C.c_void_p]
                set_stream.restype = C.c_int
                stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
                stream_rc = set_stream(handle, stream)
                if stream_rc:
                    raise RuntimeError(f"cublasSetStream_v2: {stream_rc}")
                # Row-major C=A@B.T maps to column-major C.T=B@A.T.
                operand_type = 2 if case["dtype"] == "f16" else 14  # CUDA_R_16F / CUDA_R_16BF
                def cublas_launch():
                    rc = gemmex(handle, 1, 0, n, m, k, C.byref(alpha),
                                C.c_void_p(b.data_ptr()), operand_type, k,
                                C.c_void_p(a.data_ptr()), operand_type, k,
                                C.byref(beta), C.c_void_p(cublas_out.data_ptr()),
                                0, n, 68, 99)
                    if rc:
                        raise RuntimeError(f"cublasGemmEx: {rc}")
                cublas_launch()
                torch.cuda.synchronize()
                cublas_relative = float(torch.linalg.vector_norm(out - cublas_out) /
                                        torch.linalg.vector_norm(cublas_out).clamp_min(1e-30))
                cublas_truth_relative = float(torch.linalg.vector_norm(cublas_out - truth) /
                                              torch.linalg.vector_norm(truth).clamp_min(1e-30))
                cublas_max_abs = float((cublas_out - truth).abs().max())
                cublas_check = {
                    "passed": bool(torch.isfinite(cublas_out).all()) and cublas_truth_relative < 2e-5,
                    "relative_l2_vs_openassembler": cublas_relative,
                    "relative_l2_vs_fp32_reference": cublas_truth_relative,
                    "max_abs_vs_fp32_reference": cublas_max_abs,
                    "elements_checked": m * n,
                }
                if not cublas_check["passed"]:
                    raise AssertionError({"case": case["name"], "cublas_check": cublas_check})
                cublas_status = "success"
                cublas_handle = handle
                cublas_destroy = destroy
            finally:
                if cublas_status != "success":
                    destroy(handle)
        except Exception as exc:
            cublas_status = f"unavailable:{type(exc).__name__}:{exc}"
        launchers = {"openassembler": ours, "ptxas": reference}
        graphs = {}
        for label, launch in launchers.items():
            for _ in range(3):
                launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    launch()
            graph.replay(); torch.cuda.synchronize()
            graphs[label] = graph
        for _ in range(24):
            for graph in graphs.values():
                graph.replay()
        # cuBLAS graph capture is intentionally excluded: ctypes launches can
        # appear as an empty graph on this image. Time it with explicit CUDA
        # events on the configured stream and interleave it with graph samples.
        if cublas_status == "success" and cublas_launch is not None:
            for _ in range(3):
                cublas_launch()
        torch.cuda.synchronize()
        import random
        timing_samples = {label: [] for label in graphs}
        if cublas_status == "success" and cublas_launch is not None:
            timing_samples["cublas"] = []
        paired_orders = []
        rng = random.Random(119)
        for _ in range(9):
            order = list(timing_samples)
            rng.shuffle(order)
            paired_orders.append(order)
            for label in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                if label == "cublas":
                    cublas_launch()
                else:
                    graphs[label].replay()
                end.record(); end.synchronize()
                divisor = 1 if label == "cublas" else 10
                timing_samples[label].append(start.elapsed_time(end) / divisor)
        timing_medians = {label: _statistics.median(values)
                          for label, values in timing_samples.items()}
        row = {"name": case["name"], "dtype": case["dtype"],
               "cubin_sha256": case["cubin_sha256"], "reference_sha256": case["reference_sha256"],
               "checks": checks,
               "cublas_status": cublas_status, "relative_l2_vs_cublas": cublas_relative,
               "cublas_check": cublas_check,
               "timing_medians_ms": timing_medians, "timing_samples_ms": timing_samples,
               "paired_orders": paired_orders, "full_output_checked": True}
        if collect_ncu and ncu and ncu_info and ncu_info["available_metrics"]:
            with _tempfile.TemporaryDirectory() as td:
                worker_path = os.path.join(td, "profile_worker.py")
                with open(worker_path, "w") as stream:
                    stream.write(_profile_worker())
                metrics = ",".join(ncu_info["available_metrics"])
                row["ncu_profiles"] = []
                profile_specs = [("openassembler", "cubin", case["cubin"]),
                                 ("ptxas", "cubin", case["reference"])]
                if cublas_status == "success":
                    profile_specs.append(("cublas", "cublas", ""))
                for role, kind, binary in profile_specs:
                    job_path = os.path.join(td, role + ".json")
                    with open(job_path, "w") as stream:
                        _json.dump({"kind": kind, "dtype": case["dtype"], "shape": case["shape"],
                                    "cubin": binary, "grid": case["grid"],
                                    "smem": case["smem"]}, stream)
                    command = [ncu, "--clock-control", "none", "--target-processes", "all", "--csv", "--page", "raw",
                               "--metrics", metrics, sys.executable, worker_path, job_path]
                    prof = _subprocess.run(command, capture_output=True, text=True, timeout=360)
                    rows = list(_csv.reader(_io.StringIO(prof.stdout)))
                    header_index = next((i for i, fields in enumerate(rows)
                                         if fields and fields[0] == "ID"), None)
                    parsed = rows[header_index:] if header_index is not None else []
                    row["ncu_profiles"].append({"role": role,
                                                "status": "collected" if prof.returncode == 0 else "failed",
                                                "returncode": prof.returncode,
                                                "raw_rows": parsed,
                                                "stderr": prof.stderr[-8000:],
                                                "stdout_excerpt": prof.stdout[:1000]})
        elif collect_ncu:
            row["ncu_status"] = "unavailable_in_modal_image" if not ncu else "no_hinted_metrics"
        else:
            row["ncu_status"] = "disabled"
        results["cases"].append(row)
        if cublas_handle is not None:
            cublas_destroy(cublas_handle)
        for module in modules: driver.unload(module)
    driver.close()
    return results


@APP.local_entrypoint()
def main(mode: str = "preflight", out_root: str = "output/a100-counters"):
    if mode == "preflight":
        print(json.dumps({"local": local_preflight(), "remote": remote_preflight.remote()}, indent=2))
        return
    if mode == "build-current":
        print(json.dumps(build_current(ROOT / out_root), indent=2))
        return
    if mode not in ("validate", "validate-cublas", "validate-cublas-ncu"):
        raise SystemExit("mode must be preflight, build-current, validate, validate-cublas or validate-cublas-ncu")
    build_root = ROOT / out_root
    cases = [candidate_payload(name, build_root) for name in (
        "tall_mfast_f16", "tall_morton_f16", "tall_mfast_bf16", "tall_morton_bf16")]
    result = execute.remote(cases, mode in ("validate", "validate-cublas-ncu"),
                            mode in ("validate-cublas", "validate-cublas-ncu"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = HERE / f"counter-validation_{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


