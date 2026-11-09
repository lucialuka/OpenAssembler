"""Payload-driven Nsight Compute replay of already validated role GEMMs.

``profile_artifacts`` is called *after* the numerical gates in the Modal role
runner. Each profiler invocation starts a fresh driver-only Python subprocess;
the worker does not import PyTorch or share its context. It warms one GEMM and
profiles one further GEMM inside cuProfilerStart/Stop. All kernels of that one
cuBLAS call are retained, rather than assuming cuBLAS launches exactly one.

NCU replay duration is never a performance measurement. Original CSV, stderr,
metric queries and version output are written to ``output_dir`` and included in
the return value so an ephemeral remote caller can retain them locally.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXTRA_METRICS = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "lts__t_sectors.sum",
)


def authoritative_metrics() -> tuple[list[str], dict[str, str]]:
    """Read the authoritative experiment list without importing its Modal app."""
    source = ROOT / "exp/a100_counters/run.py"
    content = source.read_bytes()
    for node in ast.parse(content, filename=str(source)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "METRIC_HINTS"
            for target in node.targets
        ):
            hints = ast.literal_eval(node.value)
            if not isinstance(hints, (list, tuple)) or not all(isinstance(x, str) for x in hints):
                raise ValueError("authoritative METRIC_HINTS is not a string sequence")
            return list(dict.fromkeys((*hints, *EXTRA_METRICS))), {
                "path": "exp/a100_counters/run.py",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    raise ValueError("authoritative METRIC_HINTS was not found")


def _number(raw: str) -> float | int | None:
    value = raw.strip().replace(",", "").replace("\u00a0", "")
    if not value or value.lower() in {"n/a", "nan", "inf", "-inf", "not supported"}:
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        parsed = float(value)
        return parsed if __import__("math").isfinite(parsed) else None
    except ValueError:
        return None


def parse_ncu_csv(content: str, requested: Sequence[str]) -> dict[str, Any]:
    """Parse NCU raw wide tables and metric-per-row CSV, preserving units.

    A missing or nonnumeric metric is represented explicitly, never as zero.
    Percentages and byte/sector counters are not combined across kernels.
    Quoted thousands separators are supported by the CSV reader.
    """
    rows = list(csv.reader(io.StringIO(content)))
    kernels: dict[tuple[str, str, str], dict[str, Any]] = {}
    header: list[str] | None = None
    units: dict[str, str] = {}
    for row in rows:
        if row and row[0].lstrip("\ufeff") == "ID" and "Kernel Name" in row:
            header = [value.lstrip("\ufeff") for value in row]
            units = {}
            continue
        if not header or len(row) != len(header):
            continue
        record = dict(zip(header, row))
        if not record.get("ID", "").strip():
            units = record
            continue
        if not record["ID"].isdigit():
            continue
        identity = (record.get("Process ID", ""), record["ID"], record.get("Kernel Name", ""))
        kernel = kernels.setdefault(identity, {"process_id": identity[0], "id": identity[1],
                                               "kernel": identity[2], "metrics": {}})
        if "Metric Name" in record:
            name = record["Metric Name"]
            if name in requested:
                raw = record.get("Metric Value", "")
                kernel["metrics"][name] = {"value": _number(raw),
                                            "unit": record.get("Metric Unit", ""), "raw": raw}
        else:
            for name in requested:
                if name in record:
                    raw = record[name]
                    kernel["metrics"][name] = {"value": _number(raw),
                                                "unit": units.get(name, ""), "raw": raw}
    for kernel in kernels.values():
        kernel["missing_metrics"] = [name for name in requested
                                     if kernel["metrics"].get(name, {}).get("value") is None]
    observed = {name for kernel in kernels.values() for name, item in kernel["metrics"].items()
                if item["value"] is not None}
    return {"kernels": list(kernels.values()),
            "missing_metrics": [name for name in requested if name not in observed]}


def _capture(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"command": command, "returncode": run.returncode,
                "stdout": run.stdout, "stderr": run.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        def string(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return {"command": command, "returncode": None, "stdout": string(error.stdout),
                "stderr": string(error.stderr), "timed_out": True}


def profile_artifacts(
    artifacts: Mapping[str, bytes], kernel: str, grids: Mapping[str, Sequence[int]],
    shape: Sequence[int], dtype: str, output_dir: str | Path, *,
    smem: int = 65536, timeout: int = 360,
) -> dict[str, Any]:
    """Profile selected OA/PTXAS cubins and a matched FP32-output cuBLAS call.

    Input ``artifacts`` maps descriptive method labels to their validated cubin
    bytes; ``grids`` maps the same labels to two/three launch dimensions. All use
    ``kernel`` (normally ``gemm``), four warps and the supplied dynamic SMEM.
    ``shape`` is M/N/K of contiguous A[M,K] @ B[N,K].T; dtype is f16 or bf16.
    Caller must retain the returned raw payload on the local machine.
    """
    if (len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape)
            or dtype not in {"f16", "bf16"}):
        raise ValueError("expected positive M/N/K and f16 or bf16")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        metrics, origin = authoritative_metrics()
    except (OSError, ValueError, SyntaxError) as error:
        return {"status": "metric_manifest_unavailable", "reason": str(error), "profiles": []}
    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if not ncu:
        candidates = sorted(Path("/opt/nvidia").glob("nsight-compute*/ncu"))
        candidates += sorted(Path("/usr/local").glob("cuda*/bin/ncu"))
        ncu = next((str(path) for path in candidates if path.is_file()), None)
    result: dict[str, Any] = {"schema": 1, "ncu": ncu, "metric_origin": origin,
                              "metrics_requested": metrics, "profiles": []}
    if not ncu:
        result.update(status="unavailable", reason="Nsight Compute executable not found",
                      missing_metrics=metrics)
        (root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    version = _capture([ncu, "--version"])
    queries = {metric: _capture([ncu, "--query-metrics", "--metrics", metric], 30)
               for metric in metrics}
    available = [metric for metric, query in queries.items() if query["returncode"] == 0]
    result.update(version=version, metric_queries=queries, available_metrics=available,
                  unavailable_metrics=[metric for metric in metrics if metric not in available])
    (root / "ncu-version.json").write_text(json.dumps(version, indent=2) + "\n")
    (root / "ncu-metric-queries.json").write_text(json.dumps(queries, indent=2) + "\n")
    if not available:
        result.update(status="no_supported_metrics", missing_metrics=metrics)
        (root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    specs = []
    for label, blob in artifacts.items():
        if label not in grids:
            raise ValueError(f"missing launch grid for {label}")
        grid = list(grids[label])
        if len(grid) == 2:
            grid.append(1)
        if len(grid) != 3 or any(type(x) is not int or x <= 0 for x in grid):
            raise ValueError(f"invalid launch grid for {label}")
        digest = hashlib.sha256(blob).hexdigest()
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", label) + "_" + digest[:12]
        binary_path = root / (name + ".cubin")
        binary_path.write_bytes(blob)
        specs.append({"label": label, "file_stem": name, "kind": "cubin",
                      "cubin_path": str(binary_path.resolve()), "cubin_sha256": digest,
                      "kernel": kernel, "grid": grid})
    specs.append({"label": "cublas", "file_stem": "cublas", "kind": "cublas"})
    for spec in specs:
        spec.update(shape=list(shape), dtype=dtype, smem=smem,
                    input_pattern="repeating_balanced_exact_binary_fractions",
                    accumulation_dtype="f32", output_dtype="f32", alpha=1, beta=0)
        job = root / (spec["file_stem"] + ".job.json")
        job.write_text(json.dumps(spec, indent=2) + "\n")
        command = [ncu, "--clock-control", "none", "--target-processes", "all",
                   "--profile-from-start", "off", "--csv", "--page", "raw",
                   "--metrics", ",".join(available), sys.executable,
                   str(Path(__file__).resolve()), "--worker", str(job.resolve())]
        captured = _capture(command, timeout)
        parsed = parse_ncu_csv(captured["stdout"], available)
        text = captured["stdout"] + "\n" + captured["stderr"]
        if captured["timed_out"]:
            status = "timeout"
        elif "ERR_NVGPUCTRPERM" in text or re.search(r"permission|not permitted", text, re.I):
            status = "permission_denied"
        elif captured["returncode"] != 0:
            status = "failed"
        elif not parsed["kernels"]:
            status = "no_kernel_metrics"
        elif parsed["missing_metrics"] or any(row["missing_metrics"] for row in parsed["kernels"]):
            status = "partial_metrics"
        else:
            status = "collected"
        (root / (spec["file_stem"] + ".csv")).write_text(captured["stdout"])
        (root / (spec["file_stem"] + ".stderr.txt")).write_text(captured["stderr"])
        result["profiles"].append({"label": spec["label"], "status": status,
                                    "spec": spec, **captured, **parsed,
                                    "raw_csv": captured["stdout"]})
    result.update(status="collected" if all(row["status"] == "collected"
                                             for row in result["profiles"]) else "incomplete",
                  scope="One warm GEMM, then all kernels from one profiled GEMM call; "
                        "fresh driver context, deterministic finite operands, no torch; "
                        "NCU replay counters are separate from paired benchmark timing.")
    (root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_worker(spec: Mapping[str, Any]) -> None:
    """Driver-only subprocess entry; no parent CUDA state or Torch import."""
    import ctypes as C
    import ctypes.util
    import glob
    import struct

    sys.path.insert(0, str(ROOT))
    from openassembler.gpu.cuda_driver import Driver
    driver = Driver()
    driver.open()
    a = b = c = module = handle = None
    destroy = None
    try:
        m, n, k = spec["shape"]
        if driver.info().cc != "8.0":
            raise RuntimeError("SM80 required for this A100 experiment")
        a, b, c = driver.alloc(m * k * 2), driver.alloc(n * k * 2), driver.alloc(m * n * 4)
        def encoded(count: int, offset: int) -> bytes:
            values = [((i + offset) % 17 - 8) / 32 for i in range(4096)]
            if spec["dtype"] == "f16":
                block = b"".join(struct.pack("<e", value) for value in values)
            else:
                block = b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0])[2:]
                                  for value in values)
            size = count * 2
            return block * (size // len(block)) + block[:size % len(block)]
        driver.htod(a, encoded(m * k, 0))
        driver.htod(b, encoded(n * k, 3))
        driver.memset(c, 0, m * n * 4)
        if spec["kind"] == "cublas":
            paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
            library = C.CDLL(paths[0] if paths else ctypes.util.find_library("cublas") or "libcublas.so.12")
            create, destroy = library.cublasCreate_v2, library.cublasDestroy_v2
            create.argtypes, create.restype = [C.POINTER(C.c_void_p)], C.c_int
            destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
            handle = C.c_void_p()
            status = create(C.byref(handle))
            if status:
                raise RuntimeError(f"cublasCreate_v2: {status}")
            setmath = library.cublasSetMathMode
            setmath.argtypes, setmath.restype = [C.c_void_p, C.c_int], C.c_int
            status = setmath(handle, 16)  # Disallow reduced-precision reduction.
            if status:
                raise RuntimeError(f"cublasSetMathMode: {status}")
            fn = library.cublasGemmEx
            fn.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                           C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int,
                           C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
            fn.restype = C.c_int
            alpha, beta = C.c_float(1), C.c_float(0)
            operand = 2 if spec["dtype"] == "f16" else 14
            def launch():
                status = fn(handle, 1, 0, n, m, k, C.byref(alpha), C.c_void_p(b), operand, k,
                            C.c_void_p(a), operand, k, C.byref(beta), C.c_void_p(c), 0, n, 68, 99)
                if status:
                    raise RuntimeError(f"cublasGemmEx: {status}")
                driver.ck(driver.cu.cuCtxSynchronize(), "cuBLAS synchronize")
        else:
            binary = Path(spec["cubin_path"]).read_bytes()
            if hashlib.sha256(binary).hexdigest() != spec["cubin_sha256"]:
                raise RuntimeError("profile cubin hash mismatch")
            module = driver.load(binary)
            fn = driver.function(module, spec["kernel"])
            args = [C.c_uint64(pointer) for pointer in (a, b, c)]
            def launch():
                driver.launch(fn, spec["grid"], (128, 1, 1), args, smem=spec["smem"])
        launch()
        driver.ck(driver.cu.cuProfilerStart(), "cuProfilerStart")
        launch()
        driver.ck(driver.cu.cuProfilerStop(), "cuProfilerStop")
        print("OA_ROLE_PROFILE_WORKER_DONE")
    finally:
        if handle is not None and destroy is not None:
            destroy(handle)
        if module is not None:
            driver.unload(module)
        for pointer in (a, b, c):
            if pointer is not None:
                driver.free(pointer)
        driver.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_worker(json.loads(Path(sys.argv[2]).read_text()))
    else:
        raise SystemExit("Use profile_artifacts(...) after correctness validation, or --worker JOB.json")
