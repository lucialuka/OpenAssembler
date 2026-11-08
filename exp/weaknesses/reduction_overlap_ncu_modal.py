"""Nsight Compute counters for the validated row-reduction overlap candidate.

This is an experiment-only counter pass.  It rebuilds the ordinary and
streamed-load/FMA rowwise kernels from source, checks each candidate in a
matched CUDA-runtime worker, and profiles the third exact kernel launch with
Nsight Compute.  A profile is retained only when the pre- and post-profile
outputs, canaries, and input immutability checks pass.

The run intentionally reports raw NCU rows as well as normalized metrics.  NCU
replay is diagnostic and must not be confused with the graph timing records in
``reduction_overlap_run_*.json``.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import textwrap

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
         .add_local_dir(str(ROOT / "exp" / "nn_round2"), "/root/repo/exp/nn_round2",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "exp" / "weaknesses"), "/root/repo/exp/weaknesses",
                        ignore=["**/__pycache__", "**/*.pyc"]))
APP = modal.App("openassembler-reduction-overlap-ncu-a100")


METRICS = [
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_xu.sum",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
]


def _parse_ncu_csv(stdout: str) -> dict:
    """Parse NCU raw wide or long CSV while retaining the original rows."""

    records = list(csv.reader(io.StringIO(stdout)))
    wide = next((i for i, row in enumerate(records)
                 if row and row[0] == "ID" and "Kernel Name" in row
                 and any("__" in name for name in row)), None)
    long = next((i for i, row in enumerate(records)
                 if "Metric Name" in row and "Metric Value" in row), None)
    metrics: dict[str, list[dict]] = {}
    if wide is not None:
        header = records[wide]
        units = records[wide + 1] if wide + 1 < len(records) else []
        for row in records[wide + 2:]:
            if len(row) != len(header) or not row or not row[0].isdigit():
                continue
            for i, name in enumerate(header):
                if "__" not in name:
                    continue
                raw = row[i]
                try:
                    value = float(raw.replace(",", ""))
                except ValueError:
                    value = None
                metrics.setdefault(name, []).append({
                    "value": value,
                    "raw_value": raw,
                    "unit": units[i] if i < len(units) else "",
                    "kernel": row[header.index("Kernel Name")],
                    "id": row[0],
                })
        return {"format": "wide", "raw_rows": records, "metrics": metrics}
    if long is not None:
        header = records[long]
        for row in records[long + 1:]:
            if len(row) != len(header) or not row or not row[0].isdigit():
                continue
            item = dict(zip(header, row))
            name = item.get("Metric Name", "")
            raw = item.get("Metric Value", "")
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                value = None
            metrics.setdefault(name, []).append({
                "value": value,
                "raw_value": raw,
                "unit": item.get("Metric Unit", ""),
                "kernel": item.get("Kernel Name"),
                "id": item.get("ID"),
            })
        return {"format": "long", "raw_rows": records, "metrics": metrics}
    return {"format": "unrecognized", "raw_rows": records, "metrics": {}}


def _child_source() -> str:
    """Return the CUDA-runtime child used by NCU for exact launch matching."""

    return textwrap.dedent(r'''
        import base64, ctypes as C, hashlib, json, math, sys
        import torch
        from openassembler.gpu.cuda_driver import Driver
        from exp.nn_round2.operator_worker import make_inputs, reference

        spec = json.load(open(sys.argv[1]))
        torch.set_num_threads(2)
        torch.cuda.init()
        torch.empty(1, device="cuda")
        driver = Driver()
        if driver.info().cc != "8.0":
            raise RuntimeError("reduction NCU worker requires an A100/SM80")
        binary = base64.b64decode(spec["cubin"], validate=True)
        if hashlib.sha256(binary).hexdigest() != spec["cubin_sha256"]:
            raise RuntimeError("profile cubin hash mismatch")
        module = driver.load(binary)
        fn = driver.function(module, spec["kernel"])
        rows, width = spec["shape"]
        warps = spec["warps"]
        threads = warps * 32
        inputs = make_inputs(spec["op"], (rows, width), "normal", spec["seed"])
        expected = reference(spec["op"], inputs)
        stores = []
        outputs = []
        for value in expected:
            count = value.numel()
            storage = torch.full((count + 512,), -12345.0,
                                 device="cuda", dtype=torch.float32)
            stores.append(storage)
            outputs.append(storage[256:-256].view(value.shape))
        tensors = inputs + outputs + [outputs[0]] * (3 - len(outputs))
        values = [C.c_uint64(t.data_ptr()) for t in tensors] + [C.c_uint32(rows)]
        params = (C.c_void_p * len(values))(*[
            C.cast(C.pointer(value), C.c_void_p) for value in values
        ])
        grid = (rows + warps - 1) // warps
        original = [value.clone().view(torch.uint8) for value in inputs]
        rtol, atol = 2e-4, 5e-5

        def launch():
            driver.ck(driver.cu.cuLaunchKernel(
                fn, grid, 1, 1, threads, 1, 1, 0,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None),
                "reduction profile launch")

        def validate():
            torch.cuda.synchronize()
            checks = []
            for actual, truth in zip(outputs, expected):
                diff = actual.double() - truth.double()
                close = torch.isclose(actual.double(), truth.double(), rtol=rtol, atol=atol)
                checks.append({
                    "passed": bool(torch.isfinite(actual).all() and close.all()),
                    "elements": actual.numel(),
                    "max_abs": float(diff.abs().max()),
                    "relative_l2": float(torch.linalg.vector_norm(diff) /
                                         torch.linalg.vector_norm(truth.double()).clamp_min(1e-30)),
                })
            canaries = all(bool((s[:256] == -12345.).all() and
                                (s[-256:] == -12345.).all()) for s in stores)
            immutable = all(torch.equal(value.view(torch.uint8), saved)
                            for value, saved in zip(inputs, original))
            return {"passed": canaries and immutable and all(c["passed"] for c in checks),
                    "canaries": canaries, "inputs_unchanged": immutable,
                    "outputs": checks}

        launch()
        before = validate()
        if not before["passed"]:
            raise RuntimeError("preprofile reduction correctness failed: " + json.dumps(before))
        launch()
        torch.cuda.synchronize()
        for output in outputs:
            output.fill_(float("nan"))
        launch()
        after = validate()
        proof = {"passed": before["passed"] and after["passed"],
                 "before": before, "after": after,
                 "resources": driver.func_attrs(fn),
                 "device": driver.info().__dict__,
                 "matching_launches": 3,
                 "profiled_matching_launch_index": 2,
                 "cubin_sha256": spec["cubin_sha256"],
                 "seed": spec["seed"]}
        print("OA_REDUCTION_COUNTER_CHECKS:" + json.dumps(proof, allow_nan=False), flush=True)
        if not proof["passed"]:
            raise RuntimeError("profiled reduction correctness failed")
        driver.unload(module)
    ''')


def _parse_proof(stdout: str) -> dict | None:
    prefix = "OA_REDUCTION_COUNTER_CHECKS:"
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            try:
                return json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                return None
    return None


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1800, max_containers=1, retries=0)
def execute(cases: list[dict], profile_timeout_seconds: int = 180) -> dict:
    import shutil
    import subprocess
    import sys
    import tempfile

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
    from exp.nn_round2.operator_suite import rowwise_source
    from exp.weaknesses.reduction_overlap_codegen import OverlapPlan, generate

    ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    result: dict = {
        "schema": 1,
        "device": None,
        "ncu": ncu,
        "metrics_requested": METRICS,
        "cases": [],
        "methodology": {
            "replay": "kernel",
            "profiled_launch": 3,
            "pre_and_post_full_output": True,
            "canaries": True,
            "inputs_unchanged": True,
            "scope": "diagnostic counters; paired graph timings remain in reduction_overlap_run records",
        },
    }
    if ncu is None:
        result["status"] = "unavailable"
        result["reason"] = "Nsight Compute is absent from the Modal image"
        return result
    version = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    result["ncu_version"] = (version.stdout + version.stderr).strip()
    supported: list[str] = []
    query_results = {}
    for metric in METRICS:
        query = subprocess.run([ncu, "--query-metrics", "--metrics", metric],
                               capture_output=True, text=True, timeout=30)
        text = (query.stdout + "\n" + query.stderr).strip()
        accepted = query.returncode == 0 and not any(token in text.lower()
                                                     for token in ("invalid", "unknown metric", "not supported"))
        query_results[metric] = {"returncode": query.returncode, "accepted": accepted,
                                 "text": text[:1200]}
        if accepted:
            supported.append(metric)
    result["metric_queries"] = query_results
    result["metrics_supported_by_query"] = supported
    if not supported:
        result["status"] = "unavailable"
        result["reason"] = "No requested metric spelling accepted by installed Nsight Compute"
        return result

    child_path = Path(tempfile.mkdtemp(prefix="oa-reduction-ncu-child-")) / "worker.py"
    child_path.write_text(_child_source())
    try:
        for case in cases:
            op = case["op"]
            width = int(case["width"])
            warps = int(case["warps"])
            rows = int(case.get("rows", 4096))
            base_name = f"op_{op}_w{width}_c{warps}"
            baseline_kernel = base_name + "_baseline_ncu"
            overlap_kernel = base_name + "_overlap_ncu"
            _, original = rowwise_source(op, width, warps)
            baseline_body = original.replace(".entry " + base_name + "(",
                                             ".entry " + baseline_kernel + "(", 1)
            overlap_body, metadata = generate(OverlapPlan(op, width, warps))
            overlap_body = overlap_body.replace(
                ".entry " + base_name + "_overlap(",
                ".entry " + overlap_kernel + "(", 1)
            header = ".version 8.7\n.target sm_80\n.address_size 64\n"
            case_row = {"case": {"op": op, "rows": rows, "width": width, "warps": warps},
                        "metadata": metadata, "profiles": []}
            with tempfile.TemporaryDirectory(prefix="oa-reduction-build-") as td:
                root = Path(td)
                for label, kernel, body in (("baseline", baseline_kernel, baseline_body),
                                            ("overlap", overlap_kernel, overlap_body)):
                    source = root / f"{kernel}.ptx"
                    out_dir = root / label
                    source.write_text(header + body)
                    out_dir.mkdir()
                    rc = cli_main(["compile", str(source), "-o", kernel,
                                   "--out-dir", str(out_dir)])
                    if rc:
                        raise RuntimeError({"compile_rc": rc, "kernel": kernel})
                    manifest = json.loads((out_dir / "manifest.json").read_text())
                    entry = manifest["entries"][0]
                    oa = (out_dir / entry["cubin"]).read_bytes()
                    ptxas = (out_dir / entry["reference"]).read_bytes()
                    for method, binary in (("oa", oa), ("ptxas", ptxas)):
                        method_kernel = kernel if method == "oa" else kernel + "_ptxas"
                        # ptxas reference contains the source entry name; rebuild a
                        # renamed reference so NCU can filter it exactly.
                        if method == "ptxas":
                            renamed = root / f"{method_kernel}.ptx"
                            renamed.write_text(header + body.replace(
                                ".entry " + kernel + "(",
                                ".entry " + method_kernel + "(", 1))
                            ptxas_path = root / f"{method_kernel}.cubin"
                            proc = subprocess.run(["ptxas", "-arch=sm_80", "-O3",
                                                   "-o", str(ptxas_path), str(renamed)],
                                                  capture_output=True, text=True)
                            if proc.returncode:
                                raise RuntimeError({"ptxas": proc.stderr, "kernel": method_kernel})
                            binary = ptxas_path.read_bytes()
                        spec = {
                            "kernel": method_kernel,
                            "op": op,
                            "shape": [rows, width],
                            "warps": warps,
                            "seed": int(f"{rows}{width}{warps}"),
                            "cubin": base64.b64encode(binary).decode(),
                            "cubin_sha256": hashlib.sha256(binary).hexdigest(),
                        }
                        spec_path = root / f"{method_kernel}.json"
                        spec_path.write_text(json.dumps(spec))
                        command = [ncu, "--clock-control", "none", "--cache-control", "none",
                                   "--replay-mode", "kernel", "--target-processes", "all",
                                   "--kernel-name-base", "function", "--kernel-name",
                                   "regex:^" + method_kernel + "$", "--launch-skip", "2",
                                   "--launch-count", "1", "--csv", "--page", "raw",
                                   "--metrics", ",".join(supported), sys.executable,
                                   str(child_path), str(spec_path)]
                        child_env = dict(os.environ)
                        child_env["PYTHONPATH"] = "/root/repo" + (
                            ":" + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
                        )
                        prof = subprocess.run(command, capture_output=True, text=True,
                                              timeout=profile_timeout_seconds, env=child_env)
                        proof = _parse_proof(prof.stdout)
                        parsed = _parse_ncu_csv(prof.stdout)
                        collected = (prof.returncode == 0 and proof is not None and
                                     proof.get("passed") and bool(parsed["metrics"]))
                        case_row["profiles"].append({
                            "label": label,
                            "method": method,
                            "kernel": method_kernel,
                            "status": "collected" if collected else "failed",
                            "returncode": prof.returncode,
                            "proof": proof,
                            "metrics": parsed["metrics"],
                            "raw_rows": parsed["raw_rows"],
                            "stderr": prof.stderr[-8000:],
                            "stdout_excerpt": prof.stdout[:2000],
                            "command": command,
                            "static": {
                                "registers": regcount(binary, method_kernel),
                                "cubin_sha256": hashlib.sha256(binary).hexdigest(),
                            },
                        })
            result["cases"].append(case_row)
    finally:
        try:
            child_path.unlink()
            child_path.parent.rmdir()
        except OSError:
            pass
    result["status"] = ("completed" if result["cases"] and
                         all(profile["status"] == "collected"
                             for case in result["cases"] for profile in case["profiles"])
                         else "completed_with_failures")
    return result


def _cases() -> list[dict]:
    return [
        {"op": op, "width": width, "warps": warps, "rows": 4096}
        for op in ("rmsnorm_backward", "softmax_backward")
        for width in (512, 513, 1024)
        for warps in (4, 8)
    ]


@APP.local_entrypoint()
def main(subset: str = "", timeout: int = 180):
    cases = _cases()
    if subset:
        cases = [case for case in cases
                 if subset in f"{case['op']}_{case['width']}_{case['warps']}"]
    result = execute.remote(cases, timeout)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = HERE / f"reduction_overlap_ncu_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
