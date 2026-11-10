"""Checkpointed A100 study of 1/2/4/8-warps rowwise CTA geometry.

Run synchronously; every finished shape is saved locally and to a dedicated
Modal volume before the next shape starts. Only independent-row grouping
changes. This runner never edits production policies or weakens numerical gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IMAGE = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
         .add_local_dir(str(ROOT / "openassembler"), "/root/repo/openassembler",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "specs"), "/root/repo/specs")
         .add_local_dir(str(ROOT / "corpus"), "/root/repo/corpus",
                        ignore=["**/.tmp/**", "**/*.cubin", "**/*.sass", "**/*.pyc"])
         .add_local_dir(str(ROOT / "env"), "/root/repo/env",
                        ignore=["cuda-*", "**/__pycache__"])
         .add_local_dir(str(ROOT / "tools"), "/root/repo/tools",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_file(str(HERE / "rowwise_geometry.py"),
                         "/root/repo/exp/dispatch_tuning/rowwise_geometry.py")
         .add_local_file(str(HERE / "rowwise_modal.py"),
                         "/root/repo/exp/dispatch_tuning/rowwise_modal.py"))
APP = modal.App("openassembler-rowwise-geometry-revalidation")
CHECKPOINTS = modal.Volume.from_name("openassembler-rowwise-geometry-evidence", create_if_missing=True)


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=300, max_containers=1,
              retries=0, volumes={"/evidence": CHECKPOINTS})
def execute(op: str, rows: int, width: int, session: str, rounds: int, samples: int) -> dict:
    import base64
    import ctypes as C
    from dataclasses import asdict
    import io
    import os
    import random
    import statistics
    import subprocess
    import sys
    import tempfile
    import time
    import zipfile

    sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    import torch
    from openassembler.cli import main as cli_main
    from openassembler.gemm_dispatch import compiler_identity
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.kernels.reductions import ReductionPlan
    from exp.dispatch_tuning.rowwise_geometry import WARPS, TOLERANCES, source
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.init()
    _anchor = torch.empty(1, device="cuda")
    driver = Driver()
    device = asdict(driver.info())
    if device["cc"] != "8.0" or "A100" not in device["name"]:
        raise ValueError("this campaign requires an A100 SM80 GPU")
    if rounds < 3 or samples < 8 or rounds > 5 or samples > 24:
        raise ValueError("bounded study requires 3..5 rounds and 8..24 samples")
    started = time.monotonic()
    identity = compiler_identity()
    plan = ReductionPlan(rows, width, op)
    key = f"{op}_{rows}x{width}"
    functions, modules, candidates, artifact_files = {}, {}, {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix="oa-rowwise-geometry-") as temporary:
            for warps in WARPS:
                text, certificate = source(plan, warps)
                directory = Path(temporary) / f"w{warps}"
                directory.mkdir()
                src = directory / f"{op}.ptx"
                src.write_text(text)
                out = directory / "out"
                if cli_main(["compile", str(src), "--opt-level", "opt", "--out-dir", str(out)]):
                    raise RuntimeError(f"native compilation failed for {key}/w{warps}")
                manifest = json.loads((out / "manifest.json").read_text())["entries"][0]
                kernel = manifest["kernels"][0]
                if kernel.get("compiled_by") != "openassembler":
                    raise RuntimeError("compiler fallback is forbidden")
                if kernel.get("rowwise") is not None:
                    raise RuntimeError("experimental geometry must not infer a production launch contract")
                info = {"certificate": certificate, "checks": [], "graph_checks": [],
                        "artifacts": {}, "disassembly_valid": True, "resources_valid": True}
                for method, field in (("oa", "cubin"), ("ptxas", "reference")):
                    label = f"{method}_w{warps}"
                    blob = (out / manifest[field]).read_bytes()
                    cubin = directory / f"{label}.cubin"
                    cubin.write_bytes(blob)
                    sass = subprocess.run(["nvdisasm", "-hex", "-c", str(cubin)],
                                          check=True, capture_output=True, text=True).stdout
                    module = driver.load(blob)
                    modules[label] = module
                    function = driver.function(module, op)
                    attrs = driver.func_attrs(function)
                    if (attrs.get("local_bytes") != 0 or not 0 < attrs.get("num_regs", 0) <= 255
                            or attrs.get("max_threads", 0) < warps * 32):
                        raise RuntimeError(f"resource gate failed: {label}/{attrs}")
                    functions[label] = (function, certificate["launch"])
                    info["artifacts"][method] = {
                        "cubin_sha256": hashlib.sha256(blob).hexdigest(),
                        "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
                        "resources": attrs, "instruction_lines": sum(";" in line and "/*" in line for line in sass.splitlines()),
                        "warpsync_count": sass.count("WARPSYNC")}
                    artifact_files[f"w{warps}/{method}.cubin"] = blob
                    artifact_files[f"w{warps}/{method}.sass"] = sass.encode()
                artifact_files[f"w{warps}/{op}.ptx"] = text.encode()
                candidates[f"w{warps}"] = info

            xstorage = torch.full((rows * width + 512,), -12345., device="cuda", dtype=torch.float32)
            x = xstorage[256:-256].view(rows, width)
            stores = {label: torch.full_like(xstorage, -12345.) for label in functions}
            outputs = {label: store[256:-256].view(rows, width) for label, store in stores.items()}

            def launch(label):
                function, config = functions[label]
                args = [C.c_uint64(x.data_ptr()), C.c_uint64(outputs[label].data_ptr())]
                ptrs = (C.c_void_p * 2)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                driver.ck(driver.cu.cuLaunchKernel(function, *config["grid"], config["threads"], 1, 1,
                          0, C.c_void_p(torch.cuda.current_stream().cuda_stream), ptrs, None), label)

            def reference():
                xd = x.double()
                if op == "softmax":
                    return torch.softmax(xd, dim=-1)
                mu = xd.mean(dim=-1, keepdim=True)
                return (xd - mu) * torch.rsqrt((xd - mu).square().mean(dim=-1, keepdim=True) + 1e-5)

            def checks(label, expected):
                y = outputs[label]
                close = torch.isclose(y.double(), expected, **TOLERANCES[op])
                return {"method": label, "full_output": bool(close.all()),
                        "finite_output": bool(torch.isfinite(y).all()),
                        "elements_checked": y.numel(), "mismatched_elements": int((~close).sum()),
                        "max_absolute_error": float((y.double() - expected).abs().max()),
                        "canaries": bool((stores[label][:256] == -12345.).all()
                                         and (stores[label][-256:] == -12345.).all()),
                        "inputs_unchanged": bool(torch.equal(xstorage.view(torch.int32), before.view(torch.int32))),
                        "baseline_oa_bitwise": bool(torch.equal(y.view(torch.int32), outputs["oa_w4"].view(torch.int32))),
                        "same_policy_ptxas_bitwise": bool(torch.equal(
                            y.view(torch.int32), outputs[label.replace("oa_", "ptxas_")].view(torch.int32)))}

            torch.manual_seed(19031 + rows + width)
            normal = torch.randn((rows, width), device="cuda")
            patterns = {"normal": normal, "zero": torch.zeros_like(normal),
                        "wide": normal * 30., "large_offset_small_variance": 1000. + normal * .01,
                        "alternating": torch.where(torch.arange(width, device="cuda") % 2 == 0,
                                                   10000., -10000.).expand(rows, width).clone()}
            for family, values in patterns.items():
                x.copy_(values)
                before = xstorage.clone()
                expected = reference()
                for label in functions:
                    outputs[label].fill_(float("nan"))
                    launch(label)
                torch.cuda.synchronize()
                for label in functions:
                    check = {"input": family, **checks(label, expected)}
                    passed = all(check[gate] for gate in
                                 ("full_output", "finite_output", "canaries", "inputs_unchanged"))
                    if label.startswith("oa_"):
                        passed &= check["baseline_oa_bitwise"]
                    check["passed"] = bool(passed)
                    candidates[label.split("_", 1)[1]]["checks"].append(check)
            x.copy_(normal)
            before = xstorage.clone()
            expected = reference()
            graphs = {}
            for label in functions:
                for _ in range(3):
                    launch(label)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        launch(label)
                outputs[label].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                graphs[label] = graph
            for label in functions:
                check = {"poisoned_output": True, **checks(label, expected)}
                check["passed"] = all(check[gate] for gate in
                                      ("full_output", "finite_output", "canaries", "inputs_unchanged"))
                if label.startswith("oa_"):
                    check["passed"] &= check["baseline_oa_bitwise"]
                candidates[label.split("_", 1)[1]]["graph_checks"].append(check)
            eligible = [f"{method}_{name}" for name, info in candidates.items()
                        if all(check["passed"] for check in info["checks"] + info["graph_checks"])
                        for method in ("oa", "ptxas")]
            timings = []
            rng = random.Random(20261110 + rows + width)
            for round_id in range(rounds):
                elapsed = {label: [] for label in eligible}
                orders = []
                for _ in range(samples):
                    order = list(eligible)
                    rng.shuffle(order)
                    orders.append(order)
                    for label in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[label].replay(); end.record(); end.synchronize()
                        elapsed[label].append(start.elapsed_time(end) / 32)
                timings.append({"round": round_id, "paired_orders": orders,
                                "samples_ms": elapsed,
                                "median_ms": {label: statistics.median(values) for label, values in elapsed.items()}})
            result = {"schema": 1, "session": session, "case": {"op": op, "rows": rows, "width": width,
                       "epsilon": 1e-5, "dtype": "f32", "layout": "contiguous_row_major"},
                      "compiler_identity": identity, "source_unchanged_during_run": identity == compiler_identity(),
                      "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "device": device, "candidates": candidates, "timings": timings,
                      "tolerances": TOLERANCES[op], "status": "ok" if len(eligible) == 8 else "candidate-rejections",
                      "reference": "unchanged FP64 formula on exact FP32 input; production epsilon and gates",
                      "timing_scope": "resident-input 32-operation CUDA graphs; excludes compilation/transfers/training",
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "worker_seconds": time.monotonic() - started}
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in artifact_files.items():
                    archive.writestr(name, data)
            payload = buffer.getvalue()
            result["artifact_archive"] = {"path": f"{key}.artifacts.zip", "sha256": hashlib.sha256(payload).hexdigest()}
            directory = Path("/evidence") / session
            directory.mkdir(parents=True, exist_ok=True)
            (directory / result["artifact_archive"]["path"]).write_bytes(payload)
            (directory / f"{key}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            CHECKPOINTS.commit()
            print(json.dumps({"case": key, "status": result["status"],
                              "median_ms": [row["median_ms"] for row in timings]}), flush=True)
            return {"result": result, "archive": base64.b64encode(payload).decode()}
    finally:
        torch.cuda.synchronize()
        for module in modules.values():
            driver.unload(module)


@APP.local_entrypoint()
def main(pilot: bool = False, rounds: int = 3, samples: int = 8, session: str = ""):
    import base64
    from exp.dispatch_tuning.rowwise_geometry import SHAPES
    session = session or "rowwise_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(session).name != session or session.startswith("."):
        raise ValueError("session must be a plain directory name")
    directory = HERE / "rowwise_runs" / session
    directory.mkdir(parents=True, exist_ok=True)
    cases = [(op, rows, width) for op in ("softmax", "layernorm")
             for rows, width in ([(7, 33), (1024, 512)] if pilot else SHAPES)]
    metadata = {"schema": 1, "session": session, "case_count": len(cases), "rounds": rounds, "samples": samples,
                "checkpoint_volume": "openassembler-rowwise-geometry-evidence", "statuses": {}}
    (directory / "campaign.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for op, rows, width in cases:
        key = f"{op}_{rows}x{width}"
        output = directory / f"{key}.json"
        if output.exists():
            continue
        response = execute.remote(op, rows, width, session, rounds, samples)
        result = response["result"]
        archive = base64.b64decode(response["archive"], validate=True)
        if hashlib.sha256(archive).hexdigest() != result["artifact_archive"]["sha256"]:
            raise ValueError("returned artifact archive hash mismatch")
        (directory / result["artifact_archive"]["path"]).write_bytes(archive)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        metadata["statuses"][key] = result["status"]
        (directory / "campaign.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(output)
