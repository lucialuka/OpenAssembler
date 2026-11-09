"""Role-labelled A100 study of forward transformer GEMM policies.

This measures the exact native CLI routes for QKV/output/FFN forward GEMMs.
It does not turn the static coverage manifest into an automatic dispatcher:
every role, policy and precision must pass its own gates and timing controls.
"""
from __future__ import annotations

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
         .add_local_dir(str(HERE), "/root/repo/exp/shape_coverage",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_dir(str(ROOT / "exp" / "tensor_scheduler"),
                        "/root/repo/exp/tensor_scheduler",
                        ignore=["**/__pycache__", "**/*.pyc"])
         .add_local_file(str(ROOT / "exp/a100_counters/run.py"),
                         "/root/repo/exp/a100_counters/run.py"))
APP = modal.App("openassembler-transformer-role-gemms-a100")


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=3600, max_containers=1, retries=0)
def execute(cases: list[dict], rounds: int = 2, samples: int = 8,
            pointer_sweep: bool = False, fragment_sweep: bool = False,
            profile: bool = False, tile_sweep: bool = False) -> dict:
    import ctypes as C
    import random
    import shutil
    import statistics
    import sys
    import tempfile

    import torch

    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/tools")
    import openassembler.oracle as oracle
    oracle._available.clear()
    oracle.available = lambda tool="ptxas": shutil.which(tool) is not None
    oracle.run_tool = lambda *args, check=True: subprocess.run(
        list(args), cwd="/root/repo", capture_output=True, text=True, check=check
    )

    from openassembler.cli import main as cli_main
    from openassembler.container.cubin import regcount
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.kernels.gemm import GemmPlan
    from exp.shape_coverage.gemm_policy import FRAGMENT_SPLITS, policy_source, verify_compile_record

    def compile_one(plan: GemmPlan, policy: str, label: str, root: str) -> tuple[bytes, bytes, str, dict]:
        source_plan = GemmPlan(
            plan.m, plan.n, plan.k, dtype=plan.dtype,
            cta_order=policy if policy in ("morton8", "grouped_n8") else "m_fast",
            fragment_schedule="load_next" if policy == "load_next" else "source",
            pointer_mode=plan.pointer_mode, pointer_advance=plan.pointer_advance,
        )
        os.makedirs(root, exist_ok=True)
        os_path = os.path
        ptx = os_path.join(root, f"{label}.ptx")
        out = os_path.join(root, "out")
        # The annotation and entry name are part of the checked compiler
        # contract. Renaming the entry silently discards pointer lowering.
        source, certificate = policy_source(source_plan, policy)
        with open(ptx, "w") as stream:
            stream.write(source)
        rc = cli_main(["compile", ptx, "-o", label, "--out-dir", out])
        if rc:
            raise RuntimeError({"compile_rc": rc, "role": label, "policy": policy})
        entry = json.load(open(os_path.join(out, "manifest.json")))["entries"][0]
        kernel = entry["kernels"][0]
        evidence = verify_compile_record(source_plan, policy, source, kernel, certificate)
        oa = open(os_path.join(out, entry["cubin"]), "rb").read()
        reference = open(os_path.join(out, entry["reference"]), "rb").read()
        evidence["cubin_sha256"] = {"oa": hashlib.sha256(oa).hexdigest(),
                                     "ptxas": hashlib.sha256(reference).hexdigest()}
        return oa, reference, "gemm", evidence

    def static(blob: bytes, kernel: str, root: str, *, native: bool) -> dict:
        from openassembler.isa.word import control
        path = __import__("os").path.join(root, kernel + ".cubin")
        with open(path, "wb") as stream:
            stream.write(blob)
        sass = subprocess.run(["nvdisasm", "-hex", "-c", path], capture_output=True,
                              text=True, check=True).stdout
        listing = oracle.parse_listing(sass)
        hmma_controls = sorted({control(ins.word).stall for ins in listing
                                if ins.text.startswith("HMMA")})
        drains = [control(ins.word) for index, ins in enumerate(listing)
                  if index > 0 and ins.text.startswith("NOP")
                  and listing[index - 1].text.startswith("HMMA")]
        if native and (hmma_controls != [8] or not drains
                       or any(ctl.stall != 15 or not ctl.yield_ for ctl in drains)):
            raise AssertionError({"native_control_policy_changed": hmma_controls,
                                  "drain_controls": [str(ctl) for ctl in drains]})
        lines = [line for line in sass.splitlines() if ";" in line and "/*" in line]
        return {"instructions": len(lines), "non_nop": sum(" NOP" not in x for x in lines),
                "registers": regcount(blob, kernel), "hmma": sass.count("HMMA"),
                "ldsm": sass.count("LDSM"),
                "hmma_stall_controls": hmma_controls,
                "post_hmma_nop_drains": [{"stall": ctl.stall, "yield": ctl.yield_}
                                          for ctl in drains],
                "cubin_sha256": hashlib.sha256(blob).hexdigest(),
                "sass_sha256": hashlib.sha256(sass.encode()).hexdigest()}

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    driver = Driver()
    if driver.info().cc != "8.0":
        raise RuntimeError("SM80 required")
    import glob
    paths = glob.glob("/usr/local/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so*")
    blas = C.CDLL(paths[0] if paths else "libcublas.so.12")
    create = blas.cublasCreate_v2
    create.argtypes, create.restype = [C.POINTER(C.c_void_p)], C.c_int
    destroy = blas.cublasDestroy_v2
    destroy.argtypes, destroy.restype = [C.c_void_p], C.c_int
    setstream = blas.cublasSetStream_v2
    setstream.argtypes, setstream.restype = [C.c_void_p, C.c_void_p], C.c_int
    setmath = blas.cublasSetMathMode
    setmath.argtypes, setmath.restype = [C.c_void_p, C.c_int], C.c_int
    gemmex = blas.cublasGemmEx
    gemmex.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                      C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p,
                      C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    gemmex.restype = C.c_int
    handle = C.c_void_p()
    if create(C.byref(handle)) or setmath(handle, 16):
        raise RuntimeError("cuBLAS initialization/precision policy failed")
    alpha, beta = C.c_float(1), C.c_float(0)
    rng = random.Random(20261109)
    output_rows = []
    for case in cases:
        base = GemmPlan(case["m"], case["n"], case["k"], dtype=case["dtype"])
        policies = ["m_fast", "morton8", "grouped_n8"]
        if pointer_sweep:
            policies.extend(["pointer_direct", "pointer_early", "pointer_direct_early"])
        if fragment_sweep:
            policies.extend(FRAGMENT_SPLITS)
        if tile_sweep:
            policies.append("tile64")
        try:
            GemmPlan(case["m"], case["n"], case["k"], dtype=case["dtype"],
                     fragment_schedule="load_next")
            policies.append("load_next")
        except ValueError:
            pass

        with tempfile.TemporaryDirectory(prefix="oa-role-gemm-") as temp:
            artifacts: dict[str, bytes] = {}
            resources: dict[str, dict] = {}
            modules, functions, grids, kernel_names, contracts, launches = {}, {}, {}, {}, {}, {}
            for policy in policies:
                try:
                    candidate = GemmPlan(
                        base.m, base.n, base.k, dtype=base.dtype,
                        cta_order=policy if policy in ("morton8", "grouped_n8") else "m_fast",
                        fragment_schedule="load_next" if policy == "load_next" else "source",
                        pointer_mode="direct" if policy in ("pointer_direct", "pointer_direct_early") else "legacy",
                        pointer_advance="early" if policy in ("pointer_early", "pointer_direct_early") else "late",
                    )
                except ValueError as exc:
                    continue
                label = f"{policy}_{case['role']}_{case['dtype']}"
                oa, ptxas, entry_name, contract = compile_one(candidate, policy, label, str(Path(temp) / label))
                contracts[policy] = contract
                launch_config = contract.get("manual_launch") or candidate.launch
                for method, blob in (("oa", oa), ("ptxas", ptxas)):
                    method_label = f"{method}_{policy}"
                    artifacts[method_label] = blob
                    module = driver.load(blob)
                    function = driver.function(module, entry_name)
                    driver.ck(driver.cu.cuFuncSetAttribute(function, 8, launch_config["smem"]),
                              "shared-memory opt-in")
                    modules[method_label], functions[method_label] = module, function
                    kernel_names[method_label] = entry_name
                    resources[method_label] = driver.func_attrs(function)
                    grid = launch_config["grid"]
                    grids[method_label] = grid[:2]
                    launches[method_label] = launch_config

            m, n, k = base.m, base.n, base.k
            dtype = torch.float16 if base.dtype == "f16" else torch.bfloat16
            methods = list(artifacts) + ["cublas"]
            def launch(method_label: str, out: torch.Tensor) -> None:
                if method_label == "cublas":
                    if setstream(handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)):
                        raise RuntimeError("cuBLAS stream binding failed")
                    operand_type = 2 if base.dtype == "f16" else 14
                    rc = gemmex(handle, 1, 0, n, m, k, C.byref(alpha),
                                C.c_void_p(b.data_ptr()), operand_type, k,
                                C.c_void_p(a.data_ptr()), operand_type, k,
                                C.byref(beta), C.c_void_p(out.data_ptr()), 0, n, 68, 99)
                    if rc:
                        raise RuntimeError(f"cuBLAS GemmEx failed: {rc}")
                    return
                args = [C.c_uint64(t.data_ptr()) for t in (a, b, out)]
                params = (C.c_void_p * 3)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
                gx, gy = grids[method_label]
                config = launches[method_label]
                driver.ck(driver.cu.cuLaunchKernel(
                    functions[method_label], gx, gy, 1, config["threads"], 1, 1, config["smem"],
                    C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None),
                    method_label)

            checks = []
            ptxas_by_policy = {}
            input_families = {}
            for family in ("zero", "integer", "normal"):
                torch.manual_seed(7100 + m + n + k)
                if family == "zero":
                    a = torch.zeros((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((n, k), device="cuda", dtype=dtype)
                elif family == "integer":
                    a = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
                    b = torch.randint(-2, 3, (n, k), device="cuda").to(dtype)
                else:
                    a = torch.randn((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((n, k), device="cuda", dtype=dtype)
                a_before, b_before = a.clone(), b.clone()
                truth = a.float() @ b.float().T
                refs = {}
                for name in artifacts:
                    if name.startswith("ptxas_"):
                        policy = name.removeprefix("ptxas_")
                        reference = torch.empty((m, n), device="cuda", dtype=torch.float32)
                        launch(name, reference)
                        torch.cuda.synchronize()
                        refs[policy] = reference.clone()
                for name in methods:
                    policy = name.split("_", 1)[1] if name != "cublas" else "m_fast"
                    storage = torch.full((m * n + 512,), -9876.5, device="cuda")
                    output = storage[256:-256].view(m, n)
                    output.fill_(float("nan"))
                    launch(name, output)
                    torch.cuda.synchronize()
                    reference = refs[policy]
                    rel = float((output - truth).norm() / truth.norm().clamp_min(1e-30))
                    canaries = bool((storage[:256] == -9876.5).all()
                                    and (storage[-256:] == -9876.5).all())
                    bitwise = (bool(torch.equal(output.view(torch.int32), reference.view(torch.int32)))
                               if name != "cublas" else None)
                    baseline_bitwise = (bool(torch.equal(output.view(torch.int32),
                                                          refs["m_fast"].view(torch.int32)))
                                        if name != "cublas" else None)
                    input_unchanged = torch.equal(a, a_before) and torch.equal(b, b_before)
                    passed = (canaries and (bitwise is not False) and (baseline_bitwise is not False) and input_unchanged
                              and bool(torch.isfinite(output).all()) and rel < 5e-5)
                    check = {"input": family, "method": name, "relative_l2_fp32": rel,
                             "canaries": canaries, "inputs_unchanged": input_unchanged,
                             "ptxas_bitwise": bitwise,
                             "baseline_cross_policy_bitwise": baseline_bitwise, "passed": passed}
                    checks.append(check)
                    if not passed:
                        raise AssertionError({"case": case, "check": check})
                ptxas_by_policy = refs
                input_families[family] = (a, b, truth)

            a, b, truth = input_families["normal"]

            graph_outputs = {name: torch.empty((m, n), device="cuda") for name in methods}
            graphs = {}
            graph_checks = []
            for name in methods:
                for _ in range(2):
                    launch(name, graph_outputs[name])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(4):
                        launch(name, graph_outputs[name])
                graph_outputs[name].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                policy = name.split("_", 1)[1] if name != "cublas" else "m_fast"
                rel = float((graph_outputs[name] - truth).norm() / truth.norm().clamp_min(1e-30))
                graph_passed = bool(torch.isfinite(graph_outputs[name]).all()) and rel < 5e-5
                if name != "cublas":
                    graph_passed &= torch.equal(graph_outputs[name].view(torch.int32),
                                                ptxas_by_policy[policy].view(torch.int32))
                    graph_passed &= torch.equal(graph_outputs[name].view(torch.int32),
                                                ptxas_by_policy["m_fast"].view(torch.int32))
                graph_checks.append({"method": name, "poisoned_output": True,
                                     "relative_l2_fp32": rel, "passed": graph_passed})
                if not graph_passed:
                    raise AssertionError({"case": case, "graph": name, "passed": False})
                graphs[name] = graph

            timing = []
            for round_id in range(rounds):
                elapsed = {name: [] for name in methods}
                orders = []
                for _ in range(samples):
                    order = list(methods)
                    rng.shuffle(order)
                    orders.append(order)
                    for name in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(); graphs[name].replay(); end.record(); end.synchronize()
                        elapsed[name].append(start.elapsed_time(end) / 4)
                timing.append({"round": round_id,
                               "paired_orders": orders,
                               "median_ms": {name: statistics.median(vals)
                                             for name, vals in elapsed.items()},
                               "samples_ms": elapsed})

            counters = None
            if profile:
                from exp.shape_coverage.profile_worker import profile_artifacts
                wanted = {"oa_m_fast", "ptxas_m_fast", "oa_pointer_direct_early",
                          "oa_load_next", "oa_fragment_split6"}
                selected = {name: blob for name, blob in artifacts.items() if name in wanted}
                counters = profile_artifacts(selected, "gemm", grids, (m, n, k),
                                             base.dtype, Path(temp) / "ncu")
            output_rows.append({"case": case, "policies": policies, "checks": checks,
                                "verified_compile_contracts": contracts,
                                "poisoned_graph_checks": graph_checks,
                                "resources": resources,
                                "launches": launches,
                                "counters": counters,
                                "static": {name: static(blob, kernel_names[name], temp,
                                                        native=name.startswith("oa_"))
                                           for name, blob in artifacts.items()},
                                "timings": timing})
            for module in modules.values():
                driver.unload(module)
            torch.cuda.synchronize()
    destroy(handle)
    driver.close()
    return {"schema": 3, "device": torch.cuda.get_device_name(), "cases": output_rows,
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "ptxas": subprocess.run(["ptxas", "--version"], capture_output=True,
                                                text=True, check=True).stdout.strip(),
                        "nvdisasm": subprocess.run(["nvdisasm", "--version"], capture_output=True,
                                                    text=True, check=True).stdout.strip()},
            "methodology": {"full_output": True, "canaries": True,
                            "input_immutability": True, "poisoned_graph": True,
                            "ptxas_bitwise": "OA versus same-policy ptxas; cuBLAS uses FP32 relative-L2 gate",
                            "baseline_cross_policy_bitwise": True,
                            "fragment_sweep": fragment_sweep,
                            "tile_sweep": tile_sweep,
                            "cublas": "GemmEx, FP16/BF16 inputs, FP32 output/compute, alpha1 beta0, disallow reduced precision reduction",
                            "counter_replay_after_timing": profile,
                            "rounds": rounds, "samples": samples}}


@APP.local_entrypoint()
def main(rounds: int = 2, samples: int = 8, heldout: bool = False,
         role: str = "all", dtype: str = "all", model_case: str = "all",
         pointer_sweep: bool = False, fragment_sweep: bool = False,
         profile: bool = False, tile_sweep: bool = False):
    from exp.shape_coverage.role_coverage import forward_gemm_cases
    cases = forward_gemm_cases(heldout=heldout)
    cases = [case for case in cases
             if (role == "all" or case["role"] == role)
             and (dtype == "all" or case["dtype"] == dtype)
             and (model_case == "all" or case["model_case"] == model_case)]
    if not cases:
        raise SystemExit("no cases selected")
    source_paths = [path for path in (ROOT / "openassembler").rglob("*")
                    if path.is_file() and path.suffix != ".pyc"]
    source_paths += sorted((ROOT / "specs" / "sm_80").glob("*.json"))
    source_paths += [ROOT / "corpus" / "sm_80" / name
                     for name in ("pairs.jsonl", "machine_model.json")]
    source_paths += [Path(__file__), HERE / "role_coverage.py", HERE / "gemm_policy.py",
                     HERE / "profile_worker.py", ROOT / "exp/a100_counters/run.py",
                     HERE / "small_tile_gemm.py",
                     ROOT / "exp/tensor_scheduler/fragment_split_search.py",
                     ROOT / "exp/tensor_scheduler/load_next_compute_current.py"]
    identity = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip(),
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in source_paths},
    }
    identity["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["experiment_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (HERE / "gemm_policy.py", ROOT / "exp/tensor_scheduler/fragment_split_search.py",
                     ROOT / "exp/tensor_scheduler/load_next_compute_current.py")}
    result = execute.remote(cases, rounds, samples, pointer_sweep, fragment_sweep, profile, tile_sweep)
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source_paths}
    result["source_identity"] = identity
    result["source_unchanged_during_run"] = after == identity["sha256"]
    if not result["source_unchanged_during_run"]:
        result["promotion_blocker"] = "local compiler sources changed during the run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = HERE / f"a100-role-gemm_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
