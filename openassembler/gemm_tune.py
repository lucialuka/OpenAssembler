"""Optional, correctness-gated native A100 GEMM tuning.

No GPU, Torch, network, or tuning work is performed on import. The normal
compiler stays stdlib-only. Explicit GPU tuning requires Torch and the CUDA
toolkit locally, or the separately installed Modal client.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from openassembler.gemm_dispatch import candidate_plans, compiler_identity, contract_key
from openassembler.kernels.gemm import GemmPlan, generate


def choose_policy(result: dict, *, minimum_speedup: float = 1.02) -> str:
    """Recompute the winner from complete checks and paired raw timing samples."""
    if type(minimum_speedup) not in (int, float) or not math.isfinite(minimum_speedup) or minimum_speedup <= 1:
        raise ValueError("a real positive speedup threshold is required")
    candidates = result.get("candidates")
    timings = result.get("timings")
    if not isinstance(candidates, dict) or "baseline" not in candidates:
        raise ValueError("tuning result lacks its baseline")
    if not isinstance(timings, list) or len(timings) < 3:
        raise ValueError("tuning requires at least three paired rounds")
    names = set(candidates)
    if len(names) < 2:
        raise ValueError("tuning requires at least two admitted candidate policies")
    medians = []
    round_ids = set()
    for timing in timings:
        if not isinstance(timing, dict):
            raise ValueError("invalid paired tuning round")
        round_id = timing.get("round")
        if type(round_id) is not int or round_id < 0 or round_id in round_ids:
            raise ValueError("paired tuning rounds need distinct nonnegative integer identities")
        round_ids.add(round_id)
        samples = timing.get("samples_ms", {})
        orders = timing.get("paired_orders", [])
        if not isinstance(samples, dict) or not isinstance(orders, list) or set(samples) != names or len(orders) < 8:
            raise ValueError("incomplete paired tuning samples")
        if any(not isinstance(order, list) or len(order) != len(names)
               or set(order) != names for order in orders):
            raise ValueError("paired order does not cover every candidate exactly once")
        if any(not isinstance(vals, list) or len(vals) != len(orders)
               or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in vals)
               for vals in samples.values()):
            raise ValueError("invalid or missing tuning timing")
        medians.append({name: statistics.median(vals) for name, vals in samples.items()})
    eligible = []
    for name, candidate in candidates.items():
        if not isinstance(candidate, dict):
            raise ValueError(f"invalid candidate: {name}")
        checks = candidate.get("checks", [])
        if (not isinstance(checks, list) or len(checks) != 3
                or {check.get("input") for check in checks if isinstance(check, dict)}
                != {"zero", "integer", "normal"}):
            raise ValueError(f"incomplete numerical coverage: {name}")
        if any(any(check.get(gate) is not True for gate in
                   ("full_output", "canaries", "inputs_unchanged", "finite_output", "ptxas_bitwise",
                    "baseline_bitwise")) for check in checks):
            raise ValueError(f"failed numerical gate: {name}")
        for check in checks:
            error = check.get("relative_l2_fp32")
            if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error < 5e-5:
                raise ValueError(f"nonfinite or out-of-budget numerical error: {name}")
        if (candidate.get("poisoned_graph") is not True
                or candidate.get("disassembly_valid") is not True
                or candidate.get("resources_valid") is not True):
            raise ValueError(f"failed graph/disassembly/resource gate: {name}")
        if name == "baseline" or all(row["baseline"] / row[name] >= minimum_speedup for row in medians):
            eligible.append(name)
    # Uncertain gains keep the reference. Rank only candidates with a repeatable
    # benefit; a lower aggregate median cannot conceal a regressing round.
    return min(eligible, key=lambda name: statistics.geometric_mean(row[name] for row in medians))


def validate_record(plan: GemmPlan, result: dict) -> tuple[GemmPlan, dict]:
    if not isinstance(result, dict) or result.get("schema") != 1:
        raise ValueError("unsupported GPU tuning record")
    if result.get("compiler_identity") != compiler_identity():
        raise ValueError("GPU tuning record belongs to a different compiler; retune")
    if result.get("contract") != contract_key(plan):
        raise ValueError("GPU tuning record has a different shape/precision/layout contract")
    device = result.get("device", {})
    if (not isinstance(device, dict) or device.get("cc") != "8.0"
            or not isinstance(device.get("name"), str) or "A100" not in device["name"]):
        raise ValueError("GPU tuning record is not an A100 SM80 measurement")
    expected = candidate_plans(plan)
    recorded_candidates = result.get("candidates")
    if not isinstance(recorded_candidates, dict) or set(recorded_candidates) != set(expected):
        raise ValueError("GPU tuning record does not cover the current admitted candidate set")
    for name, candidate in expected.items():
        recorded = recorded_candidates[name]
        if not isinstance(recorded, dict):
            raise ValueError(f"invalid recorded candidate: {name}")
        if (recorded.get("plan") != asdict(candidate)
                or recorded.get("ptx_body_sha256") != hashlib.sha256(generate(candidate).encode()).hexdigest()):
            raise ValueError(f"GPU tuning policy source changed: {name}")
        artifacts = recorded.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"GPU tuning resource reports are missing: {name}")
        for kind in ("native", "ptxas"):
            artifact = artifacts.get(kind)
            resources = artifact.get("resources") if isinstance(artifact, dict) else None
            if (not isinstance(resources, dict)
                    or any(type(resources.get(key)) is not int for key in
                           ("num_regs", "local_bytes", "max_threads", "shared_bytes", "binary_version"))
                    or not 0 < resources["num_regs"] <= 255 or resources["local_bytes"] != 0
                    or resources["max_threads"] < candidate.launch["threads"]
                    or resources["shared_bytes"] < 0
                    or resources["shared_bytes"] + candidate.launch["smem"] > 163840
                    or resources["binary_version"] != 80):
                raise ValueError(f"GPU tuning resource report is invalid: {name}/{kind}")
    selected = choose_policy(result)
    return expected[selected], {"schema": 1, "mode": "gpu-tuned", "policy": selected,
                                "reason": "correctness-gated paired A100 timing",
                                "device": device, "contract": contract_key(plan),
                                "scope": "this shape, native candidate set, worker and timing scope; no global guarantee"}


def tune_or_load(plan: GemmPlan, *, backend: str | None, record: str | None,
                 rounds: int = 3, samples: int = 12) -> tuple[GemmPlan, dict]:
    if plan.epilogue != "none":
        raise ValueError("GPU tuning currently covers unfused RCR FP16/BF16 GEMM only")
    if type(rounds) is not int or type(samples) is not int or rounds < 3 or samples < 8:
        raise ValueError("GPU tuning requires at least three rounds and eight samples")
    if rounds > 10 or samples > 100:
        raise ValueError("GPU tuning is bounded to ten rounds and one hundred samples")
    if backend is None:
        if not record:
            raise ValueError("a tuning record is required")
        result = json.loads(Path(record).read_text())
        _verify_archive(Path(record), result)
    elif backend == "local":
        result = benchmark(plan, rounds=rounds, samples=samples)
    elif backend == "modal":
        import subprocess
        import tempfile
        from openassembler.runner import modal_ready
        if why := modal_ready():
            raise ValueError(why)
        with tempfile.TemporaryDirectory(prefix="oa-gemm-tune-") as directory:
            request = Path(directory) / "request.json"
            response = Path(directory) / "result.json"
            request.write_text(json.dumps({"plan": asdict(plan), "rounds": rounds, "samples": samples}))
            command = ["modal", "run", str(Path(__file__).with_name("gemm_tune_modal.py")),
                       "--request", str(request), "--result", str(response)]
            try:
                subprocess.run(command, check=True, timeout=2100)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise ValueError(f"Modal GEMM tuning failed: {exc}") from None
            result = json.loads(response.read_text())
    else:
        raise ValueError("GEMM tuning backend must be local or modal")
    selected, decision = validate_record(plan, result)
    if backend and record:
        import base64
        import io
        import zipfile
        from openassembler.project import write_atomic
        destination = Path(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifacts = result.pop("artifact_bytes", {})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in artifacts.items():
                # Names are generated by this module; don't allow an unexpected
                # worker path to escape the evidence archive on extraction.
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("invalid tuning artifact path")
                archive.writestr(name, base64.b64decode(data, validate=True))
        payload = buffer.getvalue()
        archive_path = destination.with_suffix(".artifacts.zip")
        write_atomic(archive_path, payload)
        result["artifact_archive"] = {"path": archive_path.name,
                                      "sha256": hashlib.sha256(payload).hexdigest()}
        write_atomic(destination, json.dumps(result, indent=2, allow_nan=False) + "\n")
    return selected, {**decision, "record": str(record) if record else None}


def _verify_archive(record: Path, result: dict) -> None:
    """Check retained source/binary evidence before explicitly reusing a record."""
    import zipfile
    archive = result.get("artifact_archive", {})
    name = archive.get("path") if isinstance(archive, dict) else None
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("tuning record has no local evidence archive")
    path = record.parent / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != archive.get("sha256"):
        raise ValueError("tuning evidence archive is missing or changed")
    try:
        with zipfile.ZipFile(path) as retained:
            if len(retained.namelist()) != len(set(retained.namelist())):
                raise ValueError("duplicate artifact paths")
            for policy, candidate in result.get("candidates", {}).items():
                for kind in ("native", "ptxas"):
                    artifact = candidate.get("artifacts", {}).get(kind, {})
                    for extension, key in (("cubin", "cubin_sha256"), ("sass", "sass_sha256")):
                        data = retained.read(f"{policy}/{kind}.{extension}")
                        if hashlib.sha256(data).hexdigest() != artifact.get(key):
                            raise ValueError("retained artifact hash changed")
                from openassembler.gemm import source_for
                source = retained.read(f"{policy}/gemm.ptx").decode()
                if source != source_for(GemmPlan(**candidate["plan"])):
                    raise ValueError("retained source differs from the checked current plan")
    except (KeyError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid tuning evidence archive: {exc}") from None


def benchmark(plan: GemmPlan, *, rounds: int = 3, samples: int = 12) -> dict:
    """Compile via the real CLI and validate every native policy before timing.

    Captured sixteen-launch graphs amortize host dispatch. Complete floating-point
    outputs, poisoned output, guards, input immutability and same-PTX ptxas bits
    are checked. This is kernel tuning; it excludes packing and training state.
    """
    import base64
    import ctypes as C
    import random
    import shutil
    import subprocess
    import tempfile
    import time
    from datetime import datetime, timezone

    try:
        import torch
    except ImportError:
        raise ValueError("local GPU tuning requires the optional openassembler[torch] dependency") from None
    from openassembler.cli import main as cli_main
    from openassembler.gemm import source_for
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.isa.word import control
    from openassembler import oracle
    if not shutil.which("ptxas") or not shutil.which("nvdisasm"):
        raise ValueError("GPU tuning requires ptxas and nvdisasm on PATH")
    started = time.monotonic()
    identity = compiler_identity()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.init()
    _context = torch.empty(1, device="cuda")
    driver = Driver()
    device = asdict(driver.info())
    if device["cc"] != "8.0" or "A100" not in device["name"]:
        raise ValueError("native GEMM tuning requires an A100 SM80 GPU")
    plans = candidate_plans(plan)
    modules, methods, candidates, artifacts = {}, {}, {}, {}
    graphs, graph_outputs = {}, {}
    m, n, k = plan.m, plan.n, plan.k
    dtype = torch.float16 if plan.dtype == "f16" else torch.bfloat16
    rng = random.Random(20261109)
    # Explicit argument/shape validation also protects direct benchmark callers.
    if plan.epilogue != "none" or rounds < 3 or samples < 8 or rounds > 10 or samples > 100:
        raise ValueError("bounded tuning requires unfused GEMM, 3..10 rounds and 8..100 samples")
    try:
        with tempfile.TemporaryDirectory(prefix="oa-native-gemm-tune-") as temp:
            for name, candidate in plans.items():
                directory = Path(temp) / name
                directory.mkdir()
                source = directory / "gemm.ptx"
                source.write_text(source_for(candidate))
                out = directory / "out"
                if cli_main(["compile", str(source), "--out-dir", str(out)]):
                    raise ValueError(f"native candidate failed compilation: {name}")
                manifest = json.loads((out / "manifest.json").read_text())["entries"][0]
                kernel = manifest["kernels"][0]
                if kernel.get("compiled_by") != "openassembler" or kernel.get("gemm", {}).get("launch") != candidate.launch:
                    raise ValueError(f"native candidate lost its checked compiler contract: {name}")
                record = {"plan": asdict(candidate),
                          "ptx_body_sha256": hashlib.sha256(generate(candidate).encode()).hexdigest(),
                          "checks": [], "artifacts": {}}
                for kind, path in (("native", manifest["cubin"]), ("ptxas", manifest["reference"])):
                    blob = (out / path).read_bytes()
                    binary = directory / f"{kind}.cubin"
                    binary.write_bytes(blob)
                    sass = subprocess.run(["nvdisasm", "-hex", "-c", str(binary)], check=True,
                                          capture_output=True, text=True).stdout
                    listing = oracle.parse_listing(sass)
                    if not listing or not any(ins.text.startswith("HMMA") for ins in listing):
                        raise ValueError(f"invalid tensor disassembly: {name}/{kind}")
                    if kind == "native":
                        if any(control(ins.word).stall != 8 for ins in listing if ins.text.startswith("HMMA")):
                            raise ValueError("native HMMA control policy changed")
                        drains = [ins for index, ins in enumerate(listing) if index and ins.text.startswith("NOP")
                                  and listing[index - 1].text.startswith("HMMA")]
                        if not drains or any(control(ins.word).stall != 15 for ins in drains):
                            raise ValueError("native observer drain changed")
                    module = driver.load(blob)
                    modules[(name, kind)] = module
                    function = driver.function(module, "gemm")
                    attrs = driver.func_attrs(function)
                    if (attrs.get("local_bytes") != 0 or not 0 < attrs.get("num_regs", 0) <= 255
                            or attrs.get("max_threads", 0) < 128):
                        raise ValueError(f"invalid resources: {name}/{kind}/{attrs}")
                    if candidate.launch["smem"] > 49152:
                        driver.ck(driver.cu.cuFuncSetAttribute(function, 8, candidate.launch["smem"]), "shared opt-in")
                    methods[(name, kind)] = (function, candidate.launch)
                    record["artifacts"][kind] = {"cubin_sha256": hashlib.sha256(blob).hexdigest(),
                                                 "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
                                                 "resources": attrs, "instructions": len(listing),
                                                 "counts": {op: sum(ins.text.split()[0].split(".")[0] == op
                                                                    for ins in listing)
                                                            for op in ("HMMA", "LDSM", "MOV", "IADD3", "IMAD", "NOP")}}
                    artifacts[f"{name}/{kind}.cubin"] = base64.b64encode(blob).decode()
                    artifacts[f"{name}/{kind}.sass"] = base64.b64encode(sass.encode()).decode()
                artifacts[f"{name}/gemm.ptx"] = base64.b64encode(source.read_bytes()).decode()
                record["disassembly_valid"] = record["resources_valid"] = True
                candidates[name] = record

            def launch(name, kind, out):
                fn, config = methods[(name, kind)]
                args = [C.c_uint64(x.data_ptr()) for x in (a, b, out)]
                pointers = (C.c_void_p * 3)(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
                driver.ck(driver.cu.cuLaunchKernel(fn, *config["grid"], 128, 1, 1, config["smem"],
                          C.c_void_p(torch.cuda.current_stream().cuda_stream), pointers, None), "tuning launch")

            for family_index, family in enumerate(("zero", "integer", "normal")):
                torch.manual_seed(19191 + family_index)
                if family == "zero":
                    a = torch.zeros((m, k), device="cuda", dtype=dtype)
                    b = torch.ones((n, k), device="cuda", dtype=dtype)
                elif family == "integer":
                    a = torch.randint(-3, 4, (m, k), device="cuda").to(dtype)
                    b = torch.randint(-3, 4, (n, k), device="cuda").to(dtype)
                else:
                    a = torch.randn((m, k), device="cuda", dtype=dtype)
                    b = torch.randn((n, k), device="cuda", dtype=dtype)
                a_before, b_before = a.clone(), b.clone()
                truth = a.float() @ b.float().T
                reference = torch.empty((m, n), device="cuda", dtype=torch.float32)
                launch("baseline", "ptxas", reference)
                torch.cuda.synchronize()
                for name in plans:
                    same_policy = torch.empty_like(reference)
                    launch(name, "ptxas", same_policy)
                    storage = torch.full((m * n + 512,), -9876.5, device="cuda", dtype=torch.float32)
                    output = storage[256:-256].view(m, n)
                    output.fill_(float("nan"))
                    launch(name, "native", output)
                    torch.cuda.synchronize()
                    relative = float((output - truth).norm() / truth.norm().clamp_min(1e-30))
                    check = {"input": family, "relative_l2_fp32": relative,
                             "full_output": relative < 5e-5,
                             "finite_output": bool(torch.isfinite(output).all()),
                             "canaries": bool((storage[:256] == -9876.5).all() and (storage[-256:] == -9876.5).all()),
                             "inputs_unchanged": bool(torch.equal(a, a_before) and torch.equal(b, b_before)),
                             "ptxas_bitwise": bool(torch.equal(output.view(torch.int32), same_policy.view(torch.int32))),
                             "baseline_bitwise": bool(torch.equal(output.view(torch.int32), reference.view(torch.int32)))}
                    if not all(check[gate] for gate in ("full_output", "finite_output", "canaries", "inputs_unchanged",
                                                       "ptxas_bitwise", "baseline_bitwise")):
                        raise ValueError(f"candidate numerical gate failed: {name}/{check}")
                    candidates[name]["checks"].append(check)

            for name in plans:
                # Capture includes the current operands; poisoning A and B after
                # capture ensures replay reads fresh inputs, rather than merely
                # validating an already warmed result buffer.
                output = torch.empty_like(reference)
                graph_outputs[name] = output
                for _ in range(2):
                    launch(name, "native", output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        launch(name, "native", output)
                output.fill_(float("nan"))
                a.copy_(a_before * -1)
                b.copy_(b_before)
                graph.replay()
                torch.cuda.synchronize()
                poison_reference = torch.empty_like(reference)
                launch(name, "ptxas", poison_reference)
                torch.cuda.synchronize()
                passed = bool(torch.isfinite(output).all()) and bool(torch.equal(
                    output.view(torch.int32), poison_reference.view(torch.int32)))
                passed &= bool(torch.equal(a, -a_before) and torch.equal(b, b_before))
                if not passed:
                    raise ValueError(f"poisoned graph gate failed: {name}")
                candidates[name]["poisoned_graph"] = True
                a.copy_(a_before)
                graphs[name] = graph
            torch.cuda.synchronize()
            timings = []
            for index in range(rounds):
                elapsed = {name: [] for name in plans}
                orders = []
                for _ in range(samples):
                    order = list(plans)
                    rng.shuffle(order)
                    orders.append(order)
                    for name in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        elapsed[name].append(start.elapsed_time(end) / 16)
                timings.append({"round": index, "paired_orders": orders, "samples_ms": elapsed,
                                "median_ms": {name: statistics.median(values) for name, values in elapsed.items()}})
            if compiler_identity() != identity:
                raise ValueError("compiler sources changed during GPU tuning")
            result = {"schema": 1, "compiler_identity": identity, "contract": contract_key(plan),
                      "device": device, "candidates": candidates, "timings": timings,
                      "artifact_bytes": artifacts,
                      "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                  "ptxas": subprocess.run(["ptxas", "--version"], check=True, capture_output=True,
                                                           text=True).stdout.strip()},
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.monotonic() - started,
                      "timing_scope": "sixteen-launch CUDA graphs, native kernels only; excludes build, transfers, packing, training"}
            result["selected"] = choose_policy(result)
            return result
    finally:
        graphs.clear()
        torch.cuda.synchronize()
        for module in modules.values():
            driver.unload(module)
