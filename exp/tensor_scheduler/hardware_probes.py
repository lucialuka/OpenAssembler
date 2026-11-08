"""A100 exact-SASS tensor dependency probes.

This experiment compiles tiny SM80 PTX kernels with the CUDA toolkit inside a
Modal A100 container, rewrites only the common SASS control stall field, checks
the result with nvdisasm, then runs correctness and paired timing sweeps.  It is
intentionally separate from the production OpenAssembler tensor backend.

Examples:

    modal run exp/tensor_scheduler/hardware_probes.py --mode run --budget 600
    modal run exp/tensor_scheduler/hardware_probes.py --mode preflight

The first run records the hardware envelope.  It does not claim that any
candidate is safe to promote until the returned JSON is inspected together
with the full-output and counter gates.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal


# Modal uploads this module as /root/hardware_probes.py, so parent traversal is
# not stable inside the container. The remote function does not need ROOT; the
# local entrypoint uses HERE, which remains stable in the repository checkout.
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel").add_local_python_source(
    "openassembler", copy=False
)
APP = modal.App("openassembler-a100-tensor-dependency-probes")


def _mma(start: int = 0, a: str = "%r1,%r2,%r3,%r4", b: str = "%r5,%r6") -> str:
    return (
        f" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        f" {{%f{start},%f{start + 1},%f{start + 2},%f{start + 3}}}, {{{a}}}, {{{b}}}, "
        f" {{%f{start},%f{start + 1},%f{start + 2},%f{start + 3}}};"
    )


def _base_init() -> str:
    return """
 mov.u32 %r0, %tid.x;
 mov.b32 %r1, 0x3c003c00;
 mov.b32 %r2, 0x3c003c00;
 mov.b32 %r3, 0x3c003c00;
 mov.b32 %r4, 0x3c003c00;
 mov.b32 %r5, 0x3c003c00;
 mov.b32 %r6, 0x3c003c00;
 mov.f32 %f0, 0f00000000;
 mov.f32 %f1, 0f00000000;
 mov.f32 %f2, 0f00000000;
 mov.f32 %f3, 0f00000000;
 mov.f32 %f4, 0f00000000;
 mov.f32 %f5, 0f00000000;
 mov.f32 %f6, 0f00000000;
 mov.f32 %f7, 0f00000000;
 mov.f32 %f8, 0f00000000;
 mov.f32 %f9, 0f00000000;
 mov.f32 %f10, 0f00000000;
 mov.f32 %f11, 0f00000000;
 mov.f32 %f12, 0f00000000;
 mov.f32 %f13, 0f00000000;
"""


def _store(start: int = 0, offset: int = 0) -> str:
    suffix = f"+{offset}" if offset else ""
    return f" st.global.v4.f32 [%rd1{suffix}], {{%f{start},%f{start + 1},%f{start + 2},%f{start + 3}}};"


def _ptx(body: str, *, shared: bool = False) -> str:
    smem = ".shared .align 16 .b8 smem[4096];" if shared else ""
    return f""".version 8.7
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 out, .param .u64 inp)
{{
 .reg .pred %p<8>;
 .reg .b32 %r<64>;
 .reg .f32 %f<64>;
 .reg .b64 %rd<8>;
 {smem}
 ld.param.u64 %rd1, [out];
 ld.param.u64 %rd2, [inp];
 cvta.to.global.u64 %rd1, %rd1;
 cvta.to.global.u64 %rd2, %rd2;
 {_base_init()}
{body}
 ret;
}}
"""


def _cases() -> list[dict]:
    cases: list[dict] = []
    for count in (8, 16, 32):
        body = "".join(_mma(0) for _ in range(count)) + _store(0)
        cases.append({"name": f"dependent_{count}", "ptx": _ptx(body), "target": "HMMA_ALL", "family": "dependent"})

    for chains in (2, 3):
        body = ""
        for i in range(16):
            body += _mma(0)
            if chains >= 2:
                body += _mma(4)
            if chains >= 3:
                body += _mma(8)
        body += _store(0, 0)
        if chains >= 2:
            body += _store(4, 16)
        if chains >= 3:
            body += _store(8, 32)
        cases.append({"name": f"independent_{chains}x16", "ptx": _ptx(body), "target": "HMMA_ALL", "family": "independent"})

    shared_init = """
 mov.u32 %r20, smem;
 shl.b32 %r21, %r0, 4;
 add.u32 %r20, %r20, %r21;
 st.shared.v4.u32 [%r20], {%r1,%r2,%r3,%r4};
 bar.sync 0;
"""
    reused = shared_init + " ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10,%r11,%r12,%r13}, [%r20];" + "".join(
        _mma(0, a="%r10,%r11,%r12,%r13") for _ in range(16)
    ) + _store()
    fresh = shared_init + "".join(
        f" ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%r10,%r11,%r12,%r13}}, [%r20+{16 * (i % 8)}];" + _mma(0, a="%r10,%r11,%r12,%r13")
        for i in range(16)
    ) + _store()
    cases.extend([
        {"name": "ldsm_reused", "ptx": _ptx(reused, shared=True), "target": "LDSM_FIRST", "family": "ldsm"},
        {"name": "ldsm_fresh", "ptx": _ptx(fresh, shared=True), "target": "LDSM_ALL", "family": "ldsm"},
    ])

    async_body = """
 mov.u32 %r20, smem;
 shl.b32 %r21, %r0, 4;
 add.u32 %r20, %r20, %r21;
 cp.async.cg.shared.global [%r20], [%rd2], 16;
 cp.async.commit_group;
 cp.async.wait_group 0;
 bar.sync 0;
 ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10,%r11,%r12,%r13}, [%r20];
""" + "".join(_mma(0, a="%r10,%r11,%r12,%r13") for _ in range(16)) + _store(0)
    cases.append({"name": "async_ldsm_hmma", "ptx": _ptx(async_body, shared=True), "target": "HMMA_ALL", "family": "async"})

    observers = {
        "observer_mov": " mov.b32 %r20, %f0;" + _store(0),
        "observer_ffma": " fma.rn.f32 %f20, %f0, %f1, %f2;" + _store(0),
        "observer_store": _store(0),
        "observer_bar": " bar.sync 0;" + _store(0),
        "observer_branch": " setp.eq.u32 %p1, %r0, %r0; @%p1 bra L_done; mov.u32 %r20, %r0; L_done:" + _store(0),
    }
    for name, tail in observers.items():
        body = "".join(_mma(0) for _ in range(8)) + tail
        cases.append({"name": name, "ptx": _ptx(body, shared=name == "observer_bar"), "target": "HMMA_LAST", "family": "observer"})

    war = shared_init + " ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10,%r11,%r12,%r13}, [%r20];" \
        " add.u32 %r20, %r20, %r0;" + _mma(0, a="%r10,%r11,%r12,%r13") + _store(0)
    cases.append({"name": "address_war", "ptx": _ptx(war, shared=True), "target": "LDSM_FIRST", "family": "address_war"})
    return cases


def _parse_listing(text: str) -> list[dict]:
    insn = re.compile(r"/\*(?P<addr>[0-9a-f]{4,})\*/\s+(?P<text>.*?)\s*;\s*/\*\s*0x(?P<lo>[0-9a-f]{16})\s*\*/")
    hi = re.compile(r"^\s*/\*\s*0x(?P<hi>[0-9a-f]{16})\s*\*/\s*$")
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines[:-1]):
        m = insn.search(line)
        if not m:
            continue
        h = hi.match(lines[i + 1])
        if not h:
            continue
        rows.append({"addr": int(m["addr"], 16), "text": m["text"].strip(),
                     "word": int(m["lo"], 16) | (int(h["hi"], 16) << 64)})
    return rows


def _metric_names() -> list[str]:
    return [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed.sum",
        "smsp__inst_executed_pipe_tensor.sum",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "dram__bytes.sum",
        "lts__t_sector_hit_rate.pct",
    ]


def _parse_ncu_csv(text: str) -> dict:
    """Keep only requested metric values from Nsight Compute CSV output."""
    import csv
    import io

    rows = list(csv.reader(io.StringIO(text)))
    header_index = next((i for i, row in enumerate(rows) if row and row[0] == "ID"), None)
    if header_index is None or header_index + 1 >= len(rows):
        return {"status": "no_csv_row"}
    header = rows[header_index]
    data = next((row for row in rows[header_index + 1:]
                 if row and row[0].strip().isdigit() and len(row) >= len(header)), None)
    if data is None:
        return {"status": "no_numeric_csv_row"}
    values = {}
    for name in _metric_names():
        if name in header:
            value = data[header.index(name)] if header.index(name) < len(data) else ""
            values[name] = value
    return {"status": "parsed", "metrics": values,
            "kernel_name": data[header.index("Kernel Name")] if "Kernel Name" in header else ""}


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=1500, max_containers=1, retries=0)
def run_remote(cases: list[dict], profile: bool, budget: float) -> dict:
    import ctypes as C
    import csv
    import io
    import os
    import statistics
    import tempfile
    import torch

    from openassembler.container.cubin import regcount, splice_text, text_words
    from openassembler.gpu.cuda_driver import Driver
    from openassembler.isa.word import set_bits

    def compile_ptx(ptx: str, path: str) -> bytes:
        ptx_path = path + ".ptx"
        cubin_path = path + ".cubin"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        cmd = ["ptxas", "-arch=sm_80", "-O3", "-o", cubin_path, ptx_path]
        proc = __import__("subprocess").run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError({"cmd": cmd, "stderr": proc.stderr, "stdout": proc.stdout})
        return open(cubin_path, "rb").read()

    def disasm(cubin_path: str) -> tuple[str, list[dict]]:
        proc = __import__("subprocess").run(["nvdisasm", "-hex", "-c", cubin_path], capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError({"nvdisasm": proc.stderr})
        return proc.stdout, _parse_listing(proc.stdout)

    def exits(rows: list[dict]) -> list[int]:
        return [row["addr"] for row in rows if row["text"].split()[0] == "EXIT"]

    drv = Driver(); drv.open(); torch.cuda.init()
    device = drv.info().__dict__
    results = {"device": device, "cases": [], "profile_metrics": _metric_names(), "ptxas": "sm_80 -O3"}

    def load_and_launch(binary: bytes, out, inp, smem: int):
        module = drv.load(binary)
        fn = drv.function(module, "probe")
        if smem:
            drv.ck(drv.cu.cuFuncSetAttribute(fn, 8, smem), "shared memory")
        args = [C.c_uint64(out.data_ptr()), C.c_uint64(inp.data_ptr())]
        params = (C.c_void_p * 2)(*[C.cast(C.pointer(arg), C.c_void_p) for arg in args])
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        def launch():
            drv.ck(drv.cu.cuLaunchKernel(fn, 108, 1, 1, 32, 1, 1, smem, stream, params, None), "probe launch")
        return module, launch

    for spec in cases:
        with tempfile.TemporaryDirectory() as td:
            base = compile_ptx(spec["ptx"], os.path.join(td, spec["name"]))
            base_path = os.path.join(td, spec["name"] + ".cubin")
            raw = base
            with open(base_path, "wb") as f:
                f.write(base)
            sass, rows = disasm(base_path)
            words = text_words(base, "probe")
            hmmas = [row["addr"] // 16 for row in rows if row["text"].startswith("HMMA")]
            ldsms = [row["addr"] // 16 for row in rows if row["text"].startswith("LDSM")]
            if not hmmas:
                raise RuntimeError({"case": spec["name"], "reason": "no HMMA in ptxas output", "sass": sass})
            if spec["target"] == "HMMA_ALL":
                targets = hmmas
            elif spec["target"] == "HMMA_LAST":
                targets = hmmas[-1:]
            elif spec["target"] == "LDSM_ALL":
                targets = ldsms
            else:
                targets = ldsms[:1]
            nregs = regcount(base, "probe")
            exit_offsets = exits(rows)
            variants = []
            # Base ptxas schedule plus explicit control sweeps. The raw base is
            # retained as the reference; only the selected field is changed.
            variants.append(("ptxas", base, None))
            for stall in range(1, 12):
                mutated = list(words)
                for index in targets:
                    mutated[index] = set_bits(mutated[index], 105, 4, stall)
                variant = splice_text(base, "probe", mutated, nregs, exit_offsets)
                vp = os.path.join(td, f"{spec['name']}_s{stall}.cubin")
                with open(vp, "wb") as f:
                    f.write(variant)
                check = __import__("subprocess").run(["nvdisasm", "-c", vp], capture_output=True, text=True)
                variants.append((f"s{stall}", variant, check.returncode == 0))

            inp = torch.arange(4096, device="cuda", dtype=torch.uint8)
            reference_out = torch.full((32,), float("nan"), device="cuda", dtype=torch.float32)
            ref_mod, ref_launch = load_and_launch(base, reference_out, inp, 4096 if "smem" in spec["ptx"] else 0)
            ref_launch(); torch.cuda.synchronize(); reference_bits = reference_out.view(torch.int32).clone()
            drv.unload(ref_mod)
            rows_out = []
            for label, binary, disasm_ok in variants:
                out = torch.full((32,), float("nan"), device="cuda", dtype=torch.float32)
                smem = 4096 if "smem" in spec["ptx"] else 0
                module, launch = load_and_launch(binary, out, inp, smem)
                launch(); torch.cuda.synchronize()
                bits_ok = bool(torch.equal(out.view(torch.int32), reference_bits))
                samples = []
                for _ in range(4):
                    for _ in range(10):
                        launch()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        launch()
                    end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end) / 100.0)
                rows_out.append({"label": label, "disasm_ok": disasm_ok, "output_match": bits_ok,
                                 "timing_ms": statistics.median(samples), "timing_samples_ms": samples,
                                 "cubin_sha256": __import__("hashlib").sha256(binary).hexdigest(),
                                 "cubin_b64": base64.b64encode(binary).decode() if label == "ptxas" else ""})
                drv.unload(module)
            valid = [row for row in rows_out if row["disasm_ok"] and row["output_match"]]
            best = min(valid, key=lambda row: row["timing_ms"]) if valid else None
            case_row = {"name": spec["name"], "family": spec["family"], "target": spec["target"],
                        "hmma_count": len(hmmas), "ldsm_count": len(ldsms), "registers": nregs,
                        "tensor_texts": [row["text"] for row in rows if row["text"].startswith(("HMMA", "LDSM", "LDGSTS", "DEPBAR", "BAR"))],
                        "variants": rows_out, "best_valid": best["label"] if best else None,
                        "baseline_ms": next(row["timing_ms"] for row in rows_out if row["label"] == "ptxas")}
            if profile and best and spec["name"] in {"dependent_16", "independent_2x16", "independent_3x16",
                                                       "ldsm_reused", "async_ldsm_hmma", "observer_store", "address_war"}:
                profile_rows = []
                for role in ("ptxas", best["label"]):
                    selected = next(row for row in rows_out if row["label"] == role)
                    if not selected.get("cubin_b64"):
                        # Reconstruct the selected variant from the same raw
                        # words; keeping this branch local avoids storing every
                        # candidate in the result artifact.
                        if role == "ptxas":
                            binary = base
                        else:
                            stall = int(role[1:]); mutated = list(words)
                            for index in targets:
                                mutated[index] = set_bits(mutated[index], 105, 4, stall)
                            binary = splice_text(base, "probe", mutated, nregs, exit_offsets)
                    else:
                        binary = base64.b64decode(selected["cubin_b64"])
                    worker = os.path.join(td, f"worker_{spec['name']}_{role}.py")
                    with open(worker, "w") as f:
                        f.write("""import base64, ctypes as C, sys, torch\nfrom openassembler.gpu.cuda_driver import Driver\nraw=base64.b64decode(sys.argv[1]); d=Driver(); d.open(); torch.cuda.init(); m=d.load(raw); fn=d.function(m,'probe'); out=torch.empty((32,),device='cuda',dtype=torch.float32); inp=torch.empty((4096,),device='cuda',dtype=torch.uint8); torch.cuda.synchronize(); args=[C.c_uint64(out.data_ptr()),C.c_uint64(inp.data_ptr())]; params=(C.c_void_p*2)(*[C.cast(C.pointer(a),C.c_void_p) for a in args]); s=C.c_void_p(torch.cuda.current_stream().cuda_stream); d.cu.cuLaunchKernel(fn,108,1,1,32,1,1,4096 if len(sys.argv)>2 and sys.argv[2]=='smem' else 0,s,params,None); d.cu.cuCtxSynchronize(); print('OA_TENSOR_PROBE_DONE'); d.unload(m); d.close()\n""")
                    cmd = ["ncu", "--clock-control", "none", "--target-processes", "all", "--csv", "--page", "raw",
                           "--metrics", ",".join(_metric_names()), sys.executable, worker, base64.b64encode(binary).decode()]
                    if "smem" in spec["ptx"]:
                        cmd.append("smem")
                    prof = __import__("subprocess").run(cmd, capture_output=True, text=True, timeout=360)
                    profile_rows.append({"role": role, "returncode": prof.returncode,
                                        "csv": _parse_ncu_csv(prof.stdout),
                                        "stdout_tail": prof.stdout[-8000:],
                                        "stderr_excerpt": prof.stderr[-3000:]})
                case_row["ncu_profiles"] = profile_rows
            results["cases"].append(case_row)
    drv.close()
    return results


@APP.local_entrypoint()
def main(mode: str = "preflight", budget: float = 600, profile: bool = False, limit: int = 0):
    if mode == "preflight":
        print(json.dumps({"modal_image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel",
                          "gpu": "A100-40GB", "probe_count": len(_cases()),
                          "metric_hints": _metric_names()}, indent=2))
        return
    if mode != "run":
        raise SystemExit("mode must be preflight or run")
    cases = _cases()
    if limit:
        cases = cases[:limit]
    started = time.time()
    result = run_remote.remote(cases, profile, budget)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = HERE / f"hardware-probes_{stamp}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(out), "cases": len(result.get("cases", [])),
                      "elapsed_s": round(time.time() - started, 1)}, indent=2))

