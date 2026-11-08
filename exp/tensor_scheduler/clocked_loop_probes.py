"""Clocked SM80 HMMA loop probes for dependent and independent chains.

Unlike the tiny event-timed probes, this experiment reads `%clock64` inside the
kernel. It subtracts an empty-loop control and reports per-HMMA cycle deltas for
one, two and three accumulator chains. The result is a hardware measurement of
the tested ptxas schedule, not a production compiler policy.

Run on Modal:

    modal run exp/tensor_scheduler/clocked_loop_probes.py --mode run
"""

from __future__ import annotations

import json
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
IMAGE = modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel").add_local_python_source(
    "openassembler", copy=False
)
APP = modal.App("openassembler-a100-clocked-tensor-probes")


def _mma(start: int) -> str:
    return (
        f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        f"{{%f{start},%f{start+1},%f{start+2},%f{start+3}}},"
        f"{{%r1,%r2,%r3,%r4}},{{%r5,%r6}},"
        f"{{%f{start},%f{start+1},%f{start+2},%f{start+3}}};"
    )


def _ptx(kind: str, chains: int, reps: int) -> str:
    if kind not in {"empty", "hmma"}:
        raise ValueError(kind)
    if chains not in (1, 2, 3):
        raise ValueError(chains)
    body = [
        "mov.u32 %r0,%tid.x;",
        "mov.u32 %r7,%ctaid.x;",
        "mov.b32 %r1,0x3c003c00; mov.b32 %r2,0x3c003c00;",
        "mov.b32 %r3,0x3c003c00; mov.b32 %r4,0x3c003c00;",
        "mov.b32 %r5,0x3c003c00; mov.b32 %r6,0x3c003c00;",
    ]
    for start in (0, 4, 8):
        body.extend([f"mov.f32 %f{start},0f00000000;",
                     f"mov.f32 %f{start+1},0f00000000;",
                     f"mov.f32 %f{start+2},0f00000000;",
                     f"mov.f32 %f{start+3},0f00000000;"])
    body.extend([
        "mov.u32 %r20,0;",
        "mov.u64 %rd3,%clock64;",
        "LOOP:",
    ])
    if kind == "hmma":
        for start in range(0, chains * 4, 4):
            body.append(_mma(start))
    body.extend([
        "add.u32 %r20,%r20,1;",
        f"setp.lt.u32 %p1,%r20,{reps};",
        "@%p1 bra LOOP;",
        "mov.u64 %rd4,%clock64;",
        "sub.u64 %rd5,%rd4,%rd3;",
        "mul.wide.u32 %rd6,%r7,32;",
        "add.u64 %rd7,%rd1,%rd6;",
        "st.global.u64 [%rd7],%rd5;",
        "st.global.u32 [%rd7+8],%r20;",
        "st.global.v4.f32 [%rd7+16],{%f0,%f1,%f2,%f3};",
        "ret;",
    ])
    return """.version 8.7
.target sm_80
.address_size 64
.visible .entry probe(.param .u64 out)
{
 .reg .pred %p<4>;
 .reg .b32 %r<32>;
 .reg .b64 %rd<8>;
 .reg .f32 %f<32>;
 ld.param.u64 %rd1,[out];
 cvta.to.global.u64 %rd1,%rd1;
 BODY
}
""".replace(" BODY\n", " " + "\n ".join(body) + "\n")


def probe_specs() -> list[dict]:
    reps = 128
    specs = [{"name": "empty", "kind": "empty", "chains": 1, "reps": reps, "ptx": _ptx("empty", 1, reps)}]
    for chains in (1, 2, 3):
        specs.append({"name": f"hmma_{chains}chain", "kind": "hmma", "chains": chains,
                      "reps": reps, "ptx": _ptx("hmma", chains, reps)})
    return specs


@APP.function(image=IMAGE, gpu="A100-40GB", timeout=600, max_containers=1, retries=0)
def run_remote(specs: list[dict]) -> dict:
    import os
    import struct
    import subprocess
    import torch

    from openassembler.gpu.cuda_driver import Driver

    torch.cuda.init(); drv = Driver(); drv.open()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for spec in specs:
            ptx = os.path.join(td, spec["name"] + ".ptx")
            cubin = os.path.join(td, spec["name"] + ".cubin")
            with open(ptx, "w") as f:
                f.write(spec["ptx"])
            proc = subprocess.run(["ptxas", "-arch=sm_80", "-O3", "-o", cubin, ptx],
                                  capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError({"name": spec["name"], "stderr": proc.stderr})
            binary = open(cubin, "rb").read()
            module = drv.load(binary)
            fn = drv.function(module, "probe")
            out_ptr = drv.alloc(108 * 32)
            args = [__import__("ctypes").c_uint64(out_ptr)]
            def launch():
                drv.memset(out_ptr, 0, 108 * 32)
                drv.launch(fn, (108, 1, 1), (32, 1, 1), args)
            launch()
            samples = []
            for _ in range(9):
                launch()
                raw = drv.dtoh(out_ptr, 108 * 32)
                values = [struct.unpack_from("<Q", raw, lane * 32)[0] for lane in range(108)]
                samples.append({"min": min(values), "median": statistics.median(values),
                                "max": max(values), "values": values})
            medians = [row["median"] for row in samples]
            rows.append({**{k: spec[k] for k in ("name", "kind", "chains", "reps")},
                         "samples": samples, "median_cycles": statistics.median(medians),
                         "sass_bytes": len(binary)})
            drv.free(out_ptr)
            drv.unload(module)
    drv.close()
    empty = next(row for row in rows if row["kind"] == "empty")
    base = empty["median_cycles"]
    for row in rows:
        row["overhead_subtracted_cycles"] = row["median_cycles"] - base
        if row["kind"] == "hmma":
            row["cycles_per_hmma"] = row["overhead_subtracted_cycles"] / (row["chains"] * row["reps"])
    return {"device": drv.info().__dict__ if False else "A100", "clocked": rows,
            "empty_loop": base, "note": "clock64 deltas include loop/control overhead removed by empty control"}


@APP.local_entrypoint()
def main(mode: str = "run"):
    if mode != "run":
        raise SystemExit("mode must be run")
    result = run_remote.remote(probe_specs())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = HERE / f"clocked-loop-probes_{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)

