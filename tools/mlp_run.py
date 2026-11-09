"""Run the 2-layer MLP forward pass (examples/mlp/kernels.py: linear1 -> relu
-> linear2) on a real GPU through Modal, with both ptxas- and our-compiler-
built cubins (tools/mlp_build.py), and compare timing + correctness.

    modal run tools/mlp_run.py --dry-run                      # job count + cost, $0
    modal run tools/mlp_run.py --batch 256 --in-dim 512 --hidden 2048 --out-dim 512

Correctness: every run's output is checked against a pure-Python float64
reference (no numpy) on every element -- small enough matrices that this is
cheap, and it means "the MLP forward pass runs correctly on real hardware"
is backed by an independent reference, not just internal agreement between
our two compilers. Timing: median of --reps runs of the 3-launch sequence,
several shuffled rounds (same "a fixed order biases results" lesson as
tools/gpu_modal.py --timing; see docs/STATUS.md "Timing method").
"""

from __future__ import annotations

import random
import struct
import time

import modal

BUDGET_S = 600
app = modal.App("openassembler-mlp")
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("openassembler", copy=False)


@app.function(image=image, gpu="A100-40GB", timeout=BUDGET_S + 60, max_containers=1, retries=0)
def run_batch(jobs_blob: bytes, job_timeout: float, budget: float) -> dict:
    import os
    import tempfile

    from openassembler.gpu import batch

    d = tempfile.mkdtemp()
    jp, rp = os.path.join(d, "jobs.pkl"), os.path.join(d, "res.pkl")
    with open(jp, "wb") as f:
        f.write(jobs_blob)
    info = batch.device_info()
    t0 = time.monotonic()
    r = batch.run(jp, rp, job_timeout, min(budget, BUDGET_S))
    return {"device": info, "gpu_seconds": round(time.monotonic() - t0, 1), **r}


def _split_cols(n: int, max_block: int = 256) -> tuple[int, int]:
    """n columns -> (block.x, grid.y) so block.x * grid.y >= n and
    block.x <= max_block (well under CUDA's 1024-thread/block cap; 256 keeps
    occupancy reasonable for linear1/linear2's register use)."""
    bx = min(max_block, n)
    gy = (n + bx - 1) // bx
    return bx, gy


def make_jobs(batch: int, in_dim: int, hidden: int, out_dim: int, variants: list[str],
              reps: int, rounds: int, seed: int, arch: str = "sm_80"):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    cdir = root / "corpus" / arch / "mlp"

    rng = random.Random(seed)
    x = [rng.uniform(-1, 1) for _ in range(batch * in_dim)]
    w1 = [rng.uniform(-0.3, 0.3) for _ in range(in_dim * hidden)]
    b1 = [rng.uniform(-0.1, 0.1) for _ in range(hidden)]
    w2 = [rng.uniform(-0.3, 0.3) for _ in range(hidden * out_dim)]
    b2 = [rng.uniform(-0.1, 0.1) for _ in range(out_dim)]

    def pack(vals):
        return struct.pack(f"<{len(vals)}f", *vals)

    buffers = {"x": 4 * batch * in_dim, "w1": 4 * in_dim * hidden, "b1": 4 * hidden,
               "h": 4 * batch * hidden, "w2": 4 * hidden * out_dim, "b2": 4 * out_dim,
               "out": 4 * batch * out_dim}
    init = {"x": pack(x), "w1": pack(w1), "b1": pack(b1), "w2": pack(w2), "b2": pack(b2)}

    jobs = []
    for rnd in range(rounds):
        order = list(variants)
        random.Random(seed + rnd).shuffle(order)
        for v in order:
            mods = {k: (cdir / f"{v}@{k}.cubin").read_bytes() for k in ("linear1", "relu", "linear2")}
            # linear1/linear2's column index is tid.x + ctaid.y * ntid.x (see
            # examples/mlp/kernels.py), so a column count over CUDA's 1024
            # threads/block limit splits across block.x and grid.y instead.
            bx_hidden, gy_hidden = _split_cols(hidden)
            bx_out, gy_out = _split_cols(out_dim)
            steps = [
                ("linear1", "linear1", (batch, gy_hidden, 1), (bx_hidden, 1, 1), 0,
                 [("buf", "x", 0), ("buf", "w1", 0), ("buf", "b1", 0), ("buf", "h", 0),
                  ("u32", batch), ("u32", in_dim), ("u32", hidden)]),
                ("relu", "relu", ((batch * hidden + 255) // 256, 1, 1), (256, 1, 1), 0,
                 [("buf", "h", 0), ("buf", "h", 0), ("u32", batch * hidden)]),
                ("linear2", "linear2", (batch, gy_out, 1), (bx_out, 1, 1), 0,
                 [("buf", "h", 0), ("buf", "w2", 0), ("buf", "b2", 0), ("buf", "out", 0),
                  ("u32", batch), ("u32", hidden), ("u32", out_dim)]),
            ]
            # run_plan_job allocates fresh buffers per job (no sharing across
            # jobs in a batch), so every job needs its own copy of init and
            # must read its output back to be checked against the reference.
            jobs.append({"mode": "plan", "name": f"mlp|{v}|{rnd}", "modules": mods, "buffers": buffers,
                         "init": init, "steps": steps, "outputs": {"out": [(0, buffers["out"])]},
                         "reps": reps, "warmup": 3})
    return jobs, (x, w1, b1, w2, b2)


def _reference(x, w1, b1, w2, b2, batch, in_dim, hidden, out_dim):
    """Pure-Python float64 forward pass -- independent of both compilers."""
    out = [0.0] * (batch * out_dim)
    h = [0.0] * (batch * hidden)
    for r in range(batch):
        for c in range(hidden):
            acc = b1[c]
            for k in range(in_dim):
                acc += x[r * in_dim + k] * w1[k * hidden + c]
            h[r * hidden + c] = max(acc, 0.0)
    for r in range(batch):
        for c in range(out_dim):
            acc = b2[c]
            for k in range(hidden):
                acc += h[r * hidden + k] * w2[k * out_dim + c]
            out[r * out_dim + c] = acc
    return out


@app.local_entrypoint()
def main(batch: int = 64, in_dim: int = 128, hidden: int = 256, out_dim: int = 64,
         variants: str = "ptxas,ours", gpu: str = "A100-40GB", reps: int = 5, rounds: int = 2,
         seed: int = 1, budget: float = 300, job_timeout: float = 60, dry_run: bool = False) -> None:
    import json
    import pickle
    from pathlib import Path

    jobs, (x, w1, b1, w2, b2) = make_jobs(batch, in_dim, hidden, out_dim, variants.split(","),
                                          reps, rounds, seed)
    blob = pickle.dumps(jobs)
    flops = 2 * batch * in_dim * hidden + 2 * batch * hidden * out_dim
    print(f"{len(jobs)} jobs, {len(blob) / 2**20:.1f} MB to upload, {flops / 1e6:.1f} MFLOP/pass, GPU {gpu}")
    if dry_run:
        return
    t0 = time.time()
    res = run_batch.with_options(gpu=gpu).remote(blob, job_timeout, budget)
    dev = res["device"]
    print(f"{dev['name']} cc {dev['cc']}: {res['gpu_seconds']}s in container, {time.time() - t0:.0f}s wall")

    ref = _reference(x, w1, b1, w2, b2, batch, in_dim, hidden, out_dim)
    rows = {}
    for i, j in enumerate(jobs):
        _, v, rnd = j["name"].split("|")
        r = res["results"].get(i, {})
        row = rows.setdefault(v, {})
        if "error" in r:
            row.setdefault("errors", []).append(r["error"][:80])
            continue
        t = r.get("time_ms", {}).get("median")
        if t is not None and (row.get("ms") is None or t < row["ms"]):
            row["ms"] = t
        if "out" in r.get("out", {}):
            # run_plan_job's "out" maps each buffer to a LIST of byte spans
            # (one per (offset, nbytes) in "outputs"); we declared one span.
            got = struct.unpack(f"<{batch * out_dim}f", r["out"]["out"][0])
            max_err = max(abs(a - b) for a, b in zip(got, ref))
            row["max_abs_err"] = max(row.get("max_abs_err", 0.0), max_err)

    print(f"\n{'variant':8} {'ms':>9} {'max |err| vs float64 ref':>26}")
    for v, row in rows.items():
        ms = row.get("ms")
        err = row.get("max_abs_err")
        print(f"{v:8} {ms if ms is not None else float('nan'):9.4f} "
              f"{err if err is not None else float('nan'):26.2e} "
              f"{'' if row.get('ms') else (row.get('errors') or [''])[0]}")

    tag = gpu.lower().replace("!", "").replace("-", "_")
    out = Path(__file__).resolve().parent.parent / "corpus" / "sm_80" / \
        f"mlp_{tag}_b{batch}_i{in_dim}_h{hidden}_o{out_dim}.json"
    if out.exists():
        # the committed file is the recorded run; keep it, write this one beside it
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.write_text(json.dumps({"device": dev, "shape": {"batch": batch, "in_dim": in_dim,
                                                          "hidden": hidden, "out_dim": out_dim},
                                "results": rows}, indent=1))
    print(f"-> {out}")
