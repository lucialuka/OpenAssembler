"""`openassembler run` (and the run half of `cnr`): execute already-compiled cubins on
a real GPU, ours next to ptxas's, on identical inputs, and compare bit for bit.

Reads OUTDIR/manifest.json (schema: openassembler.project). For every selected kernel
it builds two GPU jobs in openassembler.gpu.cuda_driver.run_plan_job's format -- one
loading our cubin (cubin/<name>.cubin), one loading the ptxas -O3 reference build of
the same PTX (.build/reference/<name>.cubin) -- with the same buffers, the same
initial data and the same launches. Both run in openassembler.gpu.batch (a worker
subprocess per fault, so one kernel faulting doesn't stop the others), every compared
buffer is read back after the first run and compared byte for byte, then the step
list is timed `reps` times (cuEvents, median reported).

Inputs, two ways:
  * auto-harness (default), like openassembler.verify: every pointer (.u64) param is a
    buffer of N 4-byte elements filled with random f32 data in [-1, 1); integer params
    named n/count/size/len/num/length (any case) are N, other integers are 16 (small:
    they are usually dimensions or strides, and a big one would index past N); float
    params are random. 1-D launch, block 256, grid ceil(N/256). Kernels with a
    different geometry fault or read past the buffers -- that is reported per kernel,
    with the hint to describe the launch with --spec.
  * --spec SPEC.json, for real workloads and multi-kernel pipelines:
      {"buffers": {"x": {"count": 4096, "dtype": "f32", "init": "random"},
                   "y": {"count": 4096, "dtype": "i32", "init": "zeros"},      init: random | zeros | [values...]
                   "w": {"count": 3, "dtype": "u32", "init": [1, 2, 3]}},      dtype: f32 | i32 | u32 (4 bytes each)
       "steps": [{"kernel": "relu", "source": "examples/mlp/kernels.py",      source: only if the name is ambiguous
                  "grid": 16, "block": [256, 1, 1], "smem": 0,                 grid/block: int or [x, y, z]
                  "args": [{"buf": "x"}, {"buf": "y", "offset": 0},            offset in bytes (default 0)
                           {"i32": 4096}, {"u32": 7}, {"f32": 0.5}, {"u64": 1}]}],
       "compare": ["y"],                                                       default: every buffer
       "reps": 20}                                                             default: --reps
    Random i32/u32 data is uniform in [0, count) so it can serve as indices. Each step's
    kernel runs from our cubin in the "ours" job and from the reference cubin in the
    "ptxas" job; a pipeline is one comparison.

Backends: local (this Linux machine's NVIDIA driver), user@host (SSH, like
tools/gpu_run.py: copies only the stdlib-only openassembler/gpu package, a hard remote
`timeout`, results copied back, remote dir removed), modal (openassembler/gpu/modal_app.py
via the `modal` CLI, so modal can live in any Python env). --dry-run never touches a
GPU: it prints the jobs, upload size and (modal) an estimated cost. The last report is
saved to OUTDIR/run.json.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import platform
import random
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from openassembler import __version__
from openassembler import project as P

ROOT = Path(__file__).resolve().parent.parent
GPU_PKG = ROOT / "openassembler" / "gpu"
MODAL_APP = GPU_PKG / "modal_app.py"
N_LIKE = {"n", "count", "size", "len", "num", "length"}
SMALL_INT = 16
DEFAULT_N = 1 << 20
BLOCK = 256
DTYPES = {"f32": "<f", "i32": "<i", "u32": "<I", "f16": "<e", "bf16": "<H"}
_PRICE_FALLBACK = {"A100-40GB": 0.000583, "A100": 0.000583, "A100-80GB": 0.000694, "A10": 0.000306,
                   "L4": 0.000222, "T4": 0.000164, "L40S": 0.000542, "H100": 0.001097}
FAULT_HINT = ("if the auto-harness guessed this kernel's launch wrong (2-D grid, other buffer sizes, "
              "dynamic shared memory), describe it with --spec SPEC.json (see `openassembler run --help`)")


class RunError(Exception):
    """One clear, user-facing line; the run command prints it and exits 1."""


@dataclass
class Target:
    entry: dict
    kernel: dict
    source: Path                       # absolute

    @property
    def label(self) -> str:
        return self.kernel["name"]


@dataclass
class Case:
    """One comparison: job `ours` vs job `ref` (indices into the job list)."""
    name: str
    source: str
    ours: int
    ref: int
    compare: list[str]
    note: str = ""
    skip: str = ""                     # set: no jobs, reported as SKIP
    describe: str = ""
    expected_f32_constant: dict[str, float] = field(default_factory=dict)


@dataclass
class Plan:
    jobs: list[dict] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---- manifest -> targets ---------------------------------------------------------------

def load(out_dir: Path) -> tuple[dict, list[str]]:
    """(manifest, warnings). RunError for: no output folder, no manifest, corrupt."""
    out_dir = Path(out_dir)
    nothing = ("nothing compiled yet: run openassembler compile <src> first, "
               "or openassembler cnr <src>")
    if not out_dir.exists():
        raise RunError(f"no output folder {out_dir}: {nothing}")
    if not out_dir.is_dir():
        raise RunError(f"{out_dir} is a file, not an output folder")
    try:
        m = P.load_manifest(out_dir)
    except P.ManifestError as e:
        raise RunError(str(e)) from None
    if m is None:
        raise RunError(f"no {out_dir / P.MANIFEST}: {nothing}")
    warns = []
    if m.get("openassembler") != __version__:
        warns.append(f"{out_dir / P.MANIFEST} was written by openassembler {m.get('openassembler', '?')}, "
                     f"this is {__version__}: recompile (openassembler cnr <src>) if anything looks off")
    return m, warns


def _matches_source(m: dict, e: dict, want: str) -> bool:
    w = Path(want)
    try:
        if w.exists() and w.resolve() == P.entry_source_abs(m, e).resolve():
            return True
    except OSError:
        pass
    return e["source"] == want or e["source"].endswith("/" + want.lstrip("./")) or Path(e["source"]).name == want


def select(m: dict, out_dir: Path, kernel: str | None, source: str | None,
           only: set[tuple[str, str]] | None = None) -> tuple[list[Target], list[str]]:
    """Kernels to run (manifest order), plus warnings (stale sources). `only` restricts
    to (abs source, kernel) pairs -- cnr passes what it just compiled."""
    warns: list[str] = []
    entries = m["entries"]
    if source is not None:
        entries = [e for e in entries if _matches_source(m, e, source)]
        if not entries:
            raise RunError(f"no compiled source matches --source {source!r}; have "
                           f"{sorted(e['source'] for e in m['entries'])}")
    targets = [Target(e, k, P.entry_source_abs(m, e).resolve()) for e in entries for k in e["kernels"]]
    if only is not None:
        targets = [t for t in targets if (str(t.source), t.label) in only]
    if kernel is not None:
        hits = [t for t in targets if t.label == kernel]
        if not hits:
            have = sorted({t.label for t in targets})
            raise RunError(f"kernel {kernel!r} not found in {out_dir / P.MANIFEST}; have {have}")
        if len({t.entry["cubin"] for t in hits}) > 1 and source is None:
            raise RunError(f"kernel {kernel!r} is in {len(hits)} sources "
                           f"({', '.join(sorted(t.entry['source'] for t in hits))}); pick one with --source FILE")
        targets = hits
    if not targets:
        raise RunError(f"{out_dir / P.MANIFEST} lists no compiled kernels")
    for e in {id(t.entry): t.entry for t in targets}.values():
        if e.get("build_failed"):
            raise RunError(f"the latest build of {e['source']} failed; recompile successfully before running")
        for key in ("cubin", "reference"):
            if not e.get(key) or not (out_dir / e[key]).is_file():
                raise RunError(f"{out_dir / (e.get(key) or '?')} is listed in the manifest but missing; "
                               f"recompile with openassembler compile {e['source']}")
        src = P.entry_source_abs(m, e)
        if not src.exists():
            warns.append(f"{e['source']} no longer exists; running the cubin compiled from it earlier")
        elif e.get("source_sha256") and P.sha256_file(src) != e["source_sha256"]:
            warns.append(f"{e['source']} changed since it was compiled: out of date, run "
                         f"openassembler cnr {e['source']} to recompile")
    return targets, warns


# ---- jobs --------------------------------------------------------------------------

def _seed(*parts) -> int:
    """Deterministic per-buffer seed (same data for ours and ptxas, every run)."""
    import hashlib

    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little")


def _f32bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _job(name: str, modules: dict, buffers: dict, init: dict, init_rand: dict, steps: list,
         outputs: dict, reps: int) -> dict:
    return {"mode": "plan", "name": name, "modules": modules, "buffers": buffers, "init": init,
            "init_rand": init_rand, "steps": steps, "outputs": outputs, "reps": reps,
            "warmup": 3 if reps else 0}


def auto_case(t: Target, out_dir: Path, n: int, reps: int, plan: Plan) -> None:
    """The verify.py-style harness for one kernel: see this module's docstring."""
    k = t.kernel
    if k.get("gemm") is not None:
        return gemm_case(t, out_dir, reps, plan)
    if k.get("rowwise") is not None:
        return rowwise_case(t, out_dir, reps, plan)
    if k.get("tensor") is not None:
        plan.cases.append(Case(k["name"], t.entry["source"], -1, -1, [],
                               skip="tensor kernels need --spec with their actual buffers, precision, "
                                    "grid and shared memory, or a validated `gemm` launch contract"))
        return
    buffers, init_rand, args, desc = {}, {}, [], []
    rng = random.Random(_seed(t.entry["cubin"], k["name"], n))
    src_names = k.get("source_params") or [p[1] for p in k["params"]]
    for i, ((ty, ptx_name), pname) in enumerate(zip(k["params"], src_names)):
        ty = ty.lstrip(".")
        if ty in ("u64", "s64", "b64") and pname.lower() not in N_LIKE:
            if pname in buffers:
                pname = ptx_name
            buffers[pname] = 4 * n
            init_rand[pname] = (_seed(t.entry["cubin"], k["name"], pname, i), n)
            args.append(("buf", pname, 0))
            desc.append(f"{pname}=f32[{n}]")
        elif ty in ("u32", "s32", "b32", "u16", "s16", "b16", "u8", "s8", "b8"):
            v = n if pname.lower() in N_LIKE else SMALL_INT
            args.append(("u32", v & 0xFFFFFFFF))
            desc.append(f"{pname}={v}")
        elif ty in ("u64", "s64", "b64"):
            args.append(("u64", n))
            desc.append(f"{pname}={n}")
        elif ty == "f32":
            v = rng.uniform(-1, 1)
            args.append(("f32bits", _f32bits(v)))
            desc.append(f"{pname}={v:.4g}")
        elif ty == "f64":
            v = rng.uniform(-1, 1)
            args.append(("u64", struct.unpack("<Q", struct.pack("<d", v))[0]))
            desc.append(f"{pname}={v:.4g}")
        else:
            plan.cases.append(Case(k["name"], t.entry["source"], -1, -1, [],
                                   skip=f"parameter {pname!r} has type .{ty}, which the auto-harness can't "
                                        f"fill; describe the launch with --spec SPEC.json"))
            return
    grid = max(1, math.ceil(n / BLOCK))
    steps_ours = [("ours", k["name"], (grid, 1, 1), (BLOCK, 1, 1), 0, args)]
    steps_ref = [("ref", k["name"], (grid, 1, 1), (BLOCK, 1, 1), 0, args)]
    outputs = {b: [(0, sz)] for b, sz in buffers.items()}
    ours_bytes = (out_dir / t.entry["cubin"]).read_bytes()
    ref_bytes = (out_dir / t.entry["reference"]).read_bytes()
    base = f"{t.entry['cubin']}:{k['name']}"
    i = len(plan.jobs)
    plan.jobs.append(_job(base + "#ours", {"ours": ours_bytes}, buffers, {}, init_rand, steps_ours, outputs, reps))
    plan.jobs.append(_job(base + "#ptxas", {"ref": ref_bytes}, buffers, {}, init_rand, steps_ref, outputs, reps))
    note = "ptxas-fallback: our cubin holds ptxas's code for this kernel" \
        if k.get("compiled_by") == "ptxas-fallback" else ""
    plan.cases.append(Case(k["name"], t.entry["source"], i, i + 1, list(buffers), note=note,
                           describe=f"grid {grid} x block {BLOCK}; " + ", ".join(desc)))


def gemm_case(t: Target, out_dir: Path, reps: int, plan: Plan) -> None:
    """Use the checked generated GEMM contract, with exact small-integer inputs."""
    from openassembler.kernels.gemm import GemmPlan
    contract = t.kernel["gemm"]
    try:
        allowed = ("m", "n", "k", "dtype", "schedule", "fragment_schedule", "tile", "output_dtype", "layout", "arch",
                   "cta_order", "epilogue", "pointer_mode", "pointer_advance", "tile", "address_init")
        g = GemmPlan(**{key: contract[key] for key in allowed if key in contract})
        if contract.get("schema") != 1 or contract.get("launch") != g.launch:
            raise ValueError("launch contract differs from its validated plan")
        if 9 * g.k > 2**24:
            raise ValueError("K exceeds the exact small-integer smoke comparison range; use --spec")
    except (ValueError, TypeError, KeyError) as exc:
        raise RunError(f"invalid GEMM launch contract for {t.label}: {exc}") from None
    buffers = {"A": 2*g.m*g.k, "B": 2*g.n*g.k, "C": 4*g.m*g.n}
    initialize = {"A": (_seed("gemm", g.dtype, "A"), g.m*g.k, g.dtype),
                  "B": (_seed("gemm", g.dtype, "B"), g.n*g.k, g.dtype)}
    names = ["A", "B", "C"]
    if g.epilogue != "none":
        buffers.update({"Bias": 4*g.n, "Residual": 4*g.m*g.n})
        names += ["Bias", "Residual"]
    args = [("buf", name, 0) for name in names]
    outputs = {"C": [(0, buffers["C"])]}
    i = len(plan.jobs)
    for method, field in (("ours", "cubin"), ("ptxas", "reference")):
        module = (out_dir/t.entry[field]).read_bytes()
        steps = [(method, t.label, tuple(g.launch["grid"]), (g.launch["threads"], 1, 1),
                  g.launch["smem"], args)]
        job = _job(f"{t.entry['cubin']}:{t.label}#{method}", {method: module}, buffers, {}, {},
                   steps, outputs, reps)
        job["init_tensor16"] = initialize
        plan.jobs.append(job)
    plan.cases.append(Case(t.label, t.entry["source"], i, i+1, ["C"],
                           note=("exact small-integer smoke comparison; floating accuracy uses separate GPU tests"
                                 if g.epilogue == "none" else "fused epilogue smoke uses zero bias/residual inputs"),
                           describe=f"{g.m}x{g.n}x{g.k} {g.dtype}->f32, grid {g.launch['grid']}, "
                                    f"block {g.launch['threads']}, {g.launch['smem']} shared bytes"))


def rowwise_case(t: Target, out_dir: Path, reps: int, plan: Plan) -> None:
    """Zero-input analytic smoke plus unchanged bitwise OA/ptxas comparison."""
    from openassembler.kernels.reductions import ReductionPlan
    contract = t.kernel["rowwise"]
    try:
        allowed = ("rows", "width", "op", "epsilon", "arch")
        g = ReductionPlan(**{key: contract[key] for key in allowed if key in contract})
        if (contract.get("schema") != 1 or contract.get("launch") != g.launch
                or t.label != g.op or t.kernel["params"] != [["u64", "X"], ["u64", "Y"]]):
            raise ValueError("launch or ABI differs from its validated plan")
    except (ValueError, TypeError, KeyError) as exc:
        raise RunError(f"invalid rowwise launch contract for {t.label}: {exc}") from None
    size = 4 * g.rows * g.width
    buffers, outputs = {"X": size, "Y": size}, {"Y": [(0, size)]}
    args = [("buf", "X", 0), ("buf", "Y", 0)]
    i = len(plan.jobs)
    for method, field in (("ours", "cubin"), ("ptxas", "reference")):
        steps = [(method, t.label, tuple(g.launch["grid"]), (g.launch["threads"], 1, 1), 0, args)]
        job = _job(f"{t.entry['cubin']}:{t.label}#{method}",
                   {method: (out_dir / t.entry[field]).read_bytes()},
                   buffers, {}, {}, steps, outputs, reps)
        # 0xffffffff is an FP32 NaN. Any unwritten output fails the independent
        # finite analytic gate, including LayerNorm's expected zero output.
        job["init_byte"] = {"Y": 255}
        plan.jobs.append(job)
    expected = 1.0 / g.width if g.op == "softmax" else 0.0
    plan.cases.append(Case(t.label, t.entry["source"], i, i + 1, ["Y"],
                           note="zero-input analytic smoke; separate A100 tests cover nonzero inputs and canaries",
                           describe=f"{g.rows}x{g.width} contiguous f32 {g.op}, grid {g.launch['grid']}, block 128",
                           expected_f32_constant={"Y": expected}))


def _dim(v, what: str) -> tuple[int, int, int]:
    if type(v) is int and v > 0:
        return (v, 1, 1)
    if isinstance(v, list) and 1 <= len(v) <= 3 and all(type(x) is int and x > 0 for x in v):
        return tuple(list(v) + [1] * (3 - len(v)))  # type: ignore[return-value]
    raise RunError(f"spec: {what} must be a positive int or a list of 1-3 positive ints, got {v!r}")


def load_spec(path: Path) -> dict:
    try:
        spec = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise RunError(f"--spec {path} does not exist") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RunError(f"--spec {path} is not valid JSON ({e})") from None
    if not isinstance(spec, dict) or not isinstance(spec.get("buffers"), dict) or \
            not isinstance(spec.get("steps"), list) or not spec["steps"]:
        raise RunError(f"--spec {path}: needs \"buffers\": {{...}} and a non-empty \"steps\": [...] "
                       "(schema: openassembler run --help)")
    return spec


def spec_case(spec: dict, m: dict, out_dir: Path, reps: int, plan: Plan, label: str,
              only: set[tuple[str, str]] | None = None) -> None:
    buffers, init, init_rand, init_tensor16 = {}, {}, {}, {}
    for bi, (name, b) in enumerate(spec["buffers"].items()):
        if not isinstance(name, str) or not name or not isinstance(b, dict) or type(b.get("count")) is not int or b["count"] <= 0:
            raise RunError(f"spec: buffer {name!r} needs a positive integer \"count\"")
        dt = b.get("dtype", "f32")
        if not isinstance(dt, str) or dt not in DTYPES:
            raise RunError(f"spec: buffer {name!r} dtype {dt!r}; use one of {', '.join(DTYPES)}")
        cnt, how = b["count"], b.get("init", "random")
        buffers[name] = struct.calcsize(DTYPES[dt]) * cnt
        if how == "zeros":
            pass
        elif how == "random":
            if dt == "f32":
                init_rand[name] = (_seed("spec", name, bi), cnt)
            elif dt in ('f16', 'bf16'):
                init_tensor16[name] = (_seed('spec', name, bi), cnt, dt)
            else:
                r = random.Random(_seed("spec", name, bi))
                init[name] = struct.pack(f"<{cnt}{'i' if dt == 'i32' else 'I'}",
                                         *(r.randrange(cnt) for _ in range(cnt)))
        elif isinstance(how, list):
            if len(how) > cnt:
                raise RunError(f"spec: buffer {name!r} has {len(how)} init values but count {cnt}")
            try:
                values = []
                for value in how:
                    if dt in ('f32', 'f16', 'bf16'):
                        if type(value) not in (int, float):
                            raise ValueError(f"{dt} initialization values must be numbers")
                        values.append(float(value))
                    else:
                        lo, hi = (-2**31, 2**31) if dt == "i32" else (0, 2**32)
                        if type(value) is not int or not lo <= value < hi:
                            raise ValueError(f"{dt} initialization values must be integers in [{lo}, {hi})")
                        values.append(value)
                if dt == 'bf16':
                    from openassembler.emu.fp import BF16, F32, decode, encode
                    values = [encode(decode(_f32bits(value), F32), BF16) for value in values]
                init[name] = b"".join(struct.pack(DTYPES[dt], value) for value in values)
            except (struct.error, TypeError, ValueError, OverflowError) as e:
                raise RunError(f"spec: buffer {name!r} init values don't fit {dt}: {e}") from None
        else:
            raise RunError(f"spec: buffer {name!r} init must be \"random\", \"zeros\" or a list")

    mods_ours, mods_ref, steps_ours, steps_ref, notes = {}, {}, [], [], set()
    for si, st in enumerate(spec["steps"]):
        if not isinstance(st, dict) or not isinstance(st.get("kernel"), str):
            raise RunError(f"spec: step {si} needs \"kernel\": NAME")
        if st.get("source") is not None and not isinstance(st.get("source"), str):
            raise RunError(f"spec: step {si} source must be a file path string")
        if not isinstance(st.get("args", []), list):
            raise RunError(f"spec: step {si} args must be a list")
        try:
            (t,), w = select(m, out_dir, st["kernel"], st.get("source"), only)
        except RunError as e:
            raise RunError(f"spec step {si}: {e}") from None
        plan.warnings += [x for x in w if x not in plan.warnings]
        mod = t.entry["cubin"]
        if mod not in mods_ours:
            mods_ours[mod] = (out_dir / t.entry["cubin"]).read_bytes()
            mods_ref[mod] = (out_dir / t.entry["reference"]).read_bytes()
        if t.kernel.get("compiled_by") == "ptxas-fallback":
            notes.add(f"{t.label} is a ptxas-fallback kernel (same code in both)")
        grid, block = _dim(st.get("grid", 1), f"step {si} grid"), _dim(st.get("block", BLOCK), f"step {si} block")
        if (math.prod(block) > 1024 or any(v > lim for v, lim in zip(block, (1024, 1024, 64)))
                or any(v > lim for v, lim in zip(grid, (2147483647, 65535, 65535)))):
            raise RunError(f"spec: step {si} launch dimensions exceed CUDA limits")
        smem = st.get("smem", 0)
        if type(smem) is not int or not 0 <= smem <= 163 * 1024:
            raise RunError(f"spec: step {si} smem must be a non-negative int")
        args = []
        for a in st.get("args", []):
            if not isinstance(a, dict) or len(a) not in (1, 2):
                raise RunError(f"spec: step {si} arg {a!r}: use {{\"buf\": name}}, {{\"i32\": v}}, "
                               "{\"u32\": v}, {\"f32\": v} or {\"u64\": v}")
            if "buf" in a:
                if not isinstance(a['buf'], str) or a["buf"] not in buffers:
                    raise RunError(f"spec: step {si} uses buffer {a['buf']!r}, not in \"buffers\"")
                off = a.get("offset", 0)
                element_size = struct.calcsize(DTYPES[spec['buffers'][a['buf']].get('dtype', 'f32')])
                if (set(a) - {"buf", "offset"} or type(off) is not int
                        or not 0 <= off < buffers[a["buf"]] or off % element_size):
                    raise RunError(f"spec: step {si} buffer offset must be aligned and inside the buffer")
                args.append(("buf", a["buf"], off))
            elif "i32" in a or "u32" in a:
                kind = "i32" if "i32" in a else "u32"
                v = a[kind]
                lo, hi = (-2**31, 2**31) if kind == "i32" else (0, 2**32)
                if len(a) != 1 or type(v) is not int or not lo <= v < hi:
                    raise RunError(f"spec: step {si} invalid {kind} argument: {a!r}")
                args.append(("u32", v & 0xffffffff))
            elif "f32" in a:
                if len(a) != 1 or type(a["f32"]) not in (int, float):
                    raise RunError(f"spec: step {si} invalid f32 argument")
                args.append(("f32bits", _f32bits(float(a["f32"]))))
            elif "u64" in a:
                if len(a) != 1 or type(a["u64"]) is not int or not 0 <= a["u64"] < 2**64:
                    raise RunError(f"spec: step {si} invalid u64 argument")
                args.append(("u64", a["u64"]))
            else:
                raise RunError(f"spec: step {si} arg {a!r}: unknown kind")
        nparams = len(t.kernel.get("params", []))
        if len(args) != nparams:
            raise RunError(f"spec: step {si} passes {len(args)} args to {t.label}, which takes {nparams} "
                           f"({', '.join(p[1] for p in t.kernel['params'])})")
        steps_ours.append((mod, t.label, grid, block, smem, args))
        steps_ref.append((mod, t.label, grid, block, smem, args))
    compare = spec.get("compare", list(buffers))
    if not isinstance(compare, list) or not compare or not all(isinstance(v, str) for v in compare):
        raise RunError("spec: compare must be a non-empty list of buffer names")
    if len(set(compare)) != len(compare):
        raise RunError("spec: compare contains duplicate buffer names")
    bad = [c for c in compare if c not in buffers]
    if bad:
        raise RunError(f"spec: compare lists unknown buffers {bad}")
    outputs = {b: [(0, buffers[b])] for b in compare}
    r = spec.get("reps", reps)
    if type(r) is not int or r < 0:
        raise RunError("spec: reps must be a non-negative integer")
    i = len(plan.jobs)
    plan.jobs.append(_job(label + "#ours", mods_ours, buffers, init, init_rand, steps_ours, outputs, r))
    plan.jobs.append(_job(label + "#ptxas", mods_ref, buffers, init, init_rand, steps_ref, outputs, r))
    if init_tensor16:
        for job in plan.jobs[-2:]:
            job['init_tensor16'] = init_tensor16
    kern = " -> ".join(s[1] for s in steps_ours)
    plan.cases.append(Case(kern, label, i, i + 1, compare, note="; ".join(sorted(notes)),
                           describe=f"{len(steps_ours)} step(s), buffers {', '.join(buffers)}"))


# ---- comparison / report ------------------------------------------------------------

@dataclass
class Outcome:
    case: Case
    status: str                        # OK | MISMATCH | FAULT | SKIP | ERROR
    detail: str = ""
    t_ours: float | None = None
    t_ref: float | None = None

    @property
    def speedup(self) -> float | None:
        return self.t_ref / self.t_ours if self.t_ours and self.t_ref else None


def _first_diff(a: bytes, b: bytes) -> int | None:
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(0, n, 4):
        if a[i:i + 4] != b[i:i + 4]:
            return i // 4
    return n // 4


def compare(plan: Plan, results: dict) -> list[Outcome]:
    """results: {job index: run_plan_job result or {"error", "error_name"}} as
    openassembler.gpu.batch returns them."""
    outs = []
    for c in plan.cases:
        if c.skip:
            outs.append(Outcome(c, "SKIP", c.skip))
            continue
        ro, rr = results.get(c.ours, {"error": "no result"}), results.get(c.ref, {"error": "no result"})
        eo, er = ro.get("error"), rr.get("error")
        if eo or er:
            if eo and er:
                d = f"both ours and ptxas's build failed ({eo}); {FAULT_HINT}"
            elif eo:
                d = f"OUR cubin failed ({eo}) while ptxas's ran: a real bug in our code, please report it"
            else:
                d = f"ptxas's reference failed ({er}) while ours ran; {FAULT_HINT}"
            outs.append(Outcome(c, "FAULT", d))
            continue
        bad = []
        for b in c.compare:
            if b not in ro.get("out", {}) or b not in rr.get("out", {}):
                bad.append(f"{b}: missing output buffer")
                continue
            go = b"".join(ro.get("out", {}).get(b, []))
            gr = b"".join(rr.get("out", {}).get(b, []))
            if plan.jobs:
                expected = sum(n for _off, n in plan.jobs[c.ours]["outputs"][b])
                if len(go) != expected or len(gr) != expected:
                    bad.append(f"{b}: expected {expected} bytes, got {len(go)} / {len(gr)}")
                    continue
            i = _first_diff(go, gr)
            if i is not None:
                vo = go[4 * i:4 * i + 4].hex() or "missing"
                vr = gr[4 * i:4 * i + 4].hex() or "missing"
                bad.append(f"{b}[{i}]: ours 0x{vo} vs ptxas 0x{vr}")
            if b in c.expected_f32_constant:
                value = c.expected_f32_constant[b]
                # This adds an independent analytic smoke gate; it does not
                # replace or relax the bitwise comparison above.
                if len(go) % 4 or len(gr) % 4:
                    bad.append(f"{b}: analytic f32 smoke received an invalid byte count")
                else:
                    for method, raw in (("ours", go), ("ptxas", gr)):
                        for index, (actual,) in enumerate(struct.iter_unpack("<f", raw)):
                            if not math.isfinite(actual) or abs(actual - value) > 1e-6 + 5e-5 * abs(value):
                                bad.append(f"{b}[{index}]: {method} analytic smoke expected {value}, got {actual}")
                                break
        t_o = (ro.get("time_ms") or {}).get("median")
        t_r = (rr.get("time_ms") or {}).get("median")
        outs.append(Outcome(c, "MISMATCH" if bad else "OK", "; ".join(bad), t_o, t_r))
    return outs


def report(outs: list[Outcome], device: str = "") -> int:
    """Print the table; exit code 1 unless every case is OK."""
    if device:
        print(f"device: {device}")
    w = max([len(o.case.name) for o in outs] + [6])
    print(f"{'kernel':<{w}}  {'result':<8}  {'ours ms':>9}  {'ptxas ms':>9}  {'speedup':>7}  source")
    fmt = lambda t: f"{t:9.4f}" if t is not None else f"{'-':>9}"  # noqa: E731
    for o in outs:
        sp = f"{o.speedup:6.2f}x" if o.speedup else f"{'-':>7}"
        print(f"{o.case.name:<{w}}  {o.status:<8}  {fmt(o.t_ours)}  {fmt(o.t_ref)}  {sp}  {o.case.source}")
        if o.detail:
            print(f"{'':<{w}}    {o.detail}")
        if o.case.note:
            print(f"{'':<{w}}    note: {o.case.note}")
    ok = sum(o.status == "OK" for o in outs)
    print(f"{ok}/{len(outs)} OK (bit-exact against ptxas's build on identical inputs)"
          + ("; speedup = ptxas median / ours median" if any(o.speedup for o in outs) else ""))
    return 0 if outs and ok == len(outs) else 1


# ---- backends ------------------------------------------------------------------------

def _gpu_files() -> list[Path]:
    return [ROOT / "openassembler" / "__init__.py"] + sorted(
        p for p in GPU_PKG.glob("*.py") if p.name != "modal_app.py")


def upload_bytes(jobs_blob: bytes) -> int:
    return len(jobs_blob) + sum(p.stat().st_size for p in _gpu_files())


def price_table() -> dict:
    """Modal $/s per GPU, from tools/gpu_modal.py's PRICE_PER_S (read with ast:
    importing that file needs the modal package)."""
    import ast

    try:
        tree = ast.parse((ROOT / "tools" / "gpu_modal.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PRICE_PER_S" for t in node.targets):
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError):
        pass
    return dict(_PRICE_FALLBACK)


def estimate_seconds(jobs: list[dict]) -> float:
    """Rough GPU-container seconds: cold start + per job a worker's share, generating
    random inputs in pure Python (~0.4 us/element), and the timed launches."""
    s = 20.0
    for j in jobs:
        elems = sum(c for _seed, c in j.get("init_rand", {}).values())
        s += 0.5 + 4e-7 * elems + 0.002 * (j.get("reps", 0) + j.get("warmup", 0)) * len(j["steps"])
    return s


def describe(plan: Plan, backend: str, gpu: str, budget: float, cmd: list[str] | None) -> None:
    blob = pickle.dumps(plan.jobs)
    print(f"dry run: {len(plan.cases)} comparison(s), {len(plan.jobs)} GPU jobs on "
          f"{'this machine' if backend == 'local' else backend}; nothing was run")
    for c in plan.cases:
        if c.skip:
            print(f"  SKIP {c.name}  ({c.source}): {c.skip}")
            continue
        jo = plan.jobs[c.ours]
        print(f"  {c.name}  ({c.source}): {c.describe}; compare {', '.join(c.compare)}; reps {jo['reps']}")
    print(f"  upload: {upload_bytes(blob) / 1024:.1f} KiB (jobs + the stdlib-only openassembler/gpu package)")
    if backend == "modal":
        est = estimate_seconds(plan.jobs)
        price = price_table().get(gpu)
        if price is None:
            print(f"  estimated {est:.0f}s of {gpu} (no price on file for {gpu})")
        else:
            print(f"  estimated {est:.0f}s of {gpu} = ${est * price:.3f} "
                  f"(hard cap: --budget {budget:.0f}s = ${(budget + 60) * price:.2f})")
    if cmd:
        print("  would run: " + " ".join(shlex.quote(c) for c in cmd))


def _run_local(jobs_path: Path, res_path: Path, budget: float, job_timeout: float) -> dict:
    if platform.system() != "Linux":
        raise RunError(f"--backend local runs on THIS machine's NVIDIA GPU, which needs Linux with an NVIDIA "
                       f"driver (this is {platform.system()}). Use --backend user@host (any Linux GPU box over "
                       "SSH) or --backend modal (cloud GPU)")
    import ctypes

    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise RunError(f"--backend local: can't load libcuda.so.1 ({e}); is the NVIDIA driver installed? "
                       "Otherwise use --backend user@host or --backend modal") from None
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        subprocess.run([sys.executable, "-m", "openassembler.gpu.batch", str(jobs_path), str(res_path),
                        "--job-timeout", str(job_timeout), "--budget", str(budget)],
                       cwd=ROOT, env=env, check=True, timeout=budget + 60)
    except subprocess.TimeoutExpired:
        raise RunError(f"the local GPU batch didn't finish within --budget {budget:.0f}s + 60s") from None
    except subprocess.CalledProcessError as e:
        raise RunError(f"the local GPU batch failed (exit {e.returncode})") from None
    info = subprocess.run([sys.executable, "-c", "from openassembler.gpu.batch import device_info as d; "
                           "i = d(); print(i['name'], 'cc', i['cc'])"],
                          capture_output=True, text=True, env=env, cwd=ROOT).stdout.strip()
    return {**pickle.loads(res_path.read_bytes()), "device": info}


def _ssh_cmds(host: str, budget: float, job_timeout: float) -> list[str]:
    return [f"ssh {host} mktemp -d /tmp/openassembler_run_XXXXXX",
            f"scp openassembler/__init__.py openassembler/gpu/*.py jobs.pkl {host}:<dir>",
            f"ssh {host} 'cd <dir> && timeout {int(budget + 60)} python3 -m openassembler.gpu.batch jobs.pkl "
            f"res.pkl --job-timeout {job_timeout:g} --budget {budget:g}'",
            f"scp {host}:<dir>/res.pkl . ; ssh {host} rm -rf <dir>"]


def _run_ssh(host: str, jobs_path: Path, res_path: Path, budget: float, job_timeout: float) -> dict:
    if not shutil.which("ssh") or not shutil.which("scp"):
        raise RunError("--backend user@host needs ssh and scp on this machine")
    q = shlex.quote
    from openassembler.transport import ssh_destination
    destination = ssh_destination(host)
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]

    def sh(cmd: list[str], timeout: float = 120, capture: bool = False) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, check=True, timeout=timeout, text=True, capture_output=capture)
        except subprocess.TimeoutExpired:
            raise RunError(f"timed out: {' '.join(cmd[:3])} ...") from None
        except subprocess.CalledProcessError as e:
            raise RunError(f"{cmd[0]} to {host} failed (exit {e.returncode}): "
                           f"{(e.stderr or '').strip().splitlines()[-1:] or ''}") from None

    rdir = sh(["ssh", *opts, destination, "mktemp -d /tmp/openassembler_run_XXXXXX"], capture=True).stdout.strip()
    import re
    if not re.fullmatch(r'/tmp/openassembler_run_[A-Za-z0-9]+', rdir):
        raise RunError(f"unexpected answer from {host} creating a temp dir: {rdir!r}")
    try:
        sh(["ssh", *opts, destination, f"mkdir -p {q(rdir)}/openassembler/gpu"])
        sh(["scp", "-q", *opts, str(ROOT / "openassembler" / "__init__.py"), f"{host}:{rdir}/openassembler/"])
        sh(["scp", "-q", *opts, *[str(p) for p in _gpu_files()[1:]], f"{host}:{rdir}/openassembler/gpu/"])
        sh(["scp", "-q", *opts, str(jobs_path), f"{host}:{rdir}/jobs.pkl"])
        sh(["ssh", *opts, destination, f"cd {q(rdir)} && timeout {int(budget + 60)} python3 -m openassembler.gpu.batch "
            f"jobs.pkl res.pkl --job-timeout {job_timeout:g} --budget {budget:g}"], timeout=budget + 120)
        sh(["scp", "-q", *opts, f"{host}:{rdir}/res.pkl", str(res_path)])
        info = subprocess.run(["ssh", *opts, destination, f"cd {q(rdir)} && python3 -c 'from openassembler.gpu.batch "
                               "import device_info as d; i = d(); print(i[\"name\"], \"cc\", i[\"cc\"])'"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    finally:
        try:
            subprocess.run(["ssh", *opts, destination, f"rm -rf -- {q(rdir)}"], timeout=30, capture_output=True)
        except (OSError, subprocess.SubprocessError):
            pass
    return {**pickle.loads(res_path.read_bytes()), "device": info}


def modal_cmd(jobs_path: Path, res_path: Path, gpu: str, budget: float, job_timeout: float = 60) -> list[str]:
    return ["modal", "run", str(MODAL_APP), "--jobs", str(jobs_path), "--out", str(res_path),
            "--gpu", gpu, "--budget", f"{budget:g}", "--job-timeout", f"{job_timeout:g}"]


def modal_ready() -> str | None:
    """None if the modal CLI is installed and has credentials, else what to do."""
    if not shutil.which("modal"):
        return ("--backend modal needs the Modal CLI: pip install modal && modal setup "
                "(any Python env; openassembler only runs the `modal` command)")
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")) and \
            not (Path.home() / ".modal.toml").exists():
        return "--backend modal: not logged in to Modal; run `modal setup` (or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET)"
    return None


def _run_modal(jobs_path: Path, res_path: Path, gpu: str, budget: float, job_timeout: float = 60) -> dict:
    if (why := modal_ready()):
        raise RunError(why)
    cmd = modal_cmd(jobs_path, res_path, gpu, budget, job_timeout)
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    try:
        r = subprocess.run(cmd, cwd=ROOT, timeout=budget + 300)
    except subprocess.TimeoutExpired:
        raise RunError("modal run didn't return in time; check `modal app list` and stop any "
                       "openassembler-run app that is still running") from None
    if r.returncode != 0 or not res_path.exists():
        raise RunError(f"modal run failed (exit {r.returncode}); see its output above")
    res = pickle.loads(res_path.read_bytes())
    dev = res.get("device") or {}
    return {**res, "device": f"{dev.get('name', '?')} cc {dev.get('cc', '?')} (Modal {gpu})"}


# ---- entry point ------------------------------------------------------------------------

@dataclass
class RunOptions:
    out_dir: Path = Path("output")
    backend: str = "local"
    kernel: str | None = None
    source: str | None = None
    spec: Path | None = None
    n: int = DEFAULT_N
    reps: int = 20
    gpu: str = "A100-40GB"
    budget: float = 300
    job_timeout: float = 60
    dry_run: bool = False
    save_jobs: Path | None = None
    from_results: Path | None = None
    only: set[tuple[str, str]] | None = None     # cnr: (abs source, kernel) compiled this run


def validate_options(o: RunOptions, *, check_spec: bool = False) -> None:
    """Validate before a caller starts a potentially expensive build."""
    _check_backend(o.backend)
    if not math.isfinite(o.budget) or not math.isfinite(o.job_timeout) or o.budget <= 0 or o.job_timeout <= 0:
        raise RunError("--budget and --job-timeout must be finite positive seconds")
    if o.backend == "modal" and o.budget > 1800:
        raise RunError("Modal --budget must not exceed 1800 seconds")
    if o.n <= 0 or o.reps < 0:
        raise RunError("-n must be > 0 and --reps >= 0")
    if o.spec is not None and check_spec:
        if o.kernel:
            raise RunError("--spec names its kernels in its steps; drop --kernel (or drop --spec)")
        load_spec(o.spec)


def build_plan(o: RunOptions) -> tuple[Plan, dict]:
    validate_options(o)
    out_dir = Path(o.out_dir)
    m, warns = load(out_dir)
    plan = Plan(warnings=warns)
    if o.spec is not None:
        if o.kernel:
            raise RunError("--spec names its kernels in its steps; drop --kernel (or drop --spec)")
        spec = load_spec(o.spec)
        spec_case(spec, m, out_dir, o.reps, plan, Path(o.spec).name, o.only)
        return plan, m
    targets, w = select(m, out_dir, o.kernel, o.source, o.only)
    plan.warnings += w
    for t in targets:
        auto_case(t, out_dir, o.n, o.reps, plan)
    return plan, m


def execute(plan: Plan, o: RunOptions) -> tuple[dict, str]:
    """Run plan.jobs on the chosen backend -> ({job index: result}, device line)."""
    if o.from_results is not None:
        res = pickle.loads(Path(o.from_results).read_bytes())
        return res["results"], str(res.get("device", ""))
    if not plan.jobs:
        return {}, ""
    with tempfile.TemporaryDirectory(prefix="oa_run_") as d:
        jp, rp = Path(d) / "jobs.pkl", Path(d) / "res.pkl"
        jp.write_bytes(pickle.dumps(plan.jobs))
        if o.backend == "local":
            res = _run_local(jp, rp, o.budget, o.job_timeout)
        elif o.backend == "modal":
            res = _run_modal(jp, rp, o.gpu, o.budget, o.job_timeout)
        else:
            res = _run_ssh(o.backend, jp, rp, o.budget, o.job_timeout)
    return res["results"], str(res.get("device", ""))


def _check_backend(b: str) -> None:
    if b in ("local", "modal"):
        return
    from openassembler.transport import TransportError, ssh_host
    try:
        ssh_host(b)
    except TransportError as exc:
        raise RunError(str(exc)) from None


def run_command(o: RunOptions) -> int:
    try:
        plan, _m = build_plan(o)
        for w in plan.warnings:
            print(f"warning: {w}", file=sys.stderr)
        if o.save_jobs:
            P.write_atomic(Path(o.save_jobs), pickle.dumps(plan.jobs))
            print(f"wrote {len(plan.jobs)} jobs to {o.save_jobs}")
        if o.dry_run:
            if o.backend == "modal":
                cmd = modal_cmd(Path("<jobs.pkl>"), Path("<res.pkl>"), o.gpu, o.budget, o.job_timeout)
                why = modal_ready()
                if why:
                    print(f"note: {why}")
            elif o.backend == "local":
                cmd = [sys.executable, "-m", "openassembler.gpu.batch", "<jobs.pkl>", "<res.pkl>",
                       "--job-timeout", f"{o.job_timeout:g}", "--budget", f"{o.budget:g}"]
            else:
                cmd = None
            describe(plan, o.backend, o.gpu, o.budget, cmd)
            if o.backend not in ("local", "modal"):
                for c in _ssh_cmds(o.backend, o.budget, o.job_timeout):
                    print(f"    {c}")
            return 0
        results, device = execute(plan, o)
    except (RunError, P.ProjectError, OSError, ValueError, TypeError, OverflowError, struct.error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    outs = compare(plan, results)
    rc = report(outs, device)
    rep = {"openassembler": __version__, "backend": o.backend, "device": device,
           "cases": [{"kernel": x.case.name, "source": x.case.source, "result": x.status, "detail": x.detail,
                      "ours_ms": x.t_ours, "ptxas_ms": x.t_ref, "speedup": x.speedup,
                      "note": x.case.note, "launch_description": x.case.describe,
                      **({"analytic_smoke": {"expected_f32_constant": x.case.expected_f32_constant,
                                             "rtol": 5e-5, "atol": 1e-6, "require_finite": True}}
                         if x.case.expected_f32_constant else {})} for x in outs]}
    try:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        rep["timestamp"] = stamp
        text = json.dumps(rep, indent=1) + "\n"
        P.write_atomic(Path(o.out_dir) / "runs" / f"{stamp}.json", text)
        P.write_atomic(Path(o.out_dir) / "run.json", text)
    except OSError:
        pass
    return rc
