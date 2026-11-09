"""Build benchmark cubins: ptxas vs our compiler (safe and mined schedulers).

    PYTHONPATH=. uv run --no-project python tools/bench_build.py [--arch sm_80]

Writes corpus/<arch>/bench/<variant>@<kernel>.cubin for variants
  ptxas   ptxas -O3
  safe    our compiler, stall 6 after every fixed-latency op
  mined   our compiler, stalls from the mined ptxas latency model
Each of ours is checked in the emulator against the PTX interpreter (random inputs,
poisoned registers) before it is written; failures are reported and skipped.
"""

from __future__ import annotations

import argparse
import random
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import bench_kernels  # noqa: E402
import diff_emu  # noqa: E402
import ptx_probes  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import ra  # noqa: E402
from openassembler.backend.compile import Out, schedule  # noqa: E402
from openassembler.container.cubin import required_regcount, splice_text  # noqa: E402
from openassembler.emu.probe import run_probe  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402


VARIANTS = ("safe", "mined", "sched", "opt", "optY", "optN")
#   safe   stall 6 everywhere, source order
#   mined  A100-measured latencies (compile.hw_latency), source order
#   sched  + basic-block list scheduling (openassembler/backend/sched.py)
#   opt    + 4x unrolling of counted loops (openassembler/backend/unroll.py)


def listing(code, labels):
    """Emulator-readable listing for our code (label lines before their instruction)."""
    inv = {}
    for name, i in labels.items():
        inv.setdefault(i, []).append(name)
    lines = []
    for i, (t, _c) in enumerate(code):
        for name in inv.get(i, []):
            lines.append(f"{name}:")
        lines.append(f"        /*{16 * i:04x}*/ {t} ; /* 0x{0:016x} */\n        /* 0x{0:016x} */")
    return "\n".join(lines)


def check(name, kname, ptx, code, labels, rng, trials=3, k=37):
    sass = listing(code, labels)
    for _ in range(trials):
        inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
        kf = diff_emu.rand32(rng)
        got = run_probe(sass, inputs, k, kf)
        if run_probe(sass, inputs, k, kf, poison=0xDEADBEEF) != got:
            return "reads uninitialized register"
        exp = diff_emu.run_ptx(ptx, kname, inputs, k, kf)
        for t, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
            if not undef and not diff_emu.same(name, g, e, approx):
                return f"lane {t}: ours {g} vs PTX {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_80")
    a = ap.parse_args()
    out = ROOT / "corpus" / a.arch / "bench"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    for name, body in bench_kernels.BENCH.items():
        kname = ptx_probes.kernel_name(name)
        (out / f"{kname}.ptx").write_text(
            ptx_probes.module(a.arch, name, body, name in bench_kernels.SHARED))
    oracle.run_tool("bash", "-c", f"cd {shlex.quote(str(out))} && for f in *.ptx; do "
                    f"ptxas -arch={a.arch} -O3 -o ptxas@${{f%.ptx}}.cubin $f && "
                    f"nvdisasm -hex -c ptxas@${{f%.ptx}}.cubin > ptxas@${{f%.ptx}}.sass; done")
    enc = Encoder("sm_80")
    rng = random.Random(7)
    for name in bench_kernels.BENCH:
        kname = ptx_probes.kernel_name(name)
        ptx = (out / f"{kname}.ptx").read_text()
        base = (out / f"ptxas@{kname}.cubin").read_bytes()
        try:
            k = parse_module(ptx).kernels[kname]
            code_safe, st = ra.compile_kernel(k)
        except Exception as e:  # noqa: BLE001
            print(f"{name:18} ours: NOT SUPPORTED ({str(e)[:70]})")
            continue
        sizes = {}
        n_ptxas = sum(1 for i in oracle.parse_listing((out / f"ptxas@{kname}.sass").read_text())
                      if i.text != "NOP" and not i.text.startswith("BRA `(.L_x_"))
        for variant in VARIANTS:
            try:
                code, labels = (code_safe, st["labels"]) if variant == "safe" else _reschedule_k(k, variant)
            except Exception as e:  # noqa: BLE001
                print(f"{name:18} {variant}: NOT SUPPORTED ({str(e)[:70]})")
                continue
            err = check(name, kname, ptx, code, labels, rng)
            if err:
                print(f"{name:18} {variant}: EMULATOR CHECK FAILED: {err}")
                continue
            words = ra.encode_code(enc, code, labels)
            nop = enc.encode("NOP")
            words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
            while len(words) % 8:
                words.append(nop)
            exits = [16 * i for i, (t, _) in enumerate(code) if t.endswith("EXIT")]
            nregs = required_regcount([t for t, _ in code])
            (out / f"{variant}@{kname}.cubin").write_bytes(splice_text(base, kname, words, nregs, exits))
            sizes[variant] = len(code)
        print(f"{name:18} ptxas {n_ptxas} insns | ours " + " ".join(f"{v} {n}" for v, n in sizes.items()))


def _reschedule_k(k, model):
    """Compile PTX kernel k with scheduler `model`; returns (code, labels).
    Suffix Y / N: force the yield bit on / off on every instruction (experiment)."""
    import dataclasses
    if model[-1] in "YN" and model[:-1] in VARIANTS:
        code, labels = _reschedule_k(k, model[:-1])
        y = model[-1] == "Y"
        return [(t, dataclasses.replace(c, yield_=y or c.stall > 11)) for t, c in code], labels
    from openassembler.backend.sched import list_schedule
    from openassembler.backend.unroll import unroll_loops

    if model == "opt":
        k = unroll_loops(k)
    s = ra.Selector(k)
    ops = ra.coalesce(ra.dce(s.run()))
    phys = ra.allocate(ops, s.r1)
    lowered = ra.lower(ops, phys)
    if model in ("sched", "opt"):
        lowered = list_schedule(lowered)
        model = "mined"
    labels, body = {}, []
    for t, vl, r, w in lowered:
        if t.startswith("<label "):
            labels[t[7:-1]] = len(body)
        else:
            body.append((t, vl, r, w))
    return schedule([Out(t, vl, r, w) for t, vl, r, w in body], labels=labels, model=model), labels


if __name__ == "__main__":
    main()
