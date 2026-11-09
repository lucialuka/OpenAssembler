"""Fit the tensor-core accumulation model to recorded A100 outputs (offline, $0).

    PYTHONPATH=. uv run --no-project python tools/fit_mma.py corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz

The mma probes (tools/ptx_probes.py) compute, per thread,
    A = {r1, r2, r3, r4}, B = {r1, r2}, C = {f1, f2, f3, f1}   (f = bit-cast of r)
and store D elements 0 and 3. So from the recorded inputs we rebuild the full
A, B, C matrices with the PTX fragment layouts and compare candidate models of
the hardware's dot-product unit against the GPU's bits, without the emulator.

Model family (Fasi et al. 2021 style): products are exact; per block of `block`
products the terms and the running accumulator are aligned to the largest
exponent, each truncated to `bits` bits below it, summed exactly, then rounded
to f32 with `round`. tf32 inputs are reduced from f32 by `tf32_in`.
"""

from __future__ import annotations

import gzip
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openassembler.emu import fp  # noqa: E402
from openassembler.emu.tensor import (a_rc_k8_tf32, a_rc_k16, aligned_sum, b_rc_k8_tf32,  # noqa: E402
                               b_rc_k16, c_rc)


def matrices(probe: str, inputs, tf32_in: str):
    """Rebuild A, B, C (values) from one trial's per-thread inputs."""
    if probe == "mma.tf32":
        M, N, K, arc, brc = 16, 8, 8, a_rc_k8_tf32, b_rc_k8_tf32
        def to_tf32(bits):
            if tf32_in == "trunc":
                return fp.decode(bits & 0xFFFFE000)
            return fp.decode(fp.encode(fp.decode(bits), fp.Fmt(8, 10), "RN") << 13)
        conv_a = lambda regs: [to_tf32(r) for r in regs]  # noqa: E731
        conv_b = conv_a
    else:
        elt = fp.BF16 if probe == "mma.bf16" else fp.F16
        M, N, K, arc, brc = 16, 8, 16, a_rc_k16, b_rc_k16
        def halves(regs):
            out = []
            for r in regs:
                out += [fp.decode(r & 0xFFFF, elt), fp.decode(r >> 16, elt)]
            return out
        conv_a = conv_b = halves
    A = [[None] * K for _ in range(M)]
    B = [[None] * N for _ in range(K)]
    C = {}
    for lane, (r1, r2, r3, r4) in enumerate(inputs):
        for i, v in enumerate(conv_a([r1, r2, r3, r4])):
            r, c = arc(lane, i)
            A[r][c] = v
        for i, v in enumerate(conv_b([r1, r2])):
            r, c = brc(lane, i)
            B[r][c] = v
        for i, bits in enumerate([r1, r2, r3, r1]):
            C[c_rc(lane, i)] = fp.decode(bits)
    return A, B, C, K


def model_dot(prods, c, block, bits, rnd, trunc="rz", cmode="joint"):
    """cmode="joint": c is one more term in the first block's alignment;
    cmode="after": each block's products are aligned/summed alone, then added to the
    running accumulator with a second aligned add."""
    if any(v.nan or v.inf for v in prods + [c]):
        return None  # specials excluded from the fit
    acc = fp.signed_frac(c)
    for i in range(0, len(prods), block):
        terms = [fp.signed_frac(p) for p in prods[i:i + block]]
        if cmode == "joint":
            s = aligned_sum(terms + [acc], bits, trunc)
        else:
            s = aligned_sum([aligned_sum(terms, bits, trunc), acc], bits, trunc)
        acc = fp.signed_frac(fp.decode(fp.encode(fp.from_frac(s), fp.F32, rnd)))
    return fp.encode(fp.from_frac(acc), fp.F32, "RN")


def main(raw_path: str):
    raw = json.load(gzip.open(raw_path, "rt"))["probes"]
    for probe in ("mma.f16", "mma.bf16", "mma.tf32"):
        if probe not in raw:
            continue
        rec = raw[probe]
        tf_modes = ("trunc", "rn") if probe == "mma.tf32" else ("-",)
        results = []
        for tf32_in in tf_modes:
            cases = []  # (prods, c, gpu_bits)
            for (inputs, _k, _kf), gpu in zip(rec["trials"], rec["out"]):
                A, B, C, K = matrices(probe, inputs, tf32_in)
                for lane in range(32):
                    for slot, idx in ((0, 0), (1, 3)):
                        r, c = c_rc(lane, idx)
                        prods = [fp.mul(A[r][k], B[k][c]) for k in range(K)]
                        cases.append((prods, C[(r, c)], gpu[lane][slot]))
            K = len(cases[0][0])
            blocks = [b for b in (4, 8, 16) if b <= K]
            for block, bits, rnd, trunc, cmode in itertools.product(
                    blocks, range(23, 28), ("RZ", "RN", "RM"), ("rz", "floor"), ("joint", "after")):
                hit = n = 0
                for prods, c, g in cases:
                    m = model_dot(prods, c, block, bits, rnd, trunc, cmode)
                    if m is None:
                        continue
                    n += 1
                    hit += m == g
                results.append((hit / max(1, n), hit, n, block, bits, rnd, trunc, cmode, tf32_in))
        results.sort(reverse=True)
        print(f"{probe}: best models")
        for acc, hit, n, block, bits, rnd, trunc, cmode, tf in results[:5]:
            print(f"   {acc:7.2%} {hit}/{n} block={block:2} bits={bits} round={rnd} trunc={trunc} "
                  f"c={cmode} tf32_in={tf}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz"))
