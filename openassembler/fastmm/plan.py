"""Turn a bilinear scheme into a GPU launch plan (one level of block recursion).

C (N x N) = A B with A, B split into n x m and m x p grids of blocks of size bs:

  for r in 0..R-1:
      S_r = sum_ij u[ij, r] A_ij             lincomb kernel   (skipped if one term with coef 1)
      T_r = sum_jk v[jk, r] B_jk             lincomb kernel   (same)
      M_r = S_r T_r                          sgemm kernel on bs x bs blocks
  C_ik  = sum_r w[ik, r] M_r                 lincomb kernel, chunked to <= MAXIN inputs

Views are (buffer, element offset, leading dimension), so single-term operands read A/B/C
directly. Coefficients are exact in f32 (integers / dyadic fractions). The naive scheme
naive(n, m, p) gives the same plan structure with R = n m p and no linear combinations,
which is the fair "same kernels, cubic" baseline; one big sgemm launch is the other.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from openassembler.fastmm.factors import Scheme
from openassembler.fastmm.kernels import THREADS

MAXIN = 8


def f32bits(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclass
class Plan:
    steps: list = field(default_factory=list)       # (module, kernel, grid, block, smem, args)
    buffers: dict = field(default_factory=dict)     # name -> bytes
    lincomb_ns: set = field(default_factory=set)
    flops_mul: int = 0
    n_lincomb: int = 0


def _lincomb(plan: Plan, out, terms, rows, cols, accumulate=False):
    """out = sum c * in (or out += ... if accumulate) over a rows x cols view."""
    if accumulate:
        terms = [(out, 1)] + list(terms)
    while terms:
        chunk, terms = terms[:MAXIN], terms[MAXIN:]
        if terms:
            # more to come: write the partial sum, then fold it in as the first term
            terms = [(out, 1)] + terms
        k = len(chunk)
        plan.lincomb_ns.add(k)
        ob, oo, old = out
        args = [("buf", ob, 4 * oo), ("u32", old)]
        for (b, o, ld), c in chunk:
            args += [("buf", b, 4 * o), ("u32", ld), ("f32bits", f32bits(c))]
        args.append(("u32", cols))
        grid = ((cols + 1023) // 1024, rows, 1)
        plan.steps.append((f"lincomb{k}", f"lincomb{k}", grid, (256, 1, 1), 0, args))
        plan.n_lincomb += 1


GEMM = "sgemm128"           # block GEMM kernel: "sgemm64" | "sgemm128" (tile edge in the name)


def _gemm(plan: Plan, a, b, c, M, N, K):
    (ab, ao, alda), (bb, bo, bld), (cb, co, cld) = a, b, c
    args = [("buf", ab, 4 * ao), ("buf", bb, 4 * bo), ("buf", cb, 4 * co),
            ("u32", alda), ("u32", bld), ("u32", cld), ("u32", K)]
    t = int(GEMM[5:])
    if M % t or N % t:
        raise ValueError(f"{M} x {N} block not a multiple of the {t} x {t} tile")
    plan.steps.append((GEMM, GEMM, (N // t, M // t, 1), (THREADS, 1, 1), 0, args))
    plan.flops_mul += M * N * K


def cublas(N: int) -> Plan:
    """Reference: one cublasSgemm (FP32 CUDA cores, default math mode, no TF32)."""
    p = Plan(buffers={"A": 4 * N * N, "B": 4 * N * N, "C": 4 * N * N})
    p.steps.append(("@cublas", "sgemm", None, None, 0,
                    [("buf", "A", 0), ("buf", "B", 0), ("buf", "C", 0), ("u32", N), ("u32", N), ("u32", N),
                     ("u32", N), ("u32", N), ("u32", N)]))
    p.flops_mul = N ** 3
    return p


def direct(N: int) -> Plan:
    p = Plan(buffers={"A": 4 * N * N, "B": 4 * N * N, "C": 4 * N * N})
    _gemm(p, ("A", 0, N), ("B", 0, N), ("C", 0, N), N, N, N)
    return p


def build(s: Scheme, N: int) -> Plan:
    """One recursion level of scheme s on N x N matrices (N divisible by 64 n, 64 m, 64 p)."""
    if N % s.n or N % s.m or N % s.p:
        raise ValueError(f"N={N} not divisible by {s.n},{s.m},{s.p}")
    bn, bm, bp = N // s.n, N // s.m, N // s.p
    if bm % 8:
        raise ValueError("inner block dimension must be a multiple of 8")
    plan = Plan(buffers={"A": 4 * N * N, "B": 4 * N * N, "C": 4 * N * N,
                         "S": 4 * bn * bm, "T": 4 * bm * bp})
    R = s.rank
    Ablk = lambda i, j: ("A", i * bn * N + j * bm, N)    # noqa: E731
    Bblk = lambda j, k: ("B", j * bm * N + k * bp, N)    # noqa: E731
    Cblk = lambda i, k: ("C", i * bn * N + k * bp, N)    # noqa: E731
    for r in range(R):
        plan.buffers[f"M{r}"] = 4 * bn * bp
    for r in range(R):
        ua = [((Ablk(ij // s.m, ij % s.m)), s.u[ij][r]) for ij in range(s.n * s.m) if s.u[ij][r]]
        vb = [((Bblk(jk // s.p, jk % s.p)), s.v[jk][r]) for jk in range(s.m * s.p) if s.v[jk][r]]
        if len(ua) == 1 and ua[0][1] == 1:
            Sv = ua[0][0]
        else:
            Sv = ("S", 0, bm)
            _lincomb(plan, Sv, ua, bn, bm)
        if len(vb) == 1 and vb[0][1] == 1:
            Tv = vb[0][0]
        else:
            Tv = ("T", 0, bp)
            _lincomb(plan, Tv, vb, bm, bp)
        _gemm(plan, Sv, Tv, (f"M{r}", 0, bp), bn, bp, bm)
    for ik in range(s.n * s.p):
        terms = [((f"M{r}", 0, bp), s.w[ik][r]) for r in range(R) if s.w[ik][r]]
        _lincomb(plan, Cblk(ik // s.p, ik % s.p), terms, bn, bp)
    return plan


FUSE_MAX = 4      # fgemm kernels exist for up to this many A / B / C terms (register budget)


def build_fused(s: Scheme, N: int) -> Plan:
    """Like build(), but each rank-1 step whose A, B and C term counts are <= FUSE_MAX is
    one fgemm launch (openassembler.fastmm.kernels.fgemm): the linear combinations happen while
    tiles are loaded and C_o += w_o M_r in the epilogue. Other steps use the unfused path
    (lincomb -> sgemm128 -> accumulate). C is zeroed first."""
    if N % s.n or N % s.m or N % s.p:
        raise ValueError(f"N={N} not divisible by {s.n},{s.m},{s.p}")
    bn, bm, bp = N // s.n, N // s.m, N // s.p
    t = int(GEMM[5:])
    if bn % t or bp % t or bm % 8:
        raise ValueError("block sizes vs tile")
    plan = Plan(buffers={"A": 4 * N * N, "B": 4 * N * N, "C": 4 * N * N,
                         "S": 4 * bn * bm, "T": 4 * bm * bp, "M": 4 * bn * bp})
    plan.fgemm_shapes = set()
    plan.steps.append(("@memset", "", None, None, 0, [("buf", "C", 0), ("u32", 4 * N * N)]))
    Ablk = lambda ij: ("A", (ij // s.m) * bn * N + (ij % s.m) * bm, N)    # noqa: E731
    Bblk = lambda jk: ("B", (jk // s.p) * bm * N + (jk % s.p) * bp, N)    # noqa: E731
    Cblk = lambda ik: ("C", (ik // s.p) * bn * N + (ik % s.p) * bp, N)    # noqa: E731
    for r in range(s.rank):
        ua = [(Ablk(ij), s.u[ij][r]) for ij in range(s.n * s.m) if s.u[ij][r]]
        vb = [(Bblk(jk), s.v[jk][r]) for jk in range(s.m * s.p) if s.v[jk][r]]
        wc = [(Cblk(ik), s.w[ik][r]) for ik in range(s.n * s.p) if s.w[ik][r]]
        if max(len(ua), len(vb), len(wc)) <= FUSE_MAX:
            ka, kb, kc = len(ua), len(vb), len(wc)
            plan.fgemm_shapes.add((ka, kb, kc))
            args = [("buf", b, 4 * o) for (b, o, _), _c in ua] + [("f32bits", f32bits(c)) for _v, c in ua]
            args += [("buf", b, 4 * o) for (b, o, _), _c in vb] + [("f32bits", f32bits(c)) for _v, c in vb]
            args += [("buf", b, 4 * o) for (b, o, _), _c in wc] + [("f32bits", f32bits(c)) for _v, c in wc]
            args += [("u32", N), ("u32", N), ("u32", N), ("u32", bm)]
            k = f"fgemm_{ka}_{kb}_{kc}"
            plan.steps.append((k, k, (bp // t, bn // t, 1), (THREADS, 1, 1), 0, args))
            plan.flops_mul += bn * bp * bm
            continue
        Sv = ua[0][0] if len(ua) == 1 and ua[0][1] == 1 else ("S", 0, bm)
        if Sv[0] == "S":
            _lincomb(plan, Sv, ua, bn, bm)
        Tv = vb[0][0] if len(vb) == 1 and vb[0][1] == 1 else ("T", 0, bp)
        if Tv[0] == "T":
            _lincomb(plan, Tv, vb, bm, bp)
        _gemm(plan, Sv, Tv, ("M", 0, bp), bn, bp, bm)
        for cv, c in wc:
            _lincomb(plan, cv, [(("M", 0, bp), c)], bn, bp, accumulate=True)
    return plan
