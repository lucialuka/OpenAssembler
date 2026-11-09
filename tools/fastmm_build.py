"""Build the fast-matmul kernels with ptxas and with our compiler (PTX -> SASS -> cubin).

    PYTHONPATH=. python3 tools/fastmm_build.py [--arch sm_80] [--variant opt]

Writes corpus/<arch>/fastmm/{ptxas,ours}@<kernel>.cubin for sgemm64 and lincomb1..8.
Ours = openassembler.backend (no ptxas involved in code generation; ptxas's cubin is only used
as the ELF container template, as for every other kernel we build). Each of ours is
checked in the emulator against a Python reference before it is written:
  sgemm64   64 x 64 x 16, small-integer inputs (exact in f32), 256 threads, poisoned regs
  lincombK  one 4 x 1024 block, random inputs vs f32 evaluation in the same order
"""

from __future__ import annotations

import argparse
import random
import shlex
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from bench_build import _reschedule_k, listing  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import ra  # noqa: E402
from openassembler.container.cubin import required_regcount, splice_text  # noqa: E402
from openassembler.emu.machine import Launch, Machine, Memory  # noqa: E402
from openassembler.emu.program import load_listing  # noqa: E402
from openassembler.fastmm import kernels as K  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402

PARAM_BASE = {"sm_80": 0x160, "sm_86": 0x160, "sm_89": 0x160}
LINCOMB_MAX = 8


def f2b(x):
    return struct.pack("<f", x)


def b2f(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def rf32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def check_sgemm(sass, arch, M=64):
    N = M
    Kd = 16
    A0, B0, C0 = 0x10000000, 0x20000000, 0x30000000
    rng = random.Random(1)
    A = [[float(rng.randint(-3, 3)) for _ in range(Kd)] for _ in range(M)]
    B = [[float(rng.randint(-3, 3)) for _ in range(N)] for _ in range(Kd)]
    g = Memory()
    g.write_bytes(A0, b"".join(f2b(x) for r in A for x in r))
    g.write_bytes(B0, b"".join(f2b(x) for r in B for x in r))
    params = struct.pack("<QQQIIII", A0, B0, C0, Kd, N, N, Kd)
    L = Launch(block=(256, 1, 1), params=params, param_base=PARAM_BASE[arch], smem_static=16384,
               poison=0xDEADBEEF)
    Machine(load_listing(sass), L, g).run(max_steps=10 ** 7)
    bad = 0
    for i in range(M):
        for j in range(N):
            ref = sum(A[i][k] * B[k][j] for k in range(Kd))
            bad += b2f(g.read(C0 + 4 * (i * N + j), 4)) != ref
    return None if not bad else f"{bad}/{M * N} wrong"


def check_fgemm(sass, arch, ka, kb, kc):
    """128 x 128 x 16 step with small-integer data and coefficients (all exact in f32):
    C_o (pre-filled) += w_o (sum u_s A_s)(sum v_t B_t)."""
    M = N = 128
    Kd = 16
    rng = random.Random(100 * ka + 10 * kb + kc)
    g = Memory()
    As, Bs, Cs = [], [], []
    pa, pb, pc = [], [], []
    for s in range(ka):
        X = [[float(rng.randint(-2, 2)) for _ in range(Kd)] for _ in range(M)]
        base = 0x10000000 + s * 0x01000000
        g.write_bytes(base, b"".join(f2b(x) for r in X for x in r))
        As.append(X)
        pa.append(base)
    for t in range(kb):
        X = [[float(rng.randint(-2, 2)) for _ in range(N)] for _ in range(Kd)]
        base = 0x20000000 + t * 0x01000000
        g.write_bytes(base, b"".join(f2b(x) for r in X for x in r))
        Bs.append(X)
        pb.append(base)
    for o in range(kc):
        X = [[float(rng.randint(-8, 8)) for _ in range(N)] for _ in range(M)]
        base = 0x30000000 + o * 0x01000000
        g.write_bytes(base, b"".join(f2b(x) for r in X for x in r))
        Cs.append(X)
        pc.append(base)
    ca = [rng.choice([1.0, -1.0, 2.0]) for _ in range(ka)]
    cb = [rng.choice([1.0, -1.0, 0.5]) for _ in range(kb)]
    cc = [rng.choice([1.0, -1.0]) for _ in range(kc)]
    params = struct.pack(f"<{ka}Q{ka}f", *pa, *ca)
    params += b"\0" * (-len(params) % 8) + struct.pack(f"<{kb}Q", *pb) + struct.pack(f"<{kb}f", *cb)
    params += b"\0" * (-len(params) % 8) + struct.pack(f"<{kc}Q", *pc) + struct.pack(f"<{kc}f", *cc)
    params += struct.pack("<IIII", Kd, N, N, Kd)
    L = Launch(block=(256, 1, 1), params=params, param_base=PARAM_BASE[arch], smem_static=16384,
               poison=0xDEADBEEF)
    Machine(load_listing(sass), L, g).run(max_steps=10 ** 7)
    SA = [[sum(ca[s] * As[s][i][k] for s in range(ka)) for k in range(Kd)] for i in range(M)]
    SB = [[sum(cb[t] * Bs[t][k][j] for t in range(kb)) for j in range(N)] for k in range(Kd)]
    bad = 0
    for i in range(0, M, 3):
        for j in range(N):
            m = sum(SA[i][k] * SB[k][j] for k in range(Kd))
            for o in range(kc):
                bad += b2f(g.read(pc[o] + 4 * (i * N + j), 4)) != Cs[o][i][j] + cc[o] * m
    return None if not bad else f"{bad} wrong"


def check_lincomb(sass, n_in, arch):
    rows, cols, ld = 2, 1024, 1032
    rng = random.Random(n_in)
    O0 = 0x40000000
    g = Memory()
    params = struct.pack("<QI", O0, ld)
    pad = b"\0" * 4
    ins, cs = [], []
    for j in range(n_in):
        base = 0x10000000 * (j + 1)
        X = [[rf32(rng.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            g.write_bytes(base + 4 * r * ld, b"".join(f2b(x) for x in X[r]))
        c = rng.choice([1.0, -1.0, 0.5, -0.125, 2.0])
        ins.append(X)
        cs.append(c)
        params += pad if len(params) % 8 else b""
        params += struct.pack("<QIf", base, ld, c)
    params += struct.pack("<I", cols)
    # rows -> ctaid.y: run each row as its own CTA (the emulator runs one CTA per Machine)
    bad = 0
    for r in range(rows):
        L = Launch(grid=(1, rows, 1), block=(256, 1, 1), ctaid=(0, r, 0), params=params,
                   param_base=PARAM_BASE[arch], poison=0xDEADBEEF)
        Machine(load_listing(sass), L, g).run(max_steps=10 ** 7)
        for c in range(cols):
            acc = rf32(ins[0][r][c] * cs[0])
            for j in range(1, n_in):
                acc = rf32(ins[j][r][c] * cs[j] + acc)      # fma: one rounding (exact here?)
            got = b2f(g.read(O0 + 4 * (r * ld + c), 4))
            bad += abs(got - acc) > 1e-6 * max(1.0, abs(acc))
    return None if not bad else f"{bad} wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--variant", default="opt", help="our scheduler: safe | mined | sched | opt")
    ap.add_argument("--only", default="", help="comma-separated kernel name substrings")
    a = ap.parse_args()
    out = ROOT / "corpus" / a.arch / "fastmm"
    out.mkdir(parents=True, exist_ok=True)
    srcs = {"sgemm64": K.sgemm(a.arch), "sgemm128": K.sgemm128(a.arch)}
    for n in range(1, LINCOMB_MAX + 1):
        srcs[f"lincomb{n}"] = K.lincomb(n, a.arch)
    for ka in range(1, 5):
        for kb in range(1, 5):
            for kc in range(1, 5):
                srcs[f"fgemm_{ka}_{kb}_{kc}"] = K.fgemm(ka, kb, kc, a.arch)
    if a.only:
        # Strip empty entries: a trailing/leading/double comma would otherwise
        # leave "" in the list, and "" in k is always True, silently disabling
        # the filter (building and emulator-checking the full ~15 min set).
        needles = [x for x in a.only.split(",") if x]
        srcs = {k: v for k, v in srcs.items() if any(x in k for x in needles)}
    for name, src in srcs.items():
        (out / f"{name}.ptx").write_text(src)
    oracle.run_tool("bash", "-c", f"cd {shlex.quote(str(out))} && for f in *.ptx; do "
                    f"ptxas -arch={a.arch} -O3 -o ptxas@${{f%.ptx}}.cubin $f && "
                    f"nvdisasm -hex -c ptxas@${{f%.ptx}}.cubin > ptxas@${{f%.ptx}}.sass; done")
    enc = Encoder(a.arch)
    for name in srcs:
        k = parse_module((out / f"{name}.ptx").read_text()).kernels[name]
        code, labels = _reschedule_k(k, a.variant)
        sass = listing(code, labels)
        err = (check_fgemm(sass, a.arch, *map(int, name.split("_")[1:])) if name.startswith("fgemm") else
               check_sgemm(sass, a.arch, 128 if name == "sgemm128" else 64)) if "gemm" in name else check_lincomb(sass, int(name[7:]), a.arch)
        if err:
            print(f"{name:10} EMULATOR CHECK FAILED: {err}")
            continue
        words = ra.encode_code(enc, code, labels)
        nop = enc.encode("NOP")
        words.append(enc.encode_branch("BRA `(.L_x_0)", 16 * len(words), 16 * len(words)))
        while len(words) % 8:
            words.append(nop)
        exits = [16 * i for i, (t, _) in enumerate(code) if t.endswith("EXIT")]
        base = (out / f"ptxas@{name}.cubin").read_bytes()
        nregs = required_regcount([t for t, _ in code])
        (out / f"ours@{name}.cubin").write_bytes(splice_text(base, name, words, nregs, exits))
        n_ptxas = sum(1 for i in oracle.parse_listing((out / f"ptxas@{name}.sass").read_text())
                      if i.text != "NOP" and not i.text.startswith("BRA `(.L_x_"))
        print(f"{name:10} ptxas {n_ptxas:4} insns | ours {len(code):4} insns, {nregs} regs: emulator OK")


if __name__ == "__main__":
    main()
