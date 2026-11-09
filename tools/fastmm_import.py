"""One-time import of AlphaTensor's factorizations into a numpy-free spec file.

    uv run --no-project --with numpy python tools/fastmm_import.py
    -> specs/fastmm/schemes.json.gz   (sparse: per scheme, nonzeros of u, v, w)

factorizations_r.npz stores pickled object arrays, so numpy is needed once; everything
else (openassembler.fastmm) reads the spec with the stdlib. w is converted from AlphaTensor's
C^T indexing (k*n+i) to row-major C (i*p+k); openassembler.fastmm.factors.verify() re-checks
every scheme exactly.
"""

import gzip
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AT = ROOT / "third_party" / "alphatensor"


def sparse(mat):
    mat = np.asarray(mat)
    return [[int(i), int(r), str(Fraction(float(mat[i, r])).limit_denominator(1 << 20))]
            for i, r in zip(*np.nonzero(mat))]


def wt(w, n, p):
    w = np.asarray(w)
    return np.stack([w[k * n + i] for i in range(n) for k in range(p)])


def main():
    out = {}
    d = np.load(AT / "algorithms" / "factorizations_r.npz", allow_pickle=True)
    for k in d.files:
        n, m, p = map(int, k.split(","))
        u, v, w = d[k]
        out[f"at_{n}x{m}x{p}"] = {"n": n, "m": m, "p": p, "rank": int(u.shape[1]),
                                  "source": f"third_party/alphatensor/algorithms/factorizations_r.npz[{k}]",
                                  "u": sparse(u), "v": sparse(v), "w": sparse(wt(w, n, p))}
    src = (AT / "benchmarking" / "factorizations.py").read_text()
    ns = {"np": np}
    exec(compile(src, "factorizations.py", "exec"), ns)
    for fn, name in (("get_4x4x4_alphatensor_gpu", "at_gpu_4x4x4"),
                     ("get_4x4x4_strassen_squared", "strassen2_4x4x4"),
                     ("get_4x4x4_alphatensor_tpu", "at_tpu_4x4x4")):
        if fn not in ns:
            continue
        u, v, w = ns[fn]()
        out[name] = {"n": 4, "m": 4, "p": 4, "rank": int(u.shape[1]),
                     "source": f"third_party/alphatensor/benchmarking/factorizations.py:{fn}",
                     "u": sparse(u), "v": sparse(v), "w": sparse(wt(w, 4, 4))}
    dst = ROOT / "specs" / "fastmm" / "schemes.json.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(out)} schemes -> {dst.relative_to(ROOT)} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
