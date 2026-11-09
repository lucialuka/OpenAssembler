"""Build the 2-layer MLP forward-pass kernels (examples/mlp/kernels.py --
linear1, relu, linear2) with ptxas and with our own compiler, for every
optimization level. Each of ours is checked against the PTX reference
interpreter (openassembler.verify.run_and_diff) before being written, same
discipline as tools/fastmm_build.py and tools/bench_build.py.

    PYTHONPATH=. python3 tools/mlp_build.py [--arch sm_80] [--variant opt]

Writes corpus/<arch>/mlp/{ptxas,ours}@<kernel>.cubin.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from bench_build import _reschedule_k  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import ra  # noqa: E402
from openassembler.container.cubin import required_regcount, splice_text  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402
from openassembler.pyfront.compile import compile_kernel as py_compile
from openassembler.pyfront.compile import find_kernel  # noqa: E402
from openassembler.verify import run_and_diff  # noqa: E402

KERNELS_PY = ROOT / "examples" / "mlp" / "kernels.py"
KERNEL_NAMES = ("linear1", "relu", "linear2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--variant", default="opt", help="our scheduler: safe | mined | sched | opt")
    a = ap.parse_args()
    out = ROOT / "corpus" / a.arch / "mlp"
    out.mkdir(parents=True, exist_ok=True)

    srcs = {}
    for name in KERNEL_NAMES:
        fn = find_kernel(KERNELS_PY, name)
        srcs[name] = py_compile(fn, arch=a.arch)
    for name, src in srcs.items():
        (out / f"{name}.ptx").write_text(src)

    oracle.run_tool("bash", "-c", f"cd {shlex.quote(str(out))} && for f in *.ptx; do "
                    f"ptxas -arch={a.arch} -O3 -o ptxas@${{f%.ptx}}.cubin $f && "
                    f"nvdisasm -hex -c ptxas@${{f%.ptx}}.cubin > ptxas@${{f%.ptx}}.sass; done")

    enc = Encoder(a.arch)
    for name, src in srcs.items():
        k = parse_module(src).kernels[name]
        code, labels = _reschedule_k(k, a.variant)
        mod = parse_module(src)
        mismatches = run_and_diff(mod, name, code, labels, n_elems=32, arch=a.arch, seed=0)
        if mismatches:
            m = mismatches[0]
            print(f"{name:10} EMULATOR CHECK FAILED: {m.param}[{m.index}] "
                  f"{m.ours:#x} vs {m.expected:#x} ({len(mismatches)} total)")
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
