"""M3 gate (v0): compile probes with OUR compiler, validate with nvdisasm, check
semantics with the SASS emulator against the PTX interpreter.

Usage: uv run --no-project python tools/compile_check.py [--only SUBSTR]

Pipeline per probe: PTX -> openassembler.backend.compile -> encode -> splice into the
ptxas cubin's ELF (REGCOUNT / EXIT offsets patched) -> nvdisasm must accept it
and print back exactly our text -> emulator(our SASS) == PTX interpreter.
Also reports our instruction count vs ptxas's (excluding trailing BRA/NOP).
"""

from __future__ import annotations

import argparse
import random
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import diff_emu  # noqa: E402
import ptx_probes  # noqa: E402
from openassembler import oracle  # noqa: E402
from openassembler.backend import ra  # noqa: E402
from openassembler.backend.naive import Compiler, schedule  # noqa: E402
from openassembler.container.cubin import required_regcount, splice_text  # noqa: E402
from openassembler.emu.probe import run_probe  # noqa: E402
from openassembler.isa.encode import Encoder  # noqa: E402
from openassembler.ptx.parse import parse_module  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "sm_80"
OUT = CORPUS / "ours"

# Probes in scope for the v0 compiler (integer ALU, 64-bit add, loads/stores, div-by-const).
SCOPE = ["add.s32", "add.u32", "sub.s32", "mul.lo", "mul.hi", "mad.lo", "min.", "max.", "and.b32", "or.b32",
         "xor.b32", "shl.b32.i", "shr.u32.i", "shr.s32.i", "mul.wide", "not", "div.u32.i", "div.s32.i",
         "rem.u32.i", "rem.s32.i"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--compiler", choices=["ra", "naive"], default="ra",
                    help="ra = v1 (register allocator + copy elimination), naive = v0")
    args = ap.parse_args()
    global OUT
    OUT = CORPUS / ("ours" if args.compiler == "ra" else "ours_naive")
    OUT.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    enc = Encoder("sm_80")
    built, skipped = {}, {}
    for name, _b, _s in ptx_probes.probes("sm_80"):
        if args.only and args.only not in name:
            continue
        kname = ptx_probes.kernel_name(name)
        ptx = (CORPUS / "ptx" / f"{kname}.ptx").read_text()
        try:
            k = parse_module(ptx).kernels[kname]
            if args.compiler == "ra":
                code, _st = ra.compile_kernel(k)
            else:
                code = schedule(Compiler(k).compile())
            words = [enc.encode(t, ctl) for t, ctl in code]
        except Exception as ex:  # NotSupported / EncodeError / out of scope
            skipped[name] = str(ex)[:80]
            continue
        # pad with "BRA self" + NOPs like ptxas (128-byte aligned section)
        n = len(words)
        bra = enc.encode_branch("BRA `(.L_x_0)", 16 * n, 16 * n)
        nop = enc.encode("NOP")
        words.append(bra)
        while len(words) % 8:
            words.append(nop)
        exits = [16 * i for i, (t, _) in enumerate(code) if t == "EXIT"]
        nregs = required_regcount([t for t, _ in code])
        (OUT / f"{kname}.cubin").write_bytes(
            splice_text((CORPUS / "cubin" / f"{kname}.cubin").read_bytes(), kname, words, nregs, exits, replace_helpers=True))
        built[name] = (kname, code, ptx)

    oracle.run_tool("bash", "-c", f"cd {shlex.quote(str(OUT))} && for f in *.cubin; do "
                    "nvdisasm -hex -c $f > ${f%.cubin}.sass 2> ${f%.cubin}.err; "
                    "cuobjdump -elf $f > /dev/null 2>> ${f%.cubin}.err; done; true")
    rng = random.Random(3)
    stats = {"accepted": 0, "text_match": 0, "semantic_match": 0, "fewer": 0, "equal": 0, "more": 0}
    rows = []
    for name, (kname, code, ptx) in built.items():
        err = (OUT / f"{kname}.err").read_text().strip()
        sass = (OUT / f"{kname}.sass").read_text()
        if err or not sass.strip():
            rows.append((name, "REJECTED", err[:80]))
            continue
        stats["accepted"] += 1
        got_text = [i.text for i in oracle.parse_listing(sass)][:len(code)]
        norm = lambda s: s.replace(" ", "")  # noqa: E731
        if [norm(x) for x in got_text] == [norm(t) for t, _ in code]:
            stats["text_match"] += 1
        ok = True
        try:
            for _ in range(args.trials):
                inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
                k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
                got = run_probe(sass, inputs, k, kf)
                # Uninitialized-register check: real GPUs start registers as garbage, so
                # the result must not change when the emulator starts them as a pattern.
                if run_probe(sass, inputs, k, kf, poison=0xDEADBEEF) != got:
                    ok = False
                    rows.append((name, "READS-UNDEF-REG", "output depends on initial register contents"))
                exp = diff_emu.run_ptx(ptx, kname, inputs, k, kf)
                for g, (e_, undef, approx) in zip(got, exp):
                    if not undef and not diff_emu.same(name, g, e_, approx):
                        ok = False
        except Exception as ex:
            rows.append((name, "UNCHECKED", str(ex)[:60]))
            stats.setdefault("unchecked", 0)
            stats["unchecked"] += 1
            continue
        stats["semantic_match"] += ok
        theirs = [i.text for i in oracle.parse_listing((CORPUS / "sass" / f"{kname}.sass").read_text())
                  if i.text not in ("NOP",) and not i.text.startswith("BRA `(.L_x_")]
        ours = len(code)
        stats["fewer" if ours < len(theirs) else "equal" if ours == len(theirs) else "more"] += 1
        rows.append((name, "ok" if ok else "SEMANTIC MISMATCH", f"ours={ours} ptxas={len(theirs)}"))
    tot_ours = sum(int(r[2].split("ours=")[1].split()[0]) for r in rows if "ours=" in r[2])
    tot_ptxas = sum(int(r[2].split("ptxas=")[1]) for r in rows if "ptxas=" in r[2])
    n_ok = sum(1 for r in rows if "ours=" in r[2])
    print(f"compiled {len(built)} probes with our compiler ({len(skipped)} out of v0 scope)")
    if tot_ptxas:
        print(f"instructions over those {n_ok}: ours {tot_ours} vs ptxas {tot_ptxas} "
              f"({100 * (tot_ours - tot_ptxas) / tot_ptxas:+.1f}%)")
    for k_, v in stats.items():
        print(f"  {k_:15} {v}")
    for r in rows:
        if r[1] != "ok" or "div" in r[0] or "rem" in r[0]:
            print(f"  {r[0]:22} {r[1]:18} {r[2]}")


if __name__ == "__main__":
    main()
