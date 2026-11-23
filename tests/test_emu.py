"""SASS emulator vs PTX interpreter on the sm_80 probe corpus (M1 gate).

Runs offline (no NVIDIA tools) on the committed corpus/sm_80/{ptx,sass}.
"""

import hashlib
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import diff_emu  # noqa: E402
import ptx_probes  # noqa: E402
from openassembler.emu.fp import add, decode, encode, mul  # noqa: E402
from openassembler.emu.ops import lut3  # noqa: E402
from openassembler.emu.probe import run_probe  # noqa: E402
from openassembler.emu.machine import Launch, Machine  # noqa: E402
from openassembler.emu.program import load_listing, parse_insn  # noqa: E402

CORPUS = ROOT / "corpus" / "sm_80"
needs_corpus = pytest.mark.skipif(not (CORPUS / "sass").exists(), reason="run tools/build_corpus.py sm_80")


def test_lut3_identities():
    a, b, c = 0xF0F0F0F0, 0xCCCCCCCC, 0xAAAAAAAA
    assert lut3(0xF0, a, b, c) == a and lut3(0xCC, a, b, c) == b and lut3(0xAA, a, b, c) == c
    assert lut3(0x96, a, b, c) == a ^ b ^ c


def test_fp_rounding_edges():
    one = decode(0x3F800000)
    tiny = decode(0x33800000)  # 2^-24: exactly half an ulp of 1.0
    assert encode(add(one, tiny, "RN")) == 0x3F800000          # tie -> even
    assert encode(add(one, tiny, "RP"), mode="RP") == 0x3F800001
    big = decode(0x7F7FFFFF)
    assert encode(mul(big, decode(0x40000000))) == 0x7F800000   # overflow -> inf
    assert encode(mul(big, decode(0x40000000)), mode="RZ") == 0x7F7FFFFF
    assert encode(decode(0x00000001), ftz=True) == 0             # subnormal flush


@pytest.mark.parametrize('sign', [0, 0x80000000])
@pytest.mark.parametrize('exponent,slow', [(-125, True), (-103, True), (-102, False), (-101, False)])
def test_fchk_small_dividend_boundary(sign, exponent, slow):
    # Measured on A100 with this mantissa and denominator exponent zero.
    # This checks the observed boundary, not a complete hardware FCHK truth table.
    from openassembler.emu.ops import ALU
    machine = Machine(load_listing(''), Launch(block=(1, 1, 1)))
    warp = machine.warps[0]
    lane = warp.lanes[0]
    lane.r[4] = sign | ((exponent + 127) << 23) | 0x47ec48
    lane.r[5] = 0x3fdea823
    ALU['FCHK'](machine, lane, warp, parse_insn(0, 'FCHK P0, R4, R5', 0))
    assert lane.p[0] is slow


def test_exact_division_retained_counterexamples():
    # Frozen CUDA12.8 ptxas sequence, retained independently of generated corpus
    # availability. Expected bits come from integer-ratio RNE and the A100 replay.
    sass = (ROOT / 'tests/data/emu/p_div_rn_f32.sass.txt').read_text()
    cases = [
        (0x8147ec48, 0x21dea823, 0x9ee5dc9d),
        (0x8147ec47, 0x21dea822, 0x9ee5dc9c),
        (0x8147ec47, 0x21dea824, 0x9ee5dc9a),
        (0x8147ec49, 0x21dea824, 0x9ee5dc9d),
        (0x8147ec48, 0x3fdea823, 0x80e5dc9d),
    ]
    inputs = [(a, b, 0, 0xc5bae70c) for a, b, _ in cases]
    expected = [(value, 0xc5bae70c) for _, _, value in cases]
    assert run_probe(sass, inputs, 0, 0) == expected
    assert run_probe(sass, inputs, 0, 0, poison=0xdeadbeef) == expected


# Probes that must match bit-exactly. Grows as coverage grows; never shrinks.
GATE = [
    "add.s32.r", "add.s32.c", "add.s32.i", "sub.s32.r", "mul.lo.s32.r", "mul.hi.s32.r", "mul.hi.u32.c",
    "mad.lo.s32.r", "mad.hi.s32.r", "mad.hi.u32.i", "mul.wide.s32.r", "mul.wide.u32.c",
    "div.s32.r", "div.u32.i", "rem.s32.c", "rem.u32.r", "min.s32.r", "max.u32.c",
    "add.s64", "sub.s64", "mul.lo.s64", "min.s64", "shl.b64", "shr.u64", "shf.l", "shf.r",
    "shl.b32.r", "shr.s32.r", "shr.u32.c", "and.b32.i", "xor.b32.c", "lop3", "not", "neg", "abs",
    "popc", "clz", "brev", "bfe", "bfi", "prmt", "prmt.r", "add.cc",
    "setp.lt.s32.r", "setp.ge.u32.c", "setp.ne.s32.i", "setp.lt.s64", "setp.and", "setp.or", "setp.xor",
    "setp.2dst", "pred.logic", "slct",
    "add.f32.r", "add.rz.f32.c", "add.rm.f32.i", "add.rp.f32.r", "add.sat.f32.r", "add.ftz.f32.r",
    "mul.f32.r", "mul.rz.f32.c", "fma.rn.f32.r", "fma.rz.f32.c", "fma.rn.sat.f32.i", "div.rn.f32",
    "min.f32.r", "min.NaN.f32.c", "max.f32.i", "abs.f32", "neg.f32", "sqrt.rn.f32", "rcp.rn.f32",
    "setp.lt.f32.r", "setp.nan.f32.c", "setp.geu.f32.i", "setp.ne.f64", "testp", "copysign",
    "add.f64", "mul.f64", "fma.f64",
    "cvt.s32.s8", "cvt.u32.u16", "cvt.s64.s32", "cvt.u32.s64", "cvt.rni.s32.f32", "cvt.rzi.u32.f64",
    "cvt.rmi.s64.f32", "cvt.rn.f32.s32", "cvt.rz.f64.u64", "cvt.f64.f32", "cvt.rn.f32.f64", "cvt.rni.f32.f32",
    "ld.global.u8", "ld.global.s16", "ld.global.v4.b32", "ld.global.nc.b64", "ld.shared.s8", "ld.shared.v2.b32",
    "ld.local.b32", "ld.const", "st.global.u8", "st.global.b64", "st.shared.v4.b32",
    "atom.global.add.u32", "atom.global.min.s32", "atom.global.inc.u32", "atom.cas.b32", "atom.cas.b64",
    "atom.add.u64", "atom.shared.exch.b32", "red.global.xor.b32",
    "shfl.idx.r", "shfl.up.i", "shfl.down.r", "shfl.bfly.i", "shfl.p", "vote.all", "vote.any", "vote.uni",
    "vote.ballot", "match.any", "match.all", "redux.add.u32", "redux.min.s32", "activemask",
    "bar.sync", "bar.red.popc", "bar.red.and",
    "if", "ifelse", "divergent", "early.exit", "exit",
]


@needs_corpus
@pytest.mark.parametrize("probe", GATE)
def test_emulator_matches_ptx(probe):
    kname = ptx_probes.kernel_name(probe)
    sass = (CORPUS / "sass" / f"{kname}.sass").read_text()
    ptx = (CORPUS / "ptx" / f"{kname}.ptx").read_text()
    # Python's hash(str) changes across processes. Retain the full deterministic
    # seed in failures so a numerical counterexample can be replayed exactly.
    seed = int.from_bytes(hashlib.sha256(probe.encode('utf-8')).digest()[:8], 'big')
    rng = random.Random(seed)
    checked = 0
    for trial in range(3):
        inputs = [tuple(diff_emu.rand32(rng) for _ in range(4)) for _ in range(32)]
        k, kf = diff_emu.rand32(rng), diff_emu.rand32(rng)
        got = run_probe(sass, inputs, k, kf)
        exp = diff_emu.run_ptx(ptx, kname, inputs, k, kf)
        for t, (g, (e, undef, approx)) in enumerate(zip(got, exp)):
            if undef:
                continue
            checked += 1
            assert diff_emu.same(probe, g, e, approx), (probe, seed, trial, t, [hex(x) for x in inputs[t]], k, kf, g, e)
    assert checked > 0
