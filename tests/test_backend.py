"""Encoder, ELF/.nv.info containers, and div-by-constant sequences (offline)."""

import json
import random
from pathlib import Path

import pytest

from openassembler.container import nvinfo
from openassembler.container.elf import Elf
from openassembler.isa.encode import EncodeError, Encoder
from openassembler.isa.word import control
from openassembler.opt.divconst import lower, run

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "sm_80"
# pairs.jsonl is committed; the cubins/sass/ptx next to it are build output
# (gitignored), made by tools/build_corpus.py with the NVIDIA toolkit.
needs_corpus = pytest.mark.skipif(not (CORPUS / "cubin").is_dir() or not any((CORPUS / "cubin").glob("*.cubin")),
                                  reason="build the probe corpus first: python tools/build_corpus.py sm_80 (needs the CUDA toolkit)")
needs_smoke = pytest.mark.skipif(not (CORPUS / "smoke.cubin").exists(),
                                 reason="corpus/sm_80/smoke.cubin is build output (tools/build_corpus.py)")


@needs_corpus
def test_elf_roundtrip_all_cubins():
    files = sorted((CORPUS / "cubin").glob("*.cubin"))
    assert files
    for f in files:
        b = f.read_bytes()
        assert Elf.parse(b).serialize() == b, f.name


@needs_smoke
def test_nvinfo_roundtrip_and_values():
    e = Elf.parse((CORPUS / "smoke.cubin").read_bytes())
    recs = nvinfo.parse(e.section(".nv.info.saxpy").data)
    assert nvinfo.serialize(recs) == e.section(".nv.info.saxpy").data
    by = {r.name: r for r in recs}
    assert nvinfo.value(by["CBANK_PARAM_SIZE"]) == 0x18
    assert nvinfo.u32s(by["EXIT_INSTR_OFFSETS"]) == [0x50, 0xE0]


@needs_corpus
def test_encoder_never_emits_wrong_word_on_corpus():
    """Every word the encoder produces from a corpus text equals ptxas's word
    (refusals are allowed; wrong answers are not)."""
    enc = Encoder("sm_80")
    seen = {}
    for line in (CORPUS / "pairs.jsonl").open():
        r = json.loads(line)
        seen.setdefault(int(r["word"], 16), r["text"])
    produced = 0
    for w, t in seen.items():
        if "`" in t:
            continue  # label operands go through encode_branch
        try:
            assert enc.encode(t, control(w)) == w, t
            produced += 1
        except EncodeError:
            pass
    assert produced > 1000


def test_imad_wide_signed_cbank_template():
    """IMAD.WIDE Rd, Ra, Rb, c[bank][off] (signed, no .U32) is the form our
    backend emits when it fuses a signed `mul.wide.s32` + `add.s64` of a
    cbank-held pointer (the idiom nvcc itself emits for `arr[int i]` --
    openassembler/pyfront and real CUDA C both hit this). It had no template
    until synthesize() found and nvdisasm-verified one (specs/sm_80/
    synth_templates.json) -- this test is the regression lock for that fix."""
    enc = Encoder("sm_80")
    assert ("IMAD", ("WIDE",), ("R", "R", "R", "CBank")) in enc.templates
    assert enc.encode("IMAD.WIDE R4, R0, R2, c[0x0][0x160]") is not None


def test_encoder_rejects_out_of_range_register_and_control_fields():
    enc = Encoder("sm_80")
    with pytest.raises(EncodeError, match="rd"):
        enc.encode("MOV R256, R2")
    from openassembler.isa.word import Control
    with pytest.raises(EncodeError, match="stall"):
        enc.encode("MOV R1, R2", Control(16, False, 7, 7, 0, 0))


def test_float_immediate_setp_learned_and_round_trips():
    """FSETP's immediate operand (`if v < 0.0`, pyfront's ReLU) printed as
    decimal scientific notation by nvdisasm ('1.17549435...e-38'), which
    Imm.int32() couldn't parse before -- the slot stayed unlearned even
    though the template itself existed. Regression lock for that fix
    (openassembler/emu/operands.py: Imm.int32 / _f32_text_to_bits)."""
    from openassembler.emu.operands import Imm

    assert Imm("0.0").int32() == 0x00000000
    assert Imm("1.0").int32() == 0x3F800000
    assert Imm("+INF").int32() == 0x7F800000
    assert Imm("NAN").int32() == 0x7FC00000
    # a double's immediate exceeds f32 range; only the high 32 bits are a
    # 32-bit immediate slot's worth (see the DMUL case this came from).
    assert Imm("8.98846567431157953865e+307").int32() == 0x7FE00000

    enc = Encoder("sm_80")
    t = enc.templates[("FSETP", ("LT", "AND"), ("P", "P", "R", "Imm", "P"))]
    assert t.slots[3] == ("imm", "imm32")
    assert enc.encode("FSETP.LT.AND P0, PT, R3, 0x0, PT") is not None


@pytest.mark.parametrize("op,signed", [("div", False), ("div", True), ("rem", False), ("rem", True)])
def test_divconst_sequences_random(op, signed):
    """Fast random check; the full proof is tools/verify_divconst_exhaustive.py (all 2^32 x)."""
    rng = random.Random(0)
    for d in [2, 3, 7, 10, 641, 1000, 0x1234, 1000000007] + [rng.randrange(2, 1 << 31) for _ in range(20)]:
        seq = lower(op, signed, d)
        for _ in range(500):
            x = rng.getrandbits(32)
            if signed:
                sx = x - (1 << 32) if x >> 31 else x
                q = abs(sx) // d * (1 if sx >= 0 else -1)
                want = (q if op == "div" else sx - q * d) & 0xFFFFFFFF
            else:
                want = x // d if op == "div" else x % d
            assert run(seq, x) == want, (op, signed, d, x)
