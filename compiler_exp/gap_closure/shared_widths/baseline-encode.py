"""Template encoder: SASS text -> 128-bit word, by patching a known word.

Strategy (M3 "encoder patches templates"): pick a corpus word with the same
mnemonic+modifiers and the same operand *kinds*, then rewrite only fields whose
positions are verified (docs/findings-sm80-encoding.md):
  guard 12..15, Rd 16..23, Ra 24..31, Rb 32..39 | imm32 32..63 | cbank 38..58, Rc 64..71,
  and the control bits 105..125.
Operand-slot -> field mapping is *learned* per template group from the corpus:
for each operand position we find which field holds its value in every example.
Anything not learnable is refused (EncodeError) rather than guessed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from openassembler.emu.operands import CBank, Imm, Mem, Pred, Reg
from openassembler.emu.program import parse_insn
from openassembler.isa.word import Control, bits, set_bits

ROOT = Path(__file__).resolve().parent.parent.parent

# candidate fields an operand value may live in: name -> (lo, width)
REG_FIELDS = {"rd": (16, 8), "ra": (24, 8), "rb": (32, 8), "rc": (64, 8)}
UREG_FIELDS = {"urb": (32, 6), "ura": (24, 6), "urd": (16, 6)}
PRED_FIELDS = {"p81": (81, 3), "p84": (84, 3), "p87": (87, 3), "p77": (77, 3), "p68": (68, 3)}
PRED_NEG = {"p87": 90, "p77": 80, "p68": 71}  # negation bit for predicate *sources*


class EncodeError(Exception):
    pass


def _require_unsigned(name: str, value: int, width: int) -> None:
    if not isinstance(value, int) or not 0 <= value < (1 << width):
        raise EncodeError(f"{name}={value!r} does not fit {width} bits")


_BITFLIP: dict | None = None
_SCALE = {1: 0, 4: 1, 8: 2, 16: 3}
# (opcode, operand index) -> (lo, width) for immediates that are not the 32-bit imm slot
_SECONDARY_IMM = {("LOP3", 4): (72, 8), ("LOP3", 5): (72, 8), ("LEA", 3): (75, 5), ("LEA", 4): (75, 5)}
_SCALED = {"LDS", "STS", "ATOMS", "LDSM"}


def _load_json_tolerant(f: Path) -> dict:
    """bitflip.json may have tools/bitflip.py's text report in front of it when the
    tool's stdout was redirected into the spec file; parse from the first '{' line."""
    t = f.read_text()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i = t.find("\n{")
        return json.loads(t[i + 1:]) if i >= 0 else {}
# S2R special-register numbers (docs/research/02: TID 0x21-23, CTAID 0x25-27, LANEID 0, CLOCKLO 0x50)
# (extended from every S2R in the sm_80 corpus, bits 72..79)
_SR = {"SR_LANEID": 0x00, "SR_VIRTID": 0x03, "SR_TID.X": 0x21, "SR_TID.Y": 0x22, "SR_TID.Z": 0x23,
       "SR_CTAID.X": 0x25, "SR_CTAID.Y": 0x26, "SR_CTAID.Z": 0x27, "SR_SMEMSZ": 0x32,
       "SR_EQMASK": 0x38, "SR_LTMASK": 0x39, "SR_GEMASK": 0x3C, "SR_VIRTUALSMID": 0x43,
       "SR_CLOCKLO": 0x50}


def _disambiguate(cand: list[str], pos: int, base, fields) -> list[str]:
    """Break ties using the bit-flip map (specs/<arch>/bitflip.json): keep fields whose
    bits were observed to change operand `pos` for this opcode/form."""
    global _BITFLIP
    if len(cand) <= 1:
        return cand
    if _BITFLIP is None:
        f = ROOT / "specs" / "sm_80" / "bitflip.json"
        _BITFLIP = _load_json_tolerant(f) if f.exists() else {}
    e = _BITFLIP.get(f"{bits(base.word, 0, 9):03x}.{bits(base.word, 9, 3)}")
    if not e:
        return cand
    owner = {}
    for fl in e["fields"]:
        for b in range(fl["lo"], fl["lo"] + fl["width"]):
            owner[b] = fl["class"]
    keep = [n for n in cand if owner.get(fields[n][0], "").startswith("op") and
            str(pos) in owner[fields[n][0]][2:].split("+")]
    return keep or cand


def kinds(ops) -> tuple:
    out = []
    for o in ops:
        if isinstance(o, Reg):
            out.append(o.file + ("~" if o.bnot else "") + ("-" if o.neg else "") + ("|" if o.abs else "")
                       + (("." + o.sel) if o.sel else ""))
        elif isinstance(o, Pred):
            out.append(o.file)
        else:
            mod = ""
            if isinstance(o, CBank):
                mod = ("-" if o.neg else "") + ("|" if o.abs else "") + ("~" if o.bnot else "")
            out.append(type(o).__name__ + (".64" if isinstance(o, Mem) and o.wide else "") + mod)
    return tuple(out)


def key(insn) -> tuple:
    return (insn.op, insn.mods, kinds(insn.ops))


@dataclass
class Template:
    word: int
    slots: list  # per operand: (kind, field name or None)
    ops: list    # the template's own parsed operands (for strict equality checks)


class Encoder:
    def __init__(self, arch: str = "sm_80"):
        pairs = ROOT / "corpus" / arch / "pairs.jsonl"
        if not pairs.exists():
            have = sorted(d.name for d in (ROOT / "corpus").glob("sm_*") if (d / "pairs.jsonl").exists())
            raise EncodeError(f"no SASS encoder data for {arch!r} (only {', '.join(have) or 'sm_80'} "
                              "has been reverse-engineered; see docs/STATUS.md \"Gaps\")")
        rows = [json.loads(l) for l in open(pairs)]
        synth = ROOT / "specs" / arch / "synth_templates.json"
        if synth.exists():  # nvdisasm-verified words made by synthesize()
            rows += json.loads(synth.read_text())
        groups = defaultdict(dict)
        self.corpus_words = [int(r["word"], 16) for r in rows[:-len(json.loads(synth.read_text()))
                                                                 if synth.exists() else None]]
        for r in rows:
            w = int(r["word"], 16)
            ins = parse_insn(0, r["text"], w)
            groups[key(ins)][w] = ins
        self.templates: dict[tuple, Template] = {}
        synth_words = {int(r["word"], 16) for r in (json.loads(synth.read_text()) if synth.exists() else [])}
        arithmetic = ROOT / "specs" / arch / "arithmetic_templates.json"
        if arithmetic.exists():
            for r in json.loads(arithmetic.read_text()):
                w = int(r["word"], 16)
                ins = parse_insn(0, r["text"], w)
                groups[key(ins)][w] = ins
                synth_words.add(w)
        for k, members in groups.items():
            t = self._learn(list(members.values()), canonical=all(w in synth_words for w in members))
            if t:
                self.templates[k] = t

    @staticmethod
    def _learn(members, canonical: bool = False) -> Template | None:
        """canonical=True (synthesized, single-example templates): an operand slot whose
        field is ambiguous falls back to the canonical Volta+ field for its position
        (first R -> rd 16..23, then ra 24..31, rb 32..39, rc 64..71; cbank 38..58).
        Callers must verify such words with nvdisasm (tools/compile_check.py does)."""
        base = members[0]
        slots = []
        canon_regs = None
        owner = field_probe(base.word) if canonical else {}
        for pos, op in enumerate(base.ops):
            # A slot is learned only if exactly one field matches in every example;
            # several matches (e.g. "R6, R6" seen once) are ambiguous and refused.
            if isinstance(op, Reg):
                fields = UREG_FIELDS if op.file == "UR" else REG_FIELDS
                cand = [n for n, (lo, w) in fields.items()
                        if all(bits(m.word, lo, w) == m.ops[pos].idx for m in members)]
                cand = _disambiguate(cand, pos, base, fields)
                if len(cand) != 1 and owner:
                    # single example: pick the field whose low bit the bit-flip probe ties to this operand
                    cand = [n for n in fields if owner.get(fields[n][0]) == f"op{pos}"]
                slots.append(("reg", cand[0] if len(cand) == 1 else None))
            elif isinstance(op, Pred):
                cand = [n for n, (lo, w) in PRED_FIELDS.items()
                        if all(bits(m.word, lo, w) == m.ops[pos].idx for m in members)]
                cand = _disambiguate(cand, pos, base, PRED_FIELDS)
                slots.append(("pred", cand[0] if len(cand) == 1 else None))
            elif isinstance(op, CBank):
                if owner and owner.get(40) == f"op{pos}" and owner.get(54) == f"op{pos}":
                    slots.append(("cbank", "cb"))
                    continue
                ok = all(m.ops[pos].neg == op.neg and m.ops[pos].abs == op.abs and isinstance(m.ops[pos], CBank) and m.ops[pos].ur is None and m.ops[pos].reg is None
                         and bits(m.word, 38, 16) == m.ops[pos].off and bits(m.word, 54, 5) == m.ops[pos].bank
                         for m in members)
                slots.append(("cbank", "cb" if ok else None))
            elif isinstance(op, Mem):
                # [Ra(.64) + off]: Ra at 24..31, signed 24-bit byte offset at 40..63 (bitflip-verified)
                ok = all(isinstance(m.ops[pos], Mem) and m.ops[pos].base is not None and m.ops[pos].ur is None
                         and bits(m.word, 24, 8) == m.ops[pos].base.idx
                         and bits(m.word, 40, 24) == m.ops[pos].off & 0xFFFFFF for m in members)
                # LDS/STS/ATOMS/LDSM scale .X1/.X4/.X8/.X16 at bits 78..79 (bitflip + nvdisasm sweep)
                sc = all(bits(m.word, 78, 2) == _SCALE.get(m.ops[pos].scale, -1) for m in members)
                slots.append(("mem", "mem+scale" if ok and sc and base.op in _SCALED else ("mem" if ok else None)))
            elif isinstance(op, Imm):
                # Secondary immediates (bitflip map, specs/sm_80/bitflip.json):
                #   LOP3/PLOP3 LUT at 72..79; LEA shift at 75..79; PRMT selector reg form n/a.
                if base.op == "PLOP3" and pos == 5:
                    ok = all((bits(m.word, 64, 3) | bits(m.word, 72, 5) << 3) == m.ops[pos].int32() for m in members)
                    slots.append(("plut", "split" if ok else None))
                    continue
                if base.op == "PLOP3" and pos == 6:
                    ok = all(bits(m.word, 16, 8) == m.ops[pos].int32() for m in members)
                    slots.append(("imm2", (16, 8) if ok else None))
                    continue
                sec = _SECONDARY_IMM.get((base.op, pos))
                if sec is not None:
                    lo, wd = sec
                    try:
                        ok = all(bits(m.word, lo, wd) == m.ops[pos].int32() for m in members)
                    except ValueError:
                        ok = False
                    slots.append(("imm2", sec if ok else None))
                    continue
                # The immediate is bits 32..63. nvdisasm prints it either as the raw integer
                # (IADD3, ISETP, ...) or, for float opcodes, as a decimal FLOAT -- which can
                # look exactly like an integer ("FMUL R9, R9, 16777216" is 16777216.0, bits
                # 0x4b800000, not 0x01000000). Learn which reading this opcode uses, and
                # remember it in the slot so encode() applies the same one.
                def _reads(conv):
                    ok_ = True
                    for m in members:
                        try:
                            ok_ = ok_ and conv(m.ops[pos]) == bits(m.word, 32, 32)
                        except (ValueError, OverflowError):
                            return False
                    return ok_

                if _reads(lambda o: o.int32()):
                    slots.append(("imm", "imm32"))
                elif _reads(lambda o: o.f32bits()):
                    slots.append(("imm", "imm32f"))
                else:
                    slots.append(("imm", None))
            elif base.op in ("S2R", "S2UR", "CS2R") and pos == 1 and _SR.get(op.text) is not None:
                # special-register index at bits 72..79 (S2R SR_TID.X = 0x21, measured)
                ok = all(bits(m.word, 72, 8) == _SR.get(m.ops[pos].text) for m in members)
                slots.append(("sr", "sr" if ok else None))
            else:
                slots.append(("other", None))
        return Template(base.word, slots, list(base.ops))

    def encode_branch(self, text: str, addr: int, target: int, ctl: Control | None = None) -> int:
        """BRA/BSSY/CALL.REL with a resolved target address. Offset = target - (addr + 16),
        signed; BRA/CALL hold it at bits 34..81 in units of 4 bytes, BSSY at 34..63
        (bitflip map, specs/sm_80/bitflip.json)."""
        ins = parse_insn(0, text, 0)
        t = self.templates.get(key(ins))
        if t is None or ins.op not in ("BRA", "BSSY", "CALL"):
            raise EncodeError(f"no branch template for {text}")
        if any(isinstance(o, Pred) for o in ins.ops):
            raise EncodeError("branch with predicate operand (BRA Px, ...) not learned")
        rel = target - (addr + 16)
        if rel % 4:
            raise EncodeError("misaligned branch target")
        lo, width = (34, 30) if ins.op == "BSSY" else (34, 48)
        if not -(1 << (width - 1)) <= rel // 4 < (1 << (width - 1)):
            raise EncodeError(f"branch displacement {rel} does not fit signed {width}-bit field")
        w = set_bits(t.word, lo, width, (rel >> 2) & ((1 << width) - 1))
        if ins.op == "BSSY":
            barrier = int(ins.ops[0].text[1:])
            _require_unsigned("barrier", barrier, 4)
            w = set_bits(w, 16, 4, barrier)  # barrier index B0..B15 (bits 16..19)
        if ins.guard is not None:
            _require_unsigned("branch predicate", ins.guard.idx, 3)
        w = set_bits(set_bits(w, 12, 3, 7 if ins.guard is None else ins.guard.idx), 15, 1,
                     int(ins.guard.neg) if ins.guard else 0)
        if ctl is not None:
            w = self._ctl(w, ctl)
        return w

    @staticmethod
    def _ctl(w: int, ctl: Control) -> int:
        _require_unsigned("stall", ctl.stall, 4)
        _require_unsigned("write scoreboard", ctl.wbar, 3)
        _require_unsigned("read scoreboard", ctl.rbar, 3)
        _require_unsigned("wait mask", ctl.wait, 6)
        _require_unsigned("reuse mask", ctl.reuse, 4)
        w = set_bits(w, 105, 4, ctl.stall)
        w = set_bits(w, 109, 1, int(not ctl.yield_))
        w = set_bits(w, 110, 3, ctl.wbar)
        w = set_bits(w, 113, 3, ctl.rbar)
        w = set_bits(w, 116, 6, ctl.wait)
        return set_bits(w, 122, 4, ctl.reuse)

    def encode(self, text: str, ctl: Control | None = None) -> int:
        ins = parse_insn(0, text, 0)
        t = self.templates.get(key(ins))
        if t is None:
            raise EncodeError(f"no template for {key(ins)}")
        w = t.word
        if ins.guard is None:
            w = set_bits(set_bits(w, 12, 3, 7), 15, 1, 0)
        else:
            _require_unsigned("guard predicate", ins.guard.idx, 3)
            w = set_bits(set_bits(w, 12, 3, ins.guard.idx), 15, 1, int(ins.guard.neg))
        tmpl_ins = parse_insn(0, "", 0) if False else None  # noqa: F841
        for pos, (op, (kind, field)) in enumerate(zip(ins.ops, t.slots)):
            if field is None and op == t.ops[pos]:
                continue  # unlearned slot, but the value is the template's own: nothing to patch
            if kind == "reg":
                if field is None:
                    raise EncodeError(f"register slot {pos} not learned")
                lo, wd = (UREG_FIELDS | REG_FIELDS)[field]
                _require_unsigned(field, op.idx, wd)
                w = set_bits(w, lo, wd, op.idx)
            elif kind == "pred":
                if field is None:
                    raise EncodeError(f"predicate slot {pos} not learned")
                lo, wd = PRED_FIELDS[field]
                _require_unsigned(field, op.idx, wd)
                w = set_bits(w, lo, wd, op.idx)
                if field in PRED_NEG:
                    w = set_bits(w, PRED_NEG[field], 1, int(op.neg))
            elif kind == "cbank":
                if op.neg != t.ops[pos].neg or op.abs != t.ops[pos].abs:
                    raise EncodeError("cbank modifier differs from template")
                if field is None:
                    raise EncodeError(f"cbank slot {pos} not learned")
                _require_unsigned("constant-bank offset", op.off, 16)
                _require_unsigned("constant-bank", op.bank, 5)
                w = set_bits(set_bits(w, 38, 16, op.off), 54, 5, op.bank)
            elif kind == "mem":
                if field is None or op.base is None or op.ur is not None:
                    raise EncodeError(f"memory slot {pos} not learned")
                if not -(1 << 23) <= op.off < (1 << 23):
                    raise EncodeError("offset out of 24-bit range")
                _require_unsigned("memory base register", op.base.idx, 8)
                w = set_bits(set_bits(w, 24, 8, op.base.idx), 40, 24, op.off & 0xFFFFFF)
                if field == "mem+scale":
                    w = set_bits(w, 78, 2, _SCALE[op.scale])
                elif op.scale != t.ops[pos].scale:
                    raise EncodeError("memory scale not learned for this opcode")
            elif kind == "imm":
                if field is None:
                    raise EncodeError(f"immediate slot {pos} not learned")
                w = set_bits(w, 32, 32, op.f32bits() if field == "imm32f" else op.int32())
            elif kind == "plut":
                value = op.int32()
                if field is None or not 0 <= value <= 255:
                    raise EncodeError("invalid PLOP3 truth table")
                w = set_bits(set_bits(w, 64, 3, value & 7), 72, 5, value >> 3)
            elif kind == "imm2":
                if field is None:
                    raise EncodeError(f"secondary immediate slot {pos} not learned")
                lo, wd = field
                if op.int32() >> wd:
                    raise EncodeError(f"immediate {op.text} does not fit {wd} bits")
                w = set_bits(w, lo, wd, op.int32())
            elif kind == "sr":
                if field is None or op.text not in _SR:
                    raise EncodeError(f"special register {op.text}")
                w = set_bits(w, 72, 8, _SR[op.text])
            else:
                # labels, barrier names, float literals...: no learned field, so the
                # operand must be identical to the template's or we refuse.
                if op != t.ops[pos]:
                    raise EncodeError(f"operand {pos} ({op}) has no learned field")
        if ctl is not None:
            w = self._ctl(w, ctl)
        return w


# ---- template synthesis: new modifier/operand-form combos, verified by nvdisasm -----------

SYNTH_CACHE = ROOT / "specs" / "sm_80" / "synth_templates.json"


FIELDMAP_CACHE = ROOT / "specs" / "sm_80" / "synth_fieldmaps.json"


def _fieldmap_cache() -> dict:
    try:
        return json.loads(FIELDMAP_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def field_probe(word: int, arch: str = "SM80") -> dict[int, str]:
    """Flip every bit of `word` in the operand regions and record which operand text
    changed (one nvdisasm batch). -> {bit: "opN"}. Used to learn the field map of a
    synthesized (single-example) template instead of guessing canonical positions.

    The answer depends only on (word, arch), so it's cached in specs/sm_80/
    synth_fieldmaps.json. Without the cache, building an Encoder re-ran this ~60
    times through nvdisasm at every start (a minute on macOS, where nvdisasm runs
    in a VM) and needed the NVIDIA toolkit just to load. With it, loading is
    instant and offline; a word that isn't cached yet is probed once and added."""
    key = f"{arch}:{word:032x}"
    cache = _fieldmap_cache()
    if key in cache:
        return {int(b): v for b, v in cache[key].items()}
    out = _field_probe_uncached(word, arch)
    cache[key] = {str(b): v for b, v in out.items()}
    try:
        FIELDMAP_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    except OSError:
        pass  # read-only install: just don't persist
    return out


def _field_probe_uncached(word: int, arch: str) -> dict[int, str]:
    from openassembler import oracle

    base = oracle.disasm_words([word], arch)[0]
    bits_ = [b for b in list(range(16, 72)) + list(range(72, 91))]
    texts = oracle.disasm_words([word ^ (1 << b) for b in bits_], arch, max_rounds=200)
    _, _, ops0 = _split_ops(base)
    out = {}
    for b, t in zip(bits_, texts):
        if not t:
            continue
        _, _, ops1 = _split_ops(t)
        if len(ops1) != len(ops0):
            continue
        diff = [i for i, (x, y) in enumerate(zip(ops0, ops1)) if x != y]
        if len(diff) == 1:
            out[b] = f"op{diff[0]}"
    return out


def _split_ops(text: str):
    import re as _re
    g = ""
    if m := _re.match(r"^(@!?U?P[T0-6])\s+", text):
        g, text = m[1], text[m.end():]
    head, _, rest = text.partition(" ")
    return g, head, [o.strip() for o in rest.split(",")] if rest else []


def synthesize(enc: "Encoder", wanted_texts: list[str], arch: str = "SM80") -> int:
    """For texts with no template, try known words of the same opcode with every
    combination of 1-2 flipped bits in the modifier region (72..90) and the form bits
    (9..11), disassemble the candidates in one nvdisasm batch, and adopt any word whose
    text has exactly the wanted key. Accepted words are cached in specs/<arch>/
    synth_templates.json and loaded by Encoder() like corpus words.
    Returns the number of new templates."""
    from itertools import combinations

    from openassembler import oracle

    todo = {}
    for t in wanted_texts:
        k = key(parse_insn(0, t, 0))
        if k not in enc.templates:
            todo[k] = t
    if not todo:
        return 0
    seeds = {}
    for k in todo:
        # seeds with the same operand signature first (only the first 6 are flipped)
        same = sorted((k2 for k2 in enc.templates if k2[0] == k[0]), key=lambda k2: k2[2] != k[2])
        for k2 in same:
            seeds.setdefault(k, []).append(enc.templates[k2].word)
    region = list(range(72, 91)) + [9, 10, 11]
    cands, owners = [], []
    for k, words in seeds.items():
        for w in words[:6]:
            for n in (1, 2):
                for bs in combinations(region, n):
                    x = w
                    for b in bs:
                        x ^= 1 << b
                    cands.append(x)
                    owners.append(k)
    texts = oracle.disasm_words(cands, arch, max_rounds=400)
    found = {}
    for w, k, t in zip(cands, owners, texts):
        if t and k not in found:
            try:
                if key(parse_insn(0, t, w)) == k:
                    found[k] = (w, t)
            except Exception:  # noqa: BLE001
                pass
    def _peers(w):
        return [p for p in enc.corpus_words if bits(p, 0, 12) == bits(w, 0, 12)]
    found = {k: (minimize(w, arch, _peers(w)), t) for k, (w, t) in found.items()}
    cache = json.loads(SYNTH_CACHE.read_text()) if SYNTH_CACHE.exists() else []
    for k, (w, t) in found.items():
        cache.append({"word": f"{w:032x}", "text": t})
        ins = parse_insn(0, t, w)
        enc.templates[k] = Encoder._learn([ins], canonical=True)
    SYNTH_CACHE.write_text(json.dumps(cache, indent=1))
    return len(found)


def minimize(word: int, arch: str = "SM80", peers: list[int] | None = None) -> int:
    """Canonicalize bits that nvdisasm does not print. A seed word's leftover bits can sit
    in fields the hardware reads: a synthesized ISETP kept 0xad of an old immediate in bits
    32..39 and made every loop on the A100 run one extra iteration. Clearing such bits is
    wrong too (bits 68..70 of ISETP are a hidden PT = 7). So each bit whose flip leaves the
    text unchanged is set to the majority value among `peers` (corpus words with the same
    opcode and form bits), when that majority still disassembles identically."""
    from openassembler import oracle

    if not peers:
        return word
    ref = oracle.disasm_words([word], arch)[0]
    silent = [b for b, t in zip(range(9, 105), oracle.disasm_words(
        [word ^ (1 << b) for b in range(9, 105)], arch)) if t == ref]
    x = word
    for b in silent:
        ones = sum((p >> b) & 1 for p in peers)
        x = set_bits(x, b, 1, int(2 * ones > len(peers)))
    return x if oracle.disasm_words([x], arch)[0] == ref else word
