"""Bit-flip field discovery: flip each bit of seed words, diff nvdisasm's text.

Usage: uv run --no-project python tools/bitflip.py sm_80

For each distinct (opcode, form) in corpus/<arch>/pairs.jsonl we take one seed
word, produce 128 single-bit variants, and disassemble everything in one batch.
Each bit is classified as:

  invalid    nvdisasm rejects the word        -> constraint bit
  none       text unchanged                   -> unused / control / don't-care
  opcode     base mnemonic changed            -> opcode-space bit
  mod        a mnemonic modifier changed      -> modifier field bit
  guard      the @P guard changed
  op<k>      operand k changed                -> operand field bit

Adjacent bits with the same class form a candidate field. Results go to
specs/<arch>/bitflip.json (machine-readable) and print as a summary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openassembler import oracle  # noqa: E402
from openassembler.isa.word import fields  # noqa: E402

_GUARD = re.compile(r"^(@!?U?P[T0-6])\s+")
_BRANCHES = {"BRA", "BRX", "BSSY", "CALL", "JMP", "JMX", "RET", "BREAK", "BMOV", "WARPSYNC", "BPT"}
_HEX = re.compile(r"(-?0x[0-9a-f]+)\s*$")


def normalize(text: str | None, addr: int) -> str | None:
    """Raw-mode branch targets are absolute (addr + 16 + offset); make them relative
    so a variant at batch slot k compares equal to the seed at slot j."""
    if text is None:
        return None
    _, mn, _ = split(text)
    if mn[0] not in _BRANCHES:
        return text
    return _HEX.sub(lambda m: f"rel{int(m[1], 16) - addr - 16:+#x}", text)


def split(text: str) -> tuple[str, list[str], list[str]]:
    """'@P0 IADD3.X R1, R2, ...' -> ('@P0', ['IADD3', 'X'], ['R1', 'R2', ...])."""
    guard = ""
    if m := _GUARD.match(text):
        guard, text = m[1], text[m.end():]
    head, _, rest = text.partition(" ")
    return guard, head.split("."), [o.strip() for o in rest.split(",")] if rest else []


def classify(seed: str, var: str | None) -> str:
    if var is None:
        return "invalid"
    if var == seed:
        return "none"
    g0, m0, o0 = split(seed)
    g1, m1, o1 = split(var)
    if m0[0] != m1[0]:
        return "opcode"
    if m0 != m1:
        return "mod"
    if g0 != g1:
        return "guard"
    if len(o0) != len(o1):
        return "mod"  # operand count changed: a form/modifier bit
    diff = [k for k in range(len(o0)) if o0[k] != o1[k]]
    return "op" + "+".join(map(str, diff)) if diff else "mod"


def runs(classes: list[str]) -> list[tuple[int, int, str]]:
    """Group adjacent bits with the same class into (lo, width, class)."""
    out, lo = [], 0
    for b in range(1, len(classes) + 1):
        if b == len(classes) or classes[b] != classes[lo]:
            out.append((lo, b - lo, classes[lo]))
            lo = b
    return out


def main(arch: str) -> None:
    pairs = [json.loads(l) for l in open(oracle.ROOT / "corpus" / arch / "pairs.jsonl")]
    seeds: dict[tuple[int, int], tuple[int, str]] = {}
    for r in pairs:
        w = int(r["word"], 16)
        f = fields(w)
        seeds.setdefault((f.opcode, f.form), (w, r["text"]))

    keys = sorted(seeds)
    sm = arch.upper().replace("_", "")
    # Seeds are re-disassembled raw so their text has the same form as the variants'.
    seed_raw = oracle.disasm_words([seeds[k][0] for k in keys], sm)
    seed_norm = {k: normalize(t, 16 * j) for j, (k, t) in enumerate(zip(keys, seed_raw))}
    batch = [seeds[k][0] ^ (1 << b) for k in keys for b in range(128)]
    texts = oracle.disasm_words(batch, sm, max_rounds=256)
    texts = [normalize(t, 16 * n) for n, t in enumerate(texts)]

    spec, report = {}, []
    for i, k in enumerate(keys):
        w, seed_text = seeds[k]
        seed_text = seed_norm[k] or seed_text
        classes = [classify(seed_text, texts[i * 128 + b]) for b in range(128)]
        examples = {b: texts[i * 128 + b] for b in range(128) if classes[b] not in ("none", "invalid")}
        spec[f"{k[0]:03x}.{k[1]}"] = {
            "seed": f"{w:032x}", "text": seed_text,
            "fields": [{"lo": lo, "width": n, "class": c} for lo, n, c in runs(classes)],
            "examples": {str(b): t for b, t in examples.items()},
        }
        shown = [f"{lo}:{lo + n - 1}={c}" if n > 1 else f"{lo}={c}"
                 for lo, n, c in runs(classes[:105]) if c != "none"]
        report.append(f"{k[0]:#05x}/{k[1]}  {seed_text[:48]:<48}  " + " ".join(shown))

    out = oracle.ROOT / "specs" / arch / "bitflip.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: if stdout is (mistakenly) redirected into this same path, the
    # report goes to the old inode and cannot overwrite the start of the JSON.
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=1))
    tmp.replace(out)
    print("\n".join(report))
    ctrl_changes = sum(1 for e in spec.values() for f in e["fields"]
                       if f["lo"] >= 105 and f["class"] not in ("none",))
    print(f"\n{len(keys)} seeds, {len(batch)} variants -> {out.relative_to(oracle.ROOT)}"
          f"; control-region (105..127) text-affecting runs: {ctrl_changes}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sm_80")
