# SM80 SASS instruction encoding

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Translate assembly instructions into the exact machine words consumed by A100. Instead of assuming undocumented bit positions, vary inputs to NVIDIA's assembler/disassembler, compare the resulting words and test the recovered fields.

## Findings and implementation decision

The production encoder emits the supported SM80 instructions from generated templates and field maps. The original corpus contained 541 PTX probes, 13,816 instructions and 88 observed opcodes; a 137-seed × 128-bit sweep identified candidate fields. Common controls include stall/yield, dependency scoreboards and operand reuse. Successful decoding is not evidence that a schedule is safe or fast. Unsupported forms are refused rather than guessed.

The production API also rejects malformed field values before bit packing:
registers, predicates, constant-bank fields, memory bases, branch displacements,
barrier indices and control-field widths are range checked. This closes the
silent-truncation path where an out-of-range request could encode a different
register or control value. The checks are correctness hardening, not a claimed
performance gain.

One important correction was a hidden ISETP immediate fragment: disassembly omitted bits that still affected execution. Template minimization now preserves such bits from the observed words.

## Method and validity checks

Build lane-varying PTX probes, capture original words, mutate one bit at a time, disassemble every variant and classify changes. Re-encode text and controls and compare bytes/text. Hardware replay is a separate gate. A round-trip tool can retain original words for unsupported instructions; its partial cases must not be counted as fully generated kernels.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_word.py tests/test_backend.py
PYTHONPATH=. .venv/bin/python tools/build_corpus.py sm_80
PYTHONPATH=. .venv/bin/python tools/bitflip.py sm_80
PYTHONPATH=. .venv/bin/python tools/reencode_cubin.py sm_80
```
The corpus-build and bit-flip commands regenerate ignored artifacts and spec research outputs; run them in an experiment checkout when preserving a frozen release.

## Code and retained evidence

- [Encoder](../../../openassembler/isa/encode.py)
- [Control-word decoder](../../../openassembler/isa/word.py)
- [Committed probe words](../../../corpus/sm_80/pairs.jsonl)
- [Recovered field data](../../../specs/sm_80/bitflip.json)
- [Encoder tests](../../../tests/test_word.py)

## What remains open

Full instruction/modifier coverage and every later architecture need new probes and hardware validation. Text equivalence can hide meaningful bits; never infer semantic equivalence from a printed mnemonic alone.
