# SASS encoding and control fields

## Corpus scale

The retained SM80 probe corpus contains 541 standalone PTX forms, 13,816
observed instructions and 88 opcode families. A 137-seed bit-flip campaign
mutated 128-bit words to identify candidate fields. The corpus is broader than
the production backend; unsupported forms remain evidence, not silently enabled
features.

## Recovered field classes

The observed SM80 word layout includes fields for:

* destination and source registers, including aligned vector groups;
* predicates and predication polarity;
* constant-bank operands and immediates;
* memory bases, offsets and address modes;
* branch targets and displacement encoding;
* control bundles containing stall/yield, read/write scoreboards and reuse;
* barrier identifiers and synchronization operands;
* opcode/modifier-specific fields whose meaning cannot be inferred globally.

The encoder uses templates and field maps instead of a universal guessed bit
layout. Range checks reject malformed values before packing. This closed a
silent truncation class where an out-of-range register or control value could
encode a different valid instruction.

## Hidden-bit correction

One ISETP immediate fragment showed that `nvdisasm` omitted bits that still
affected execution. Template minimization therefore preserves observed bits
that are not visible in the printed mnemonic. This is why re-encoding only the
displayed text is insufficient for exact word recovery.

## Validation gates

An encoding change is accepted only when:

1. the PTX parser accepts the source;
2. the emulator produces the expected result;
3. the encoder produces valid words;
4. `nvdisasm` can disassemble the container;
5. exact words/text match where the form is claimed supported; and
6. hardware output and canary checks pass when the form is executable.

The production dossier is [SM80 SASS instruction encoding](../research/ideas/sass-encoding.md). The raw field data are under
[`specs/sm_80`](../../specs/sm_80) and [`corpus/sm_80`](../../corpus/sm_80).

