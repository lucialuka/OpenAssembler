# Wide scalar arithmetic, shifts and comparisons

Status: implemented

## Gap and proposed lowering

G11 included six admitted forms that selected but did not encode:
`sub.s64/u64`, `shl.b64`, and `shr.b64/s64/u64`. Subtraction required an
unlearned complemented-register template; wide literal shifts required
unrecovered immediate SHF forms. Wider typed Python also exposed conversion
values that aliased mutable source registers instead of preserving snapshots.

The repair uses existing recovered instructions exclusively. Subtraction
explicitly inverts the high word with LOP3 before ordinary IADD3.X. Wide shifts
materialize their count for register SHF forms. Arithmetic right shift across
word 32 combines logical low-word shifting with arithmetic high-word shifting
and selects the appropriate low word. No encoder fields or tables changed.

The maintained scalar contract now includes:

- Signed and unsigned 64-bit subtraction modulo `2^64`, including borrows,
  aliased destinations and predicated old values. Existing addition is retained.
- `shl.b32/b64` and `shr.b32/u32/s32/b64/u64/s64` with literal or register counts.
  Counts are unsigned 32-bit values. Counts at least the width produce zero,
  except signed right shifts, which sign-fill. Negative literals still reject.
- `setp.eq/ne/lt/le/gt/ge.s64/u64` and unsigned `lo/ls/hi/hs.u64`. Ordered
  comparisons use signed or unsigned high words and an unsigned low-word tie
  break. Predicate homes preserve guarded assignments and source snapshots.
- Existing integer conversions across signed/unsigned 32/64-bit types now
  preserve values when a source register changes after a conversion.

These are typed device integer semantics, not unbounded Python arithmetic.
FP64 arithmetic, wide multiply/divide/remainder, saturation/carry chains and
arbitrary PTX modifier combinations remain separate work.

## Experiments and independent validation

The [isolated candidate](../../compiler_exp/gap_closure/scalar_wide/candidate.py)
and its generated modules were validated before production promotion. The
[reference corpus](../../compiler_exp/gap_closure/scalar_wide/cases.py) contains
signed/unsigned boundaries, low-word borrow cases, fixed-seed random words,
counts 0 through 65 and large counts through `0xffffffff`. Every source
updates an aliased destination under a predicate, retains old values on false
lanes and leaves inactive tail lanes untouched. Conversion controls mutate
their original source after taking a snapshot.

The [baseline](../../compiler_exp/gap_closure/scalar_wide/baseline-v2-20261121.json)
retains 76 passing cases, 16 wrong-result conversion cases and 416 rejections
across four optimization levels. The first candidate's eight missing NE
predicate-template rejections remain in its earlier record; the final
implementation composes existing AND comparisons and PLOP instead. The
[final candidate](../../compiler_exp/gap_closure/scalar_wide/candidate-v2-20261121.json)
passes all 508 encoded cases under both ordinary and poisoned-register
execution. All 127 fixtures independently match the
[PTX interpreter](../../compiler_exp/gap_closure/scalar_wide/ptx-reference-20261121.json).

The [A100 report](../../compiler_exp/gap_closure/scalar_wide/a100-20261121T131238Z.json)
passes **1,016 launches**: 127 fixtures × four OA optimization levels × OA and
same-PTX ptxas controls. It identifies Python 3.12.1, CUDA toolkit 12.8.1 and
NVIDIA A100-SXM4-40GB. Each launch uses 576 threads for 527 valid lanes. Every
output byte, guard, preserved lane and input immutability check passes.
Independent references use integer bit arithmetic without SASS emulation or
floating tolerance. The archive retains exact PTX, complete input/expected/output
bytes, manifests, cubins and SASS. These are correctness results, with no timing
or generalized architecture claim.

The [promotion audit](../../compiler_exp/gap_closure/scalar_wide/promotion-audit-20261121.json)
checks all 9,883,648 retained output bytes and exact promoted backend identities.
The toolkit logs retain a cosmetic disassembly spelling difference (`0x0f`
versus `0xf`) for LOP3's inversion LUT; the words and all outputs pass. Production
imports no experiment modules.

## Reproduction and remaining coverage

```sh
.venv/bin/python compiler_exp/gap_closure/scalar_wide/offline.py /tmp/oa-wide.json
.venv/bin/python compiler_exp/gap_closure/scalar_wide/ptx_gate.py
.venv/bin/python -m pytest -q tests/test_backend_wide.py
modal run compiler_exp/gap_closure/scalar_wide/run_modal.py
.venv/bin/python compiler_exp/gap_closure/scalar_wide/validate.py compiler_exp/gap_closure/scalar_wide/a100-20261121T131238Z.json /tmp/oa-wide-audit.json
.venv/bin/python compiler_exp/gap_closure/scalar_wide/ledger_snapshot.py
```

The hardware command stages the retained candidate bytes matching promoted
source hashes. The source generator reads retained baseline modules. The
[regression fixtures](../../tests/data/scalar_wide_fixtures.json) remain independent
of experiment imports; tests compute expected values directly from input bits.
Additional regression controls encode and execute every literal shift from 0
through 63, and keep negative counts and unsupported FP64/saturating forms
rejected.

The [new ledger](../../compiler_exp/gap_closure/scalar_wide/ledger-wide-20261121.json)
keeps all older snapshots intact. All 512 declarations now encode in its
register-backed representative domain. Parameter-backed fixtures encode 504;
eight existing constant-bank selection/comparison restrictions remain. Each
generated ledger fixture still records execution/toolkit/hardware/performance
as `not-run`: the hardware corpus above has separate exact source identities.
Closing the six recorded failures does not establish all operand combinations,
complete ISA conformance, general wide arithmetic or a new GPU target.

## Typed verification through the Python build route

Wide Python entries publish `openassembler-pyfront-types-v1` JSON comments with
schema version, exact entry name and ordered parameter names, scalar/array
roles and canonical PTX element types. The
[generic verifier](../../openassembler/verify.py) validates that declaration
against the parsed parameter ABI before allocating data. It distinguishes a
`u64` scalar from a pointer, allocates `f32/s32/u32/s64/u64` arrays at their real
element width and preserves complete 64-bit scalar bits in both execution
engines. Untagged PTX retains the existing pointer heuristic. Invalid, duplicate,
unknown-entry or mismatched metadata cannot silently select that legacy path.

The typed route runs ordinary and poisoned-register SASS against the PTX
interpreter and compares every element, including upper words. Both engines'
adjacent 64-byte guards are checked against their original values, so identical
out-of-bounds writes in both engines still fail. Arrays use deterministic edge
values plus seeded data; common dimension names remain bounded, while other
integer scalar probes include full-width boundary bits. This does not infer
arbitrary shape, stride or runtime scalar validity constraints, and it is not
general memory-safety proof.

The [typed-verifier experiment](../../compiler_exp/gap_closure/python_wide/verification/typed-verifier-20261121.json)
retains the old checker rejecting an explicit contract and treating an unsigned
64-bit data scalar as a pointer. The new route passes 20 type/seed cases and
rejects deliberate upper-word corruption and matching guard writes. The
[regressions](../../tests/test_verify_typed.py) additionally reject malformed
metadata; the typed and existing generic-verifier tests pass 33 cases. These
are CPU checker results; the Python frontend's hardware gates are separate.

```sh
.venv/bin/python compiler_exp/gap_closure/python_wide/verification/experiment.py
.venv/bin/python -m pytest -q tests/test_verify_typed.py tests/test_verify.py
```
