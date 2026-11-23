# Global memory widths and predicate assignment

Status: implemented

## Gap and numerical/effect contract

G12's earlier admission/selection mismatch blocked vector-two global loads and
vector-four stores despite existing recovered SM80 instructions. Scalar global
byte, halfword and 64-bit stores also lacked a maintained lowering. G11's
admitted `mov.pred` reached generic register MOV and rejected; predicate
reassignment across guards and branches additionally lacked fixed predicate
homes. Guarded vector loads did not count all destination definitions.

The repaired global-memory matrix is:

| Operation | Element types | Required address alignment | Native instruction |
|---|---|---:|---|
| Vector-two load | `b32/u32/s32/f32` | 8 bytes | `LDG.E.64` |
| Vector-four store | `b32/u32/s32/f32` | 16 bytes | `STG.E.128` |
| Scalar byte store | `b8/u8/s8` | 1 byte | `STG.E.U8` |
| Scalar halfword store | `b16/u16/s16` | 2 bytes | `STG.E.U16` |
| Scalar 64-bit store | `b64/u64/s64/f64` | 8 bytes | `STG.E.64` |

Stores copy exact low-order bytes; signedness does not change those bytes.
The 64-bit floating route copies bits without introducing FP64 arithmetic.
Register values and integer/IEEE binary64 immediates have explicit low/high-word
materialization; a binary32 float literal used as a 64-bit source rejects with
`NotSupported` rather than an internal exception.
Existing load cache modifiers (`nc/cg/cs/lu/ca`) and store cache modifiers
(`wb/cg/cs/wt`) retain the same value contract and may lose their performance
hint. Ordering/scope/volatile semantics are still rejected. Alignment, valid
addresses and sufficient allocation lengths remain caller invariants; this is
not a new runtime bounds, race or alignment checker. Shared narrow/64-bit
accesses and vectors of 64-bit elements remain outside this change.

Predicate MOV and logical/setp assignments preserve snapshots, false-guard old
values, branch joins, loop-carried values and self-guarded writes. Freshly
computed predicates write directly into a fixed home when needed. Vector load
destinations participate in the existing fixed-home analysis, so an inactive
load preserves every old vector element. The allocator still rejects excessive
predicate/register pressure; this work does not add spilling.
Packed and unpacked 64-bit MOV values also preserve snapshots when a scalar or
wide source is reassigned later; mutable home components are copied before
renaming or destination updates.

## Candidate, validation and promotion

The isolated [candidate generator](../../compiler_exp/gap_closure/memory_widths/candidate.py)
started from recorded baseline hashes and generated separate candidate modules.
Production code remained unchanged while the
[experiment](../../compiler_exp/gap_closure/memory_widths/offline.py) encoded and
executed all fixtures against independent byte-level expected buffers. The
[baseline](../../compiler_exp/gap_closure/memory_widths/baseline-fixture-v2-20261121.json)
rejects all 23 cases across four optimization levels. The
[initial candidate](../../compiler_exp/gap_closure/memory_widths/candidate-direct-predicate-20261121.json)
passes its 92 encoded cases under normal and poisoned-register execution.
Independent review then found two untested pack/unpack aliasing failures and
an uncaught binary64 immediate operand exception. The
[snapshot controls before repair](../../compiler_exp/gap_closure/memory_widths/snapshot-before-20261121.json)
retain eight wrong-result failures across four levels; the
[immediate controls before repair](../../compiler_exp/gap_closure/memory_widths/immediate-before-20261121.json)
retain eight operand exceptions. Separate
[snapshot](../../compiler_exp/gap_closure/memory_widths/candidate_snapshot.py) and
[immediate](../../compiler_exp/gap_closure/memory_widths/candidate_immediates.py)
candidates fix those concrete defects. The
[final candidate](../../compiler_exp/gap_closure/memory_widths/immediate-candidate-20261121.json)
passes all 112 encoded cases under normal and poisoned-register execution.

The [final A100 report](../../compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json)
records **224 launches**: 28 cases × four OA optimization levels × OA and
same-PTX ptxas controls. Python is 3.12.1, CUDA toolkit 12.8.1 and the actual
device is NVIDIA A100-SXM4-40GB. Every output byte, both outer guards, inactive
lanes, narrow-store neighbors and input immutability pass. Each kernel launches
96 threads for 67 valid lanes. Input words include signed/unsigned boundaries,
infinity/NaN bit patterns and fixed-seed random words. No floating arithmetic
tolerance is involved. The report's archive retains combined source, input and
expected buffers, complete outputs, manifests, cubins and SASS; the source and
binary hashes identify each implementation. There is no performance claim.

The earlier [copy-based predicate candidate](../../compiler_exp/gap_closure/memory_widths/a100-20261121T122940Z.json)
also passes. The selected candidate computes directly into predicate homes,
avoiding its extra predicate copies. Production
[selector](../../openassembler/backend/ra.py) and
[admission schema](../../openassembler/backend/contracts.py) were promoted only
after their exact hashes matched the final hardware report.
The [promotion audit](../../compiler_exp/gap_closure/memory_widths/final-promotion-audit-20261121.json)
checks all 458,752 retained output bytes, binary identities and the exact
promoted backend hashes. Earlier successful reports retain their narrower
fixture sets and earlier backend identities.

The first toolkit attempt rejected a fixture's incorrectly typed `f64` source
register; the corrected fixture declares a raw `b64` register. The rejected
[toolkit log](../../compiler_exp/gap_closure/memory_widths/a100-candidate-20261121.log)
and both original offline snapshots remain retained. They must not be counted
as successful toolkit validation. A separate
[harness correction](../../compiler_exp/gap_closure/memory_widths/harness-correction.json)
records an aborted manifest-indexing attempt before hardware results.

## Reproduction and remaining boundaries

```sh
.venv/bin/python compiler_exp/gap_closure/memory_widths/offline.py /tmp/oa-memory-validation.json
.venv/bin/python -m pytest -q tests/test_backend_memory_widths.py tests/test_backend_contracts.py tests/test_backend_loops.py tests/test_backend_ops.py tests/test_verify.py
modal run compiler_exp/gap_closure/memory_widths/run_modal.py
.venv/bin/python compiler_exp/gap_closure/memory_widths/ledger_snapshot.py
.venv/bin/python compiler_exp/gap_closure/memory_widths/validate.py compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json /tmp/oa-memory-audit.json
```

The hardware command uses the retained exact candidate modules, matching the
promoted source hashes. The generator records baseline anchors and is intended
for the original baseline checkout; rerunning it on the promoted source rejects
anchor drift. The later snapshot/immediate candidate constructors read retained
prior candidate sources, keeping each repair's baseline available. The
[regression fixtures](../../tests/data/backend_memory_widths.json) are retained
independently of experiment imports. Offline regression compares every byte,
including false-guard and tail bytes. Negative tests keep unsupported widths,
memory ordering and malformed vectors rejected. The relevant suite passes
327 backend and Python frontend tests locally.

The [new scalar ledger](../../compiler_exp/gap_closure/memory_widths/ledger-memory-20261121-v3.json)
retains a new version and leaves the original 426-declaration snapshot intact.
All 496 current declarations select in its representative domains; 490
register-backed and 482 parameter-backed fixtures encode. Its independent
CPU/toolkit/packaging/hardware/performance stages remain explicitly `not-run`:
the 28 hardware fixtures above have separate exact source identities. This
repairs the named mismatches and expands G12's bounded support. Wider shared
memory, arbitrary operand/alignment domains, ordered memory, general G11
arithmetic and complete G02 semantic conformance remain open.
