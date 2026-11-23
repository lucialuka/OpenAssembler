# Cross-stage scalar PTX contract ledger

Status: validated-not-integrated

G02 requires a machine-readable distinction between syntax admission, actual
lowering and independent correctness evidence. The
[ledger](../../compiler_exp/gap_closure/contracts/ledger.json) enumerates all
**426 exact scalar declarations** from the pre-memory-repair admission schema. Every form
has separate register-backed and parameter-backed fixtures, explicit operand
metadata, stage outcomes and rejection reasons. This experiment changes no
production admission, selector, allocator, encoder or machine table. It is a
retained historical snapshot; the later [memory/predicate repair](gaps-memory.md)
has its own versioned ledger and hardware gates.

## Method and scope

The [generator](../../compiler_exp/gap_closure/contracts/ledger.py) reads
`scalar_forms()` rather than maintaining a second mnemonic list. The
[fixture artifact](../../compiler_exp/gap_closure/contracts/fixtures.json)
retains the exact PTX under each row's `fixture_id`, with a SHA-256 and target
source line. The source/corpus/specification hashes identify the observed
compiler. The gate records separate parser, scalar-admission, instruction
selection, register-allocation, predicate-allocation/lowering, scheduling and
encoding decisions. A failing stage leaves dependent stages `not-run`.

Fixtures use unpredicated parameter or global-load operands, distinct pointer
bindings and declared 16-byte address alignment. Results stay live through
explicit stores. Memory rows record element bits, vector width, access bytes,
space and zero address offset. Shift representatives use literal seven; shuffle
uses delta one, clamp 31 and the full member mask. Shared-load fixtures include
an initializing store and barrier. Control instructions use their own explicit
non-numerical fixture domain.

These are representative operand domains, not exhaustive semantics contracts.
The ledger explicitly marks independent CPU numerical checks, toolkit parsing,
packaging, GPU execution and performance **not-run** for every row. An OA parser
acceptance does not imply that ptxas accepts the same source. An encoding pass
does not prove numerical values, alias/race safety, alignment guards,
collective participation or safe hardware scheduling. No earlier hardware
record is implicitly attributed to a new ledger fixture.

## Observations

On Python 3.12.10 with the recorded compiler/corpus identities:

| Stage | Register-backed fixtures | Parameter-backed fixtures |
|---|---:|---:|
| Parsed | 426 | 426 |
| Scalar admission passed | 426 | 426 |
| Selection passed | 401 | 401 |
| Register and predicate allocation/lowering passed | 401 | 401 |
| Scheduling passed | 401 | 401 |
| Encoding passed | 395 | 387 |

Selection rejects 24 exact vector-two global-load forms (four 32-bit types,
six cache modifiers), plus the representative `mov.pred`. Six register-backed
fixtures select but fail encoding: `shl.b64`, `shr.b64`, `shr.s64`, `shr.u64`,
`sub.s64`, and `sub.u64`. The parameter-backed fixtures additionally expose
constant-bank template restrictions for four `selp` and four floating `setp`
forms. These are exact fixture failures, not a claim that every operand variant
of those forms fails. Other operand variants may still encounter different
restrictions. Passing a less demanding register fixture does not erase its
retained parameter-fixture failure.

The **144-row memory matrix** crosses load/store, global/shared, twelve element
types (`u8/s8/u16/s16/b32/u32/s32/f32/b64/u64/s64/f64`) and scalar/vector-two/vector-four
widths. Its default-cache observations make G12 precise:

| Memory fixture family | Admission/encoding observation |
|---|---|
| Scalar global loads, 8/16/32/64 bits | Twelve type fixtures admit and encode |
| Global vector-two 32-bit loads | Four admit, then selection rejects |
| Global vector-four 32-bit loads | Four admit and encode |
| Scalar and vector-two global 32-bit stores | Eight admit and encode |
| Vector-four global 32-bit stores | Four reject at admission |
| Narrow and scalar-64 global stores | Reject at admission |
| Shared 32-bit scalar/vector-two/vector-four load/store | 24 admit and encode |
| Other shared widths/types and global vector-64 forms | Reject at admission |

There are also **27 boundary controls** for missing memory/ordering/modifier
forms and register, negative, 32 and 63 shift counts. Some controls deliberately
pass admission and reject later. They are labeled boundary controls; they are
not silently converted into positive coverage or assumed to reject at every
stage.

## Reproduction and remaining work

After the bounded atomic repair, use the versioned current snapshot command:

```sh
python compiler_exp/gap_closure/atomics/ledger_snapshot.py
```

Its [current ledger](../../compiler_exp/gap_closure/atomics/ledger-atomics-20261121.json)
contains 524 declarations, all encoding in the register-backed representative
domain, with 516 parameter-backed fixtures encoding. The
[wide arithmetic dossier](gaps-wide.md) records the six repaired encoder failures
and its separate hardware gates; the [atomic dossier](gaps-atomics.md) records
the twelve new exact relaxed-GPU forms. A 168-row
memory matrix additionally includes `b8/b16`. Independent execution evidence
for its generated fixtures remains `not-run`; the separate hardware fixture
identities are documented in the memory dossier. The original ledger and
audit below retain the earlier compiler identity and rejection observations.
Their exact regeneration requires that recorded baseline checkout; running
the old generator against the changed schema would overwrite historical files
and is not the current reproduction route.

```sh
python compiler_exp/gap_closure/contracts/ledger.py
python compiler_exp/gap_closure/contracts/audit.py
```

The [audit](../../compiler_exp/gap_closure/contracts/audit.json) checks all 426
declarations, 1,023 recorded rows, 964 unique fixtures, exact target instructions,
fixture/compiler hashes and stage dependencies. Re-running with process hash
seeds 1 and 777 produced byte-identical ledger JSON. The raw ledger and fixture
artifact are deterministic and may be regenerated without CUDA tools or a GPU.

The retained inventory addresses the evidence-separation part of G02. Full operand
combinations, modifier-order legality, independent numerical references,
packaged modules, toolkit controls, hardware edge cases and performance remain
open. The later memory dossier repairs the named G12 admission/selection
mismatches and `mov.pred` route; those results do not retroactively change this
snapshot's rejection rows. Tensor
instructions retain their separate contract checker and are outside this scalar
ledger.
