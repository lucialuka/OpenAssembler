# Native static per-thread local storage

Status: implemented for the stated scalar local-frame contract; G13 remains partial

G13 includes per-thread local arrays and their executable metadata, alongside
module global/constant placement and relocations. The selected implementation adds
32-bit local load/store with static declarations and a private per-thread frame.
Generic addressing, module symbols/relocations and general device stacks retain
their original exit gates.

The [baseline probe](../../compiler_exp/gap_closure/local_storage/local.ptx)
initializes eight local slots, reads a thread-dependent slot and stores its
result globally. The pre-feature production compiler rejected the local symbol on its first address use.
The [oracle record](../../compiler_exp/gap_closure/local_storage/oracle-20261123.json)
retains CUDA toolkit compilation at `-O0` and `-O3`, complete SASS and `.nv.info`
records. At `-O0` the tool reserves 32 bytes, subtracts that amount from the
ABI stack pointer and emits local accesses. At `-O3` it eliminates the local
array and declares a zero-byte frame. Reusing that optimized donor metadata for
a new native local implementation would therefore be incorrect.

The [layout sweep](../../compiler_exp/gap_closure/local_storage/layout-sweep-20261123.json)
checks 32/48/64/128-byte declarations with alignments 4/16/32. Alignments through
16 use matching frame/subtraction sizes in this corpus. A 64-byte, 32-aligned
array instead has an 80-byte declared frame, a 64-byte stack decrement and an
explicit 32-byte alignment mask. Frame reservation and pointer arithmetic are
distinct contracts; these observations do not justify guessing a general rule.
The [artifact archive](../../compiler_exp/gap_closure/local_storage/oracle-artifacts-20261123.zip)
and [identity](../../compiler_exp/gap_closure/local_storage/archive-20261123.json)
retain all source, donor cubins, SASS and metadata, including ignored raw cubins.

## Validated solution and production contract

The [candidate](../../compiler_exp/gap_closure/local_storage/candidate.py)
retains the exact pre-feature sources and stages selection, admission, build
packaging, local layout/metadata and ownership independently. It assigns static
symbols nonoverlapping offsets at their declared alignment and rounds the frame
extent to 16 bytes. Sizes must be positive, alignments one of 1/2/4/8/16, and the
complete frame at most 65,536 bytes. The entry prologue subtracts that frame
from the ABI stack pointer. Local symbol addresses explicitly materialize as
`mov.u32`/`mov.b32`; local pointers are 32-bit byte addresses.

`ld.local` and `st.local` admit scalar `b32/u32/s32/f32`, using existing recovered
LDL/STL templates. False predicates preserve the old destination and storage.
Constant-bank/immediate stored values use the ordinary register route. Literal
symbol offsets must be naturally aligned and stay within their declaration.
Dynamic pointer offsets require caller-provided alignment, initialization and
bounds. No device bounds check, local initialization proof or arbitrary pointer
provenance analysis is implied. Vector/narrow/64-bit local access, volatile/
ordered forms, larger alignments, generic pointers and local `cvta` reject.

Both observed per-function `(symbol index, reserved bytes)` metadata records
are patched to OA's frame size after native text replacement: FRAME_SIZE 0x12
and its retained 0x11 companion. The code keeps the latter's numeric identity
explicit rather than inventing a broader attribute interpretation. Missing,
duplicate or malformed record pairs reject. Symbol indices, other functions'
records and unrelated metadata are retained. The builder uses its own frame
layout even when the donor optimized local storage away. Manifest and native
ownership bind `local_bytes` to the cubin metadata, and ordinary replay rejects
inconsistent declarations.

Production [layout/metadata](../../openassembler/backend/storage.py),
[selector](../../openassembler/backend/ra.py), [build packaging](../../openassembler/build.py)
and [ownership validation](../../openassembler/native_ownership.py) import no
experiment modules. Static local arrays provide the independently validated
storage foundation for the separate [scalar/group/predicate spilling route](gaps-spilling.md).

## Independent experiments and hardware

The [offline gate](../../compiler_exp/gap_closure/local_storage/candidate-offline.json)
passes 44 fixture/optimization combinations against an independent integer
reference, the PTX interpreter and normal/poisoned SASS. Eleven fixtures cover
all four element spellings, symbol/pointer accesses, distinct padded local
objects, predicated loads/stores, loop overwrites, tails and the maximum frame.
Every thread retains its private local memory. The
[focused tests](../../compiler_exp/gap_closure/local_storage/production-tests-final.log)
pass 88 local-storage/ownership/ordinary CLI checks, including malformed layout,
metadata, widths, modifiers and literal-bound rejections.

The [final A100 report](../../compiler_exp/gap_closure/local_storage/candidate-a100-20261123T093837Z.json)
passes **704 launches** on A100-SXM4-40GB, Python 3.12.1 and CUDA toolkit 12.8.1:
eleven fixtures × four OA optimization levels × OA/same-PTX ptxas controls ×
four active lengths (0/1/259/384) × two changed input seeds. Three CTAs each
launch 128 threads. Every output byte, input immutability, guard and inactive
tail passes. OA reserves 48/64/80/65,536-byte frames; driver resource reports
match those metadata sizes. Ptxas controls use zero-byte frames for ten fixtures
and 65,520 bytes for the maximum-layout fixture. Controls are compared by full
outputs, without requiring their internal frame layout to equal OA's.

The initial [512-launch gate](../../compiler_exp/gap_closure/local_storage/candidate-a100-20261123T093539Z.json)
predates the added layout domains. The first larger run stopped on a harness
assumption that every optimized control frame was zero. Its
[log](../../compiler_exp/gap_closure/local_storage/hardware-final.log) is retained;
no complete report/archive was retrieved from that exception, so it is not
counted as a hardware gate. The corrected harness records actual control frames
and still enforces OA's exact reservation and numerical/guard/resource checks.

The [promotion audit](../../compiler_exp/gap_closure/local_storage/promotion-audit.json)
rechecks **2,883,584 retained bytes**, every binary/source identity, manifest and
driver frame agreement, corrupted output rejection and 44 exact current native
word/register comparisons. The exact five candidate source modules were then
promoted. The [new policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T094137Z.json)
retains all 49 policies across 98 original native sessions; the added source
module participates in current identities and is absent from frozen identities
by the recorded Git inventory. The
[application audit](../../compiler_exp/gap_closure/local_storage/application-continuity.json)
checks 1,298 prior shuffle/atomic/decode/checkpoint kernel occurrences against
current generation, retaining their original numerical/source-exact proofs.

The [553-form ledger](../../compiler_exp/gap_closure/local_storage/ledger-local-20261123.json)
passes representative admission-through-encoding gates. Its independent
semantics/toolkit/packaging/hardware/timing stages remain `not-run`, since its
fixtures differ from the hardware corpus. An initial fixture mistakenly used
an unsupported local vector store to initialize scalar-load probes; its four
recorded rejections remain retained. Scalar initialization repairs the fixture
without broadening production admission.

The later [direct function specialization](gaps-device-functions.md) assigns
fresh static local objects to each expanded call and validates their total
frame on hardware. Spill-slot reuse, generic pointers, native stack growth,
module constants/globals and linking remain separate original exit gates. No
performance claim is made.

```sh
python compiler_exp/gap_closure/local_storage/oracle.py
python compiler_exp/gap_closure/local_storage/layout_sweep.py
```

The first commands query the toolkit. Candidate CPU/hardware/audit reproduction:

```sh
python compiler_exp/gap_closure/local_storage/offline.py
python -m pytest -q tests/test_backend_local_storage.py
modal run compiler_exp/gap_closure/local_storage/run_modal.py
python compiler_exp/gap_closure/local_storage/audit.py compiler_exp/gap_closure/local_storage/candidate-a100-20261123T093837Z.json /tmp/oa-local-audit.json
```
