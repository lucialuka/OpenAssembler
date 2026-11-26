# Native 32-bit atomics, reductions and bounded ordering

Status: implemented

## Gap and exact contract

G14 lacked a maintained PTX-to-native atomic route. Recovered SM80 templates,
SASS emulation and PTX reference behavior already existed; admission and
selection rejected the operations. This repair connects those existing layers
for these exact forms:

| Family | Operation/type | Result |
|---|---|---|
| `atom` | `add.u32`, `add.s32` | Previous 32-bit value; update wraps modulo `2^32` |
| `atom` | `exch.b32` | Previous bits, replacing them with the supplied bits |
| `atom` | `cas.b32` | Previous bits; replace only when they equal the comparison bits |
| `red` | `add.u32`, `add.s32` | Atomic update without a returned value |

Each form admits the default `global` spelling and explicit
`relaxed.gpu.global` spelling. This is GPU-scope, relaxed memory ordering;
relaxed atomics do not order unrelated data accesses. NVIDIA's
[PTX atom contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
defines omitted semantics/scope as relaxed/GPU. The observed source identity and
relevant qualifiers are retained in the
[reference record](../../compiler_exp/gap_closure/atomics/ptx-reference-source-v2.json).

Addresses must point to valid naturally aligned four-byte global storage.
Register/constant-bank/immediate integer operands materialize through the
ordinary register route; CAS preserves comparison/new-value operand order.
False predicates preserve both memory and the old destination register.
Address offsets divisible by four are admitted; caller alignment and bounds
remain explicit preconditions. The new operations remain side effects even
when their returned values are dead. Scheduling preserves the existing ordered
memory sequence and asynchronous register dependencies.

The native instructions are recovered `ATOMG.E.ADD/EXCH/CAS.STRONG.GPU` and
`RED.E.ADD.STRONG.GPU`. No encoder field or table was invented or changed.
The November 23 extension below adds specific GPU ordered and CTA shared forms.
Other scope/type/operation combinations still reject; modifiers are never
silently dropped.
This is a bounded G14 implementation and does not establish the full memory
model or general atomics. The separate
[Python atomic primitives](gaps-python-primitives.md) now expose bounded effects.

## GPU acquire/release and CTA shared extension

G14 also needs faithful visibility and shared-memory lowering. The admitted
global `atom` forms now accept explicit `acquire.gpu.global`,
`release.gpu.global` and `acq_rel.gpu.global`, for `add.u32/s32`, `exch.b32`
and `cas.b32`. Shared `atom.shared` and `atom.relaxed.cta.shared` accept
`add.u32/s32` and `exch.b32`. Shared addresses are 32-bit byte offsets or known
module shared symbols, with four-byte-aligned literal offsets. Storage must be
initialized before use and callers must provide the required CTA synchronization.
These are six exact shared and twelve new ordered global forms.

The [retained toolkit observations](../../compiler_exp/gap_closure/atomic_ordering/frozen-oracle-20261121.json)
show the already recovered atomic/fence instructions. The candidate uses
`MEMBAR.ALL.GPU` followed by `ERRBAR` before release, `CCTL.IVALL` after acquire,
and recovered `ATOMS.ADD/EXCH` for shared operations. Scheduling preserves their
memory order and drains pending asynchronous scoreboards at fences. Independent
ALU instructions can move between them. No encoding field or generated table
was guessed or changed. System/global CTA scopes, shared acquire/release,
shared CAS/reductions, 64-bit/FP atomics and ordered ordinary loads/stores remain
unsupported.

The first [hardware failure](../../compiler_exp/gap_closure/atomic_ordering/a100-20261121T142833Z.json)
is retained: the publication kernel encountered an illegal address. A minimal
[branch counterexample](../../compiler_exp/gap_closure/atomic_ordering/pointer-branch-20261123.json)
isolates an existing selector defect: lazily materialized parameter registers
were reused on paths that skipped their definition. Keeping symbolic wide
values across control-flow uses repairs it. Additional
[wide-product lifetime testing](../../compiler_exp/gap_closure/atomic_ordering/wide-snapshot-before-20261123.json)
found that a deferred multiply could consume a mutable operand after its PTX
definition. Taking definition-time snapshots repairs the
[counterexample](../../compiler_exp/gap_closure/atomic_ordering/wide-snapshot-after-20261123.json).
Both defects fail all four original compiler levels and now have independent
bit-exact regressions; their repair is part of this production promotion.

Two final source-matched hardware gates pass:

- The [publication/shared/value gate](../../compiler_exp/gap_closure/atomic_ordering/a100-20261123T080159Z.json)
  passes **472 launches** on A100-SXM4-40GB. Two CTAs exchange a 16-word payload
  through flag/ack handshakes for 128 epochs, with pre-acquire payload reads,
  bounded polling, exact final state and stale-epoch checks. Acquire/release and
  acq_rel variants each repeat 16 times per compiler/level. Shared tests use four
  CTAs, four warps per CTA, 17 bins, predicated participants, complete old-value
  invariants, signed overflow and symbol-plus-offset addressing. The two pointer
  and wide-product regressions also execute on hardware.
- The [complete form gate](../../compiler_exp/gap_closure/atomic_ordering/forms-a100-20261123T081214Z.json)
  passes **768 launches** on A100 80GB PCIe. All twelve ordered forms and twelve
  existing relaxed controls run at four OA levels beside same-PTX ptxas, with
  independent-address and repeated 33-CTA contention domains. Every state byte,
  preserved result, old-value ticket/CAS winner/connected exchange trail,
  allocation guard, inactive tail and immutable input passes.

Both use Python 3.12.1 and CUDA toolkit 12.8.1. Their source hashes match the
four promoted backend modules. The
[publication audit](../../compiler_exp/gap_closure/atomic_ordering/promotion-audit-20261123.json)
and [form audit](../../compiler_exp/gap_closure/atomic_ordering/promotion-forms-audit-20261123.json)
recheck **44,465,024 retained bytes**, binary identities and 140 current-versus-
archived native word/register comparisons. Deliberately corrupted outputs reject.
The first client's connection reset occurred after its complete successful
report/archive were saved; the read-only audit verifies those artifacts.
Earlier candidate successes and the failed initial gate remain separate records.
No memory-model exhaustiveness or timing conclusion follows from these litmus
and contention domains.

Production [regressions](../../tests/test_backend_atomic_ordering.py) preserve
shared state, control flow, definition-time snapshots, fence order and acquire
scoreboard waits. The [542-form ledger](../../compiler_exp/gap_closure/atomic_ordering/ledger-ordering-20261123.json)
records representative parse-through-encoding results for every declaration.
Its generated fixtures still mark independent semantics, toolkit, packaging,
hardware and timing `not-run`; the executed hardware fixtures above are distinct.

```sh
python -m pytest -q tests/test_backend_atomics.py tests/test_backend_atomic_ordering.py
modal run compiler_exp/gap_closure/atomic_ordering/run_modal.py
modal run compiler_exp/gap_closure/atomic_ordering/run_forms_modal.py
python compiler_exp/gap_closure/atomic_ordering/audit.py compiler_exp/gap_closure/atomic_ordering/a100-20261123T080159Z.json /tmp/oa-ordering-audit.json
python compiler_exp/gap_closure/atomic_ordering/ledger_snapshot.py
```

The following sections retain the earlier relaxed-only campaign and its original
counts. They are not additional final-candidate replications.

## Experiments and correctness gates

The [candidate](../../compiler_exp/gap_closure/atomics/candidate.py) was generated
from retained baseline selector/admission sources and validated separately.
The [baseline](../../compiler_exp/gap_closure/atomics/baseline-20261121.json)
rejects all 96 fixture/geometry/optimization combinations. The
[candidate](../../compiler_exp/gap_closure/atomics/candidate-20261121.json)
passes 96 combinations under normal/poisoned-register SASS and the PTX
interpreter. Sequential CTA simulation is an offline semantic gate; it cannot
establish concurrent atomic behavior.

The [A100 report](../../compiler_exp/gap_closure/atomics/a100-20261121T140843Z.json)
passes **384 launches**: 12 exact forms × four OA optimization levels × OA and
same-PTX ptxas controls × one independent-address and three contended runs.
Python is 3.12.1, toolkit CUDA 12.8.1, and the device NVIDIA A100-SXM4-40GB.
Each launch uses 33 CTAs of 128 threads, 4,099 valid lanes, and an even-lane
participation predicate. Contended runs target 17 shared bins across CTAs.
Guards, inactive lanes, complete state, old-value returns and input immutability
pass. Signed/unsigned boundary patterns include overflow/wrap cases.

Independent references do not assume hardware thread order. Addition checks
the entire histogram and consecutive old-value tickets per bin. CAS checks one
successful winner and every returned old value. Exchange requires one complete
directed trail from the initial value to the final value, using every
returned-old → offered-new update edge. Independent-address cases additionally
check each exact result and conditional CAS success/failure.

Independent review found that the initial exchange multiset check alone could
admit a disconnected return cycle. The stronger checker was applied to every
retained hardware output; all pass. The
[audit](../../compiler_exp/gap_closure/atomics/promotion-audit-v2-20261121.json)
checks all **21,697,152 retained bytes**, binary hashes and all launch domains;
its explicit negative controls reject histogram corruption, bad old-value
tickets and a disconnected exchange cycle with a conserved value multiset.
The original fixture/checker source and earlier audit remain retained. This
reference strengthening did not change the executed candidate or promoted
backend identities. The archive retains all source, initial/output buffers,
manifests, cubins and OA listings. These are correctness results, with no timing
or ordering-litmus claim.

## Production and reproduction

Production [selection](../../openassembler/backend/ra.py) and
[admission](../../openassembler/backend/contracts.py) were promoted after the
exact candidate hashes matched the A100 report. The
[regressions](../../tests/test_backend_atomics.py) cover 109 tests, including
dead returned values, discarded reduction results and negative scope/order/type
controls. The broad backend/memory/wide/verifier regression run passes 914
tests; the two dead-result controls were then added and the complete atomic
suite rerun. No production module imports an experiment driver.

```sh
.venv/bin/python -m pytest -q tests/test_backend_atomics.py
.venv/bin/python compiler_exp/gap_closure/atomics/offline.py /tmp/oa-atomics.json
modal run compiler_exp/gap_closure/atomics/run_modal.py
.venv/bin/python compiler_exp/gap_closure/atomics/validate.py compiler_exp/gap_closure/atomics/a100-20261121T140843Z.json /tmp/oa-atomics-audit.json
.venv/bin/python compiler_exp/gap_closure/atomics/ledger_snapshot.py
```

These commands reproduce the retained November 21 candidate. Its source-identity
audit requires that recorded checkout; the current backend includes the
separately tested extension above. A fresh hardware run creates a dated record
and does not replace prior evidence. The
[versioned scalar ledger](../../compiler_exp/gap_closure/atomics/ledger-atomics-20261121.json)
contains 524 declarations: all encode in its representative register domain,
and 516 encode with parameter-backed operands. The eight earlier constant-bank
restrictions remain. Independent execution/toolkit/hardware/performance stages
for those generated ledger fixtures remain `not-run`; this hardware study has
separate exact sources. Other memory-ordering forms, shared CAS/64-bit/FP atomics,
mixed atomic/non-atomic races and multi-device coherence remain open.
