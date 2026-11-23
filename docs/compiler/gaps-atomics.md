# Native GPU-relaxed 32-bit atomics and reductions

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
Acquire/release/acq_rel semantics, CTA/system scope, shared memory, 64-bit and
floating atomics still reject. Their modifiers are never silently dropped.
This is a bounded G14 implementation and does not establish the full memory
model, general atomics or Python GPU primitive coverage.

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

The hardware command stages the retained candidate source matching promoted
backend bytes. A fresh run creates a dated record; it does not replace prior
hardware evidence. The
[versioned scalar ledger](../../compiler_exp/gap_closure/atomics/ledger-atomics-20261121.json)
contains 524 declarations: all encode in its representative register domain,
and 516 encode with parameter-backed operands. The eight earlier constant-bank
restrictions remain. Independent execution/toolkit/hardware/performance stages
for those generated ledger fixtures remain `not-run`; this hardware study has
separate exact sources. Broader memory ordering, shared/64-bit/FP atomics,
mixed atomic/non-atomic races and multi-device coherence remain open.
