# Async copy, LDSM and shared memory

## Observed protocol

The validated tensor pipeline uses:

```text
cp.async.cg.shared.global
cp.async.commit_group
cp.async.wait_group
bar.sync
ldmatrix / LDSM
HMMA
```

The compiler models async-copy completion separately from ordinary memory
scoreboards. LDSM consumes a shared-memory address asynchronously, so rewriting
that address register too early creates a read-WAR hazard even when the PTX
source appears sequential.

## Scalar shared widths and barrier issue gap

The [scalar shared-width dossier](../compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory)
adds exact byte/halfword/64-bit loads/stores with four oracle-derived narrow-store
templates and 12,096 varied-word disassembly checks. The first executable A100
gate failed 90 optimized OA 64-bit exchanges while safe OA and same-PTX controls
passed; complete later snapshots showed the expected stored bytes. Retaining
a six-cycle barrier issue gap in ordinary code containing LDS/STS.64 repairs
the tested exchange. This also covers the existing vector-two 32-bit route.
The expanded final gate passes 1,216 launches with guards, raw wide bits and
cross-warp exchanges. It does not establish a universal barrier latency, race
proof or permission to change async-copy/LDSM dependencies.

## Conditional writers and native convergence

The [Python shared-wide cycle](../compiler/gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
found a distinct failure at every scheduler level. Mixed declarations failed
64 of 768 OA launches while same-PTX controls passed. Full scoreboard draining
and a zero-base layout did not repair it. Predicating the conditional writers
and a BSSY/BSYNC ablation each passed. The maintained scalar pass now inserts
one existing B0 pair for forward regions proven to reach the same next CTA BAR.
It preserves outstanding waits, six-cycle convergence issue gaps, label targets
and matched token lifetime. A final expanded 1,280-launch A100 gate passes
nested branches, repeated phases, 32/64-bit elements and 48 KiB boundaries.
The functional model's implicit joins do not independently prove native SIMT
convergence. Cyclic/irreducible flow, memory races and async-copy/LDSM contracts
remain separate.

## Shared layouts

The selected GEMM uses a symmetric XOR shared-memory swizzle. It preserves the
matrix fragment mapping used by LDSM while reducing the tested conflict pattern.
Asymmetric variants, larger stage counts and software fragment caches were
tested separately; they passed some correctness gates but did not establish a
general timing win.

## Copy-address reuse without changing the protocol

The 64×64 production family can derive later copy-chunk addresses from the
first pointer instead of repeating row/column arithmetic. The derivation uses
64-bit additions, preserving carry, and an exhaustive address certificate
checks thread/chunk ownership, shared swizzle and pointer-boundary cases.
It does not remove `commit_group`, `wait_group`, LDSM address-WAR or barrier
dependencies.

The November 11 replay selected copy-address reuse in 20 exact contracts. On
128×512×512 FP16, reuse with direct/early pointers reduced time about 3.8–4.8%
against the previous exact tile64/direct incumbent across selection and
held-out rounds. The much larger 2.237–2.261× ratio over the 128-tile reference
also includes geometry and pointer-policy changes; it is not an isolated
address-reuse effect. See [tensor address generation](../research/ideas/tensor-address-generation.md)
and the [42-contract replay](../../exp/dispatch_tuning/RESULTS.md).

## Barrier and wait findings

The exact-SASS synchronization sweeps show that changing one control field can
be safe in one protocol and wrong in another. For example, tested DEPBAR
threshold mutations failed at some values while higher values passed. A small
passing barrier or reuse probe is not enough to remove a production dependency.

The current evidence and experiments are indexed in
[`docs/evidence/a100-mma`](../evidence/a100-mma/README.md),
[`exp/tensor_scheduler`](../../exp/tensor_scheduler) and the shared-layout
dossier [Shared-memory swizzling and bank conflicts](../research/ideas/shared-layout-swizzling.md).

