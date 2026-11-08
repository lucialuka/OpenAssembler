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

## Shared layouts

The selected GEMM uses a symmetric XOR shared-memory swizzle. It preserves the
matrix fragment mapping used by LDSM while reducing the tested conflict pattern.
Asymmetric variants, larger stage counts and software fragment caches were
tested separately; they passed some correctness gates but did not establish a
general timing win.

## Barrier and wait findings

The exact-SASS synchronization sweeps show that changing one control field can
be safe in one protocol and wrong in another. For example, tested DEPBAR
threshold mutations failed at some values while higher values passed. A small
passing barrier or reuse probe is not enough to remove a production dependency.

The current evidence and experiments are indexed in
[`docs/evidence/a100-mma`](../evidence/a100-mma/README.md),
[`exp/tensor_scheduler`](../../exp/tensor_scheduler) and the shared-layout
dossier [Shared-memory swizzling and bank conflicts](../research/ideas/shared-layout-swizzling.md).

