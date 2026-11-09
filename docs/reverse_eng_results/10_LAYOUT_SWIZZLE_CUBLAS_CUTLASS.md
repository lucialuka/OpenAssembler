# Layouts, CUTLASS and cuBLASLt observations

## Public CUTLASS techniques

The public CUTLASS SM80 code exposes a useful tuning vocabulary:

* CTA, warp and instruction tile shapes;
* stage count and async-copy pipeline depth;
* operand layouts and shared-memory swizzles;
* fragment ownership and register windows;
* CTA traversal and epilogue vector width;
* fused epilogues and reduction placement.

The experiment layer implemented a load-next/compute-current fragment schedule
and grouped-N traversal based on these public patterns. Current source-hashed
A100 replays now expose both as explicit opt-in production options under exact
contracts. Load-next passes FP16/BF16 at 1024×4096×1024 and 4096×1024×1024;
large held-out FP16 shapes regress. Grouped-N passes repeated FP16/BF16 tall
cases and improves the 16384×4096×4096 case by about 6.9–7.3%, while wide and
square cases remain unsuitable for automatic dispatch.

These are shape-specific experiment results, not universal dispatch rules.

## cuBLAS and cuBLASLt boundary

Public cuBLASLt exposes heuristic selection, workspace limits, alignment,
reduction schemes, tiles, stages, split-K and epilogues. OpenAssembler can use
that control vocabulary for an independent measured manifest. It cannot claim
to have extracted proprietary cuBLAS kernels from public APIs, and no such
extraction was performed.

## CTA order and layout interpretation

Morton order and grouped-N CTA traversal change the order in which CTAs touch
tiles. They are different from the row/column fragment layout consumed by MMA.
The A100 evidence shows locality and scheduling effects are shape dependent;
static instruction count alone does not choose the winner.

See [Public CUTLASS and cuBLASLt techniques](../research/ideas/cutlass-cublas-public-techniques.md),
[Grouped-N CTA traversal](../research/ideas/grouped-n-cta-swizzle.md) and
[CUTLASS load-next schedule](../research/ideas/cutlass-load-next-compute-current.md).

