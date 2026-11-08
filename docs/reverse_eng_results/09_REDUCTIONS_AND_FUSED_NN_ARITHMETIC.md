# Reductions and fused NN arithmetic

## Why reductions remain a major gap

Backward RMSNorm and softmax reductions expose memory completion, shuffle,
reduction-tree and synchronization costs. The source-inclusive streamed
load/FMA candidate reduced OA dynamic instructions by 10–16% and registers by
12–31 across 12 A100 cases, but Nsight Compute still measured roughly 82–91%
long-scoreboard stalls for OA versus about 19–49% for matched ptxas.

This shows that scalar/FMA cleanup is useful but insufficient. The remaining
work is load overlap, reduction-tree structure, vectorization, synchronization
and memory locality.

## Contraction policy

FMA contraction can reduce instruction count and registers, but it changes
rounding. The current experiments retain an explicit precision policy and keep
normal/structured tolerance checks separate from bitwise ptxas comparisons.

## Native operator composition

Isolated operators can pass while a whole training step fails state comparison.
The first divergent intermediate must be located before a candidate is promoted.
Application-level results using TorchInductor/Triton must not be described as
native OpenAssembler reduction code.

The detailed evidence is in [Fused backward reductions](../research/ideas/fused-backward-reductions.md),
[NN arithmetic campaign](../research/ideas/nn-arithmetic-campaign.md) and the
Nsight report [reduction_overlap_ncu_report.md](../../exp/weaknesses/reduction_overlap_ncu_report.md).

