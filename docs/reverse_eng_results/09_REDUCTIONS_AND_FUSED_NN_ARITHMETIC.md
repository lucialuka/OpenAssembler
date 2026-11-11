# Reductions and fused NN arithmetic

## Production rowwise geometry

The native FP32 softmax and non-affine population LayerNorm generators support
contiguous rows of width 1–1024, including row/column tails. Each warp owns one
row. The measured production table now chooses two or eight warps per CTA for
five exact contracts; unknown or stale contracts keep four. This changes CTA
grouping, not reduction order or arithmetic.

Selection covered 58 operator/shape contracts with 1/2/4/8-warps candidates;
seven choices crossed a 3% per-round threshold and five repeated that result
in independent held-out replay. LayerNorm 129×1023 and 4096×1024 missed the
frozen threshold and remain on four warps. Current generated PTX and native
words match the archived promoted candidates. These are OA-vs-OA gains, not
proof of superiority over ptxas/cuDNN or a native training step. The exact
table, five input families and reproduction commands are in
[shape-aware rowwise dispatch](../research/ideas/shape-aware-rowwise-dispatch.md).

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

The detailed evidence and per-case counter table are in
[Fused backward reductions](../research/ideas/fused-backward-reductions.md#streamed-loadfma-follow-up-and-counters)
and [NN arithmetic campaign](../research/ideas/nn-arithmetic-campaign.md).
The immutable Nsight record is
[reduction_overlap_ncu_20261108T061002.104721Z.json](../../exp/weaknesses/reduction_overlap_ncu_20261108T061002.104721Z.json).

