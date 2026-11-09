# Open questions and next steps

## Highest-value technical questions

1. Can a virtual-register tensor scheduler interleave independent HMMA chains
   while preserving accumulator RAW, LDSM address-WAR, async visibility and
   observer edges after physical allocation?
2. Which combination of fragment ownership, register windows, reuse hints and
   shared layout lowers OA's scalar/control envelope without reducing active
   warps?
3. Can reductions combine load overlap, vectorized reduction trees and explicit
   FMA policy while reducing long-scoreboard stalls rather than only static
   instruction count?
4. Can a shape/alignment/workspace manifest select grouped-N, Morton, m-fast,
   load-next and pointer policies without overfitting one A100 run?
5. Can actual forward/backward/optimizer integration preserve full model state
   while using native OA kernels at the boundaries that matter?

The November 9 revalidation integrated two bounded opt-in paths with full
source/resource contracts: `--fragment-schedule load_next` for two wide shapes
and `--cta-order grouped_n8` for N-grouped launches. Their held-out results do
not support global automatic dispatch. The four-edge tensor scheduler,
producer/consumer warp split, fused reductions, sparse MMA and stable GELU
backward remain experiment or capability records.

## Hardware boundaries

A100 does not provide Hopper TMA, thread-block clusters, DPX or the H100 FP8
Tensor Core path. A compiler flag cannot enable those capabilities. Software
approximations must be measured as separate designs.

## Promotion rule

Do not promote a shorter instruction listing by itself. A production change
requires exact numerical checks, valid disassembly, resource/occupancy limits,
paired A100 timing on held-out shapes and counters that support the claimed
mechanism. Preserve a conservative fallback for shapes outside the evidence.

