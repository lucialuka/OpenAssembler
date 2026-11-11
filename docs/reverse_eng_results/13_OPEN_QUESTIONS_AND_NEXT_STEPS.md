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
4. Can the exact measured table extend beyond its 42 GEMM and five rowwise
   contracts to new shapes, transposed storage, tails, precision, alignment and
   workspace classes while preserving independent validation?
5. Can actual forward/backward/optimizer integration preserve full model state
   while using native OA kernels at the boundaries that matter?

The November 11 [production replay](../../exp/dispatch_tuning/RESULTS.md)
resolved a bounded part of the dispatch question. Independent selection and
held-out sessions installed 34 nonbaseline policies and eight reference
retentions: 22 tile64 choices, ten direct-pointer policies and two Morton8
choices. Copy-address reuse serves 20 exact contracts. Five separate rowwise
contracts choose two/eight CTA warps, with unchanged row arithmetic. These
results establish exact routing; they do not establish an optimum for unknown
shapes or a win over ptxas/cuBLAS across transformer workloads.

The earlier `--fragment-schedule load_next` and `--cta-order grouped_n8`
contracts remain explicit/tuning candidates, but neither won this frozen
campaign. A prior loss is not grounds to delete a candidate that might win on
another contract. The four-edge tensor scheduler, producer/consumer warp
split, fused backward reductions, sparse MMA and stable GELU backward remain
experiment or capability records.

## Next validation targets

First collect current exact-policy counters to separate geometry, address
reuse, pointer overlap and locality effects; the newest timing matrix does not
replace earlier Nsight studies or prove a stall mechanism. Then prioritize
non-tensor instruction selection and schedule/allocation work against the
remaining same-source ptxas gap. Maintain the HMMA8/drain15 fallback until a
general scheduler-produced stream passes full binary, numerical, resource and
independent A100 gates. Native graph integration and first-divergence/full
state validation are required before a kernel gain can support a training-step
or tokens-per-second claim.

## Hardware boundaries

A100 does not provide Hopper TMA, thread-block clusters, DPX or the H100 FP8
Tensor Core path. A compiler flag cannot enable those capabilities. Software
approximations must be measured as separate designs.

## Promotion rule

Do not promote a shorter instruction listing by itself. A production change
requires exact numerical checks, valid disassembly, resource/occupancy limits,
paired A100 timing in an independent held-out session and counters that support
any claimed mechanism. A frozen candidate must pass every paired round against
the baseline and incumbent; never replace it with a new held-out winner after
seeing the result. Preserve a conservative fallback outside the evidence and
use unseen-shape controls before broadening a heuristic.

