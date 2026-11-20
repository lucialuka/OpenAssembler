# Memory movement and latency hiding on A100

This page connects compiler coverage decisions with existing performance
evidence and the current bounded overlap-policy experiment. It is not a claim
that any one technique accelerates every ML workload.

## Evidence for Expanding Coverage

| Mechanism | Existing evidence and tradeoff |
|---|---|
| Two-stage `cp.async` shared-memory pipeline | Validated production copy/compute path; added stages consume shared memory and can reduce resident CTAs. [Deeper-pipeline study](../research/ideas/deeper-pipelines-and-tile-shapes.md) rejects its measured three/four-stage variants. | 
| Earlier independent pointer work | Some exact shapes benefit from direct pointer homes and early advancement. [Pointer study](../research/ideas/early-gemm-pointer-advance.md) retains regressions on other shapes. |
| Fragment load/compute overlap | Earlier loads can extend live ranges and increase register pressure. [Load-next study](../research/ideas/cutlass-load-next-compute-current.md) and [fragment windows](../research/ideas/prefetch-and-fragment-windows.md) do not establish a universal improvement. |
| Producer/consumer warp split | Dedicated loaders consume threads/registers and require correct phase/arrival accounting. [Warp study](../research/ideas/producer-consumer-warps.md) lost dense GEMM timing; [two-slot mbarrier](../research/ideas/mbarrier-overlap.md) wins only its small controlled workload. |
| CTA tile shape/locality | Smaller tiles may improve small-grid utilization; traversal may reduce memory traffic. Use the [exact measured table](../../exp/dispatch_tuning/RESULTS.md), not a blanket occupancy rule. |
| Packing, storage reuse and fusion | Reduce repeated memory traffic and launches at the model boundary; include packing/lifetimes/optimizer state. [Model bridge](../research/ideas/native-model-bridge.md) and [weight packing](../research/ideas/inference-weight-packing-cache.md) have their own contracts. |

Keep async-copy completion/visibility, LDSM address source reads, HMMA
accumulator dependencies and observer drains intact. Existing scalar latency
measurements are not a general tensor scheduler model. More live registers,
more buffers, a lower load count or higher occupancy is not itself a speedup.

The official [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
describes asynchronous global-to-shared transfers and architecture resource
limits. These hardware mechanisms motivate hypotheses; project-specific
selection requires actual generated-code correctness and timing.

## Current experiment

[Runner](../../compiler_exp/latency_hiding/run_modal.py) compares the existing
admitted production policies with fixed FP16 contiguous RCR GEMM operands,
FP32 accumulation/output and no epilogue. Shapes are 128×512×1024,
1024×1024×1024 and 256×768×1024. The third shape extends beyond the original
exact dispatch cohort. Every case tests all candidate plans accepted by the
production policy generator; this is a comparison of concrete legal variants.

Each result uses three randomized paired rounds of eight samples with sixteen
launches per CUDA graph. Zero/integer/normal inputs, complete output comparison,
same-PTX ptxas bit identity, input immutability, poisoned graph replay, canaries,
disassembly, local/shared/register resources and retained source/binary hashes
are mandatory. The harness enforces the established HMMA8/drain15 controls.
No vendor GPU kernel is admitted into the OA deployment; ptxas and Torch
calculations are independent experimental correctness controls.

Two sequential sessions run the same workload/policy sets. Session zero freezes
its selected policy before session one. Session one evaluates that frozen
selection; its own nominal winner does not replace it. A second session can
reuse a worker and is not guaranteed a different physical GPU. Device/runtime
metadata is retained in each result. No Nsight counters are collected by this
runner, so timings alone cannot establish the causal source of a gain.

Compilation, transfers, packing, host application code, backward and optimizer
updates are excluded. This experiment does not time ptxas or cuBLAS, claim a
whole-training improvement or promote a new default. Before changing a policy,
require held-out benefit against both its OA incumbent and baseline, account
for selection uncertainty, and audit the complete retained artifacts.

```sh
modal run compiler_exp/latency_hiding/run_modal.py --action timing
```

The immutable protocol, frozen selection, raw timing samples and artifact ZIPs
are stored under `compiler_exp/latency_hiding/runs/`. The descriptive replay
and separate production admission outcomes follow.

## Descriptive findings

The [complete paired records and summary](../../compiler_exp/latency_hiding/runs/20261115T051124Z/summary.json)
retain 70 candidate/session combinations: 11, 13 and 11 policies per shape in
each of two sessions. All three input-family gates and poisoned graph checks
pass for every combination. The actual GPU is an A100-SXM4-80GB, despite the
requested A100-40GB tier. Treat the actual metadata as authoritative.

| M×N×K | Frozen policy | Second-session speedup over baseline | Interpretation |
|---|---|---|---|
| 128×512×1024 | `tile64_reuse_direct_early` | 2.377–2.388× | Previously uncovered exact contract; separately admitted below. |
| 1024×1024×1024 | `tile64_reuse_direct` | 1.389–1.391× | Only 1.001–1.005× over the existing `tile64_reuse_direct_early` policy; retain the incumbent. |
| 256×768×1024 | `tile64_reuse_direct_early` | 2.384× | Previously uncovered exact contract; separately admitted below. |

The tile64 variants use 88 driver-reported registers and 32 KiB dynamic shared
memory, versus 216 registers/64 KiB for the reference. The combined-loop policy
uses 232 registers and loses: baseline/candidate ratios are about 0.975–0.980.
Early pointer advancement alone is also slightly slower. The selected result
combines tile geometry, address work and live-range changes; without counters
these comparisons do not uniquely attribute the gain to latency hiding.

The separate [admission campaign](../../compiler_exp/latency_hiding/runs/admission-20261115/)
uses distinct Modal apps, fresh subprocesses and the unchanged production
`gemm_promotion` auditor. It is separate from the descriptive replay above.
Compiler sources are frozen; table provenance may refresh after an independent
binary-continuity audit because explicit-policy benchmarks do not use measured
dispatch. Changed source or data affecting code generation rejects admission.

## Production disposition

The [production auditor's decisions](../../compiler_exp/latency_hiding/runs/admission-20261115/promotion-decisions.json)
admit `tile64_reuse_direct_early` for FP16 128×512×1024 and 256×768×1024.
Both source/ABI/precision/layout contracts meet the unchanged complete gates
in distinct Modal applications and fresh subprocesses. Across all six paired
selection/heldout rounds, gains over the OA baseline are **2.369–2.386×** and
**2.384–2.396×**, respectively. Each result records an A100-SXM4-80GB.

Those two exact entries are appended to
[the installed table](../../openassembler/kernels/sm80_gemm_dispatch.json).
The original 42 entries retain their evidence and selections: the table now
has **44 contracts, 36 nonbaseline policies and eight baseline retentions**.
The 1024³ incumbent stays unchanged because the descriptive alternative did
not establish a repeatable 2% improvement over it. Unknown contracts retain
the baseline. The original [42-contract study](../../exp/dispatch_tuning/RESULTS.md)
remains historical evidence for its own cohort, not a claim about these new cases.

Both new entries use FP16 inputs with FP32 accumulation/output, full tiles,
RCR layout, no epilogue and the recorded alignment/workspace contract. These
resident-kernel gains are relative to OA's previous baseline, not ptxas/cuBLAS
or complete model training. Independent apps establish separate sessions;
they do not prove a different physical GPU or generalize performance to every
A100 variant or application. Raw nominal winners never override frozen choices.

## Next experiments

After resolving the application build/launch boundary, prioritize excessive
address work and memory traffic in actual uncovered operators. Compare generic
legal OA lowerings with specialized recipes. Investigate pressure-aware load
placement, bounded double buffering, vectorized/coalesced tails and fusion
without changing numerical policy. A new multistage or warp-specialized
generator needs complete fill/drain/arrival and allocation accounting first.

Measure compute, DRAM/L2 traffic, scoreboard stalls, active warps, registers,
local spills and shared-memory limits together when counter access is available.
Separate insufficient parallel work from memory latency and from launch/host
overhead. Choose the next mechanism from that evidence rather than assuming
more prefetching is the answer.

## Existing model performance, November 19

The separate [model campaign](../../exp/model_perf/RESULTS.md) narrows the next
bottleneck to small-grid tensor blocks and serial bias-gradient rows. It admits
exact model-only tile choices and deterministic reduction kernels after full
training/replay checks. The coalesced transpose-pack candidate is slower and
rejected. This evidence does not enlarge the global GEMM dispatch table or
attribute a general win to coalescing. Whole-task and independently paired
regions remain distinct from standalone kernel distributions.

## November 20 model-state memory traffic

The [model semantics study](model-semantics.md#memory-and-latency-decisions)
compares split/fused bias-corrected AdamW with full state gates, documents
direct broadcast indexing and deterministic embedding-gradient costs, and
separates append-copy cache correctness from future paging/online attention.
It does not change the incumbent async/Tensor Core schedules or infer
whole-training speed from component CUDA-event measurements.
