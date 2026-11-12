# Whole-repository review: investigation and admitted changes

The supplied review correctly separates native correctness from competitive
performance. Its source commit e071448 matches the clean checkout at the start
of this investigation. The pasted review and selected raw summaries/CSV tables
are retained under [input](input/). Original reviewer files are unmodified.
The per-kernel diagnostic analysis is [here](input/bottleneck-analysis.json).

## Causes and response

| Observation | Root cause / implication | Response |
|---|---|---|
| Small FP32 models spend roughly 1–4 ms per host step | Every OA launch rebuilds ctypes arguments, binds context, records tensor use and crosses the Python/driver boundary; repeated host submissions leave GPU gaps | Capture the identical complete OA MSE/backward/SGD sequence once and replay a bounded graph with fixed storage, stream and learning rate |
| Tensor MLP's MSE costs about 1.56 ms in the review | A single GPU thread loads and reduces all 16,384 products serially; this is larger than any GEMM in that sample | Coalesce subtraction/square loads across a warp, then consume products in original index order; no tree reduction or rounding-policy change |
| Tensor bundle takes about 14 s to build | Native generation and textual disassembly dominate; repeated clean output is not a cache hit | Make disassembly listings optional while preserving every native module byte and ownership inventory |
| Most matrix kernels lose to same-PTX ptxas and cuBLAS | Host replay cannot repair instruction count, resource usage or small-grid GPU utilization | Keep matrix optimization as a separate next experiment; no vendor win is claimed |
| Audit ignores names containing memcpy/memset | Name substrings are not reliable activity types | Compare raw Kineto category=kernel events by exact name/count; copies and GPU annotations are classified separately |
| Catalogue/import regressions fail | Two missing inventory rules; optional-runtime imports share collection state | Add both rules and run the import guard in an independent interpreter |

Unique-kernel sums in the diagnostic analysis are not launch-weighted model
critical paths. GPU event intervals around ordinary Python submissions include
idle host-dispatch gaps. They are distinct from kernel-only timings below.

## Experiment and admission protocol

All candidates started solely under exp/review_response. No production change
preceded the corresponding numerical and paired performance evidence.
Admission used at least 2% gain in all three paired rounds on two independent
A100 workers, with source snapshots and raw samples retained. Actual GPU variants
are recorded per worker; absolute values are never pooled across devices.

Training replay compares four changed-input complete updates against CPU
Module/autograd (or the explicit FP16 RNE product reference). Every prediction,
loss, input gradient, parameter gradient and updated parameter is checked.
The replay and ordinary OA paths additionally match bit for bit, preserve
inputs/targets, poison scratch and verify allocation guards. Capture leaves
parameter bytes unchanged. Independent profiler categories/names/counts match
18, 74 and 32 compute launches per step. Parameter mutation, stream changes and
closed replays reject.

## Unchanged-kernel training replay

The [first passing campaign](runs/20261112T133348Z/result.json) and
[independent repeat](runs/20261112T133630Z/result.json) each include 424 complete
numerical field checks. Each model runs in a fresh process. Per round there are
25 shuffled paired samples; each sample includes resident-input host submission
and synchronization. Ordinary OA and replay use identical saved native cubins.

| Model | First worker, three host speed ratios | Independent repeat |
|---|---|---|
| FP32 MLP | 6.888, 7.236, 7.249 | 8.738, 8.923, 8.984 |
| FP32 attention/LayerNorm/GELU | 11.684, 11.189, 12.772 | 15.397, 15.539, 15.661 |
| Explicit FP16 RNE Tensor Core MLP | 1.179, 1.177, 1.233 | 1.247, 1.244, 1.245 |

Both passing replay campaigns report A100-SXM4-40GB. First campaign preparation
costs are about 196, 139 and 627 ms respectively, including new runtime loading
and allocation. Preparation is excluded from steady state; no complete
deployment amortization is claimed. Source and artifact archives live beside
each result. No eager/Inductor arm is rerun here and prior framework times are
not used to infer a new framework speedup.

## Exact-order loss kernel and build diagnostics

[First completed loss cohort](runs/ablation-20261112T133827Z/result.json) uses
A100 80GB PCIe; [independent repeat](runs/ablation-20261112T134228Z/result.json)
uses A100-SXM4-40GB. Each size checks zero/integer/normal/dynamic-range inputs,
exact OA baseline output bits, an independent double formula, immutable input
bytes, both allocation guards, and exact profiler counts before timing. Each
round uses 15 shuffled samples of eight kernel invocations per CUDA graph.

| Elements | First worker speed range | Repeat speed range | Production admission |
|---|---:|---:|---|
| 1 | 0.611–0.619× | 0.593–0.604× | Rejected; serial retained |
| 33 | 0.981–0.988× | 0.983–0.992× | Rejected; serial retained |
| 112 | 1.438–1.445× | 1.436–1.439× | Exact size |
| 136 | 1.415–1.417× | 1.426–1.428× | Exact size |
| 513 | 1.614–1.615× | 1.623× | Exact size |
| 16,384 | 1.753× | 1.755× | Exact size |
| 65,539 | 1.755× | 1.754–1.755× | Exact size |

The kernel uses supported full-mask XOR shuffles, with lane 0 receiving products
in increasing index order. The FP32 subtract, square, left-to-right addition
and final multiplication retain the serial formula. Masked tail products are
zero; only lane 0 stores. Unknown sizes remain serial. The emulator regression
also compares PTX, scheduled SASS and poisoned SASS for representative tails.

The [composed loss plus replay campaign](runs/20261112T134229Z/result.json)
passes all 424 full-state fields and paired bits. The Tensor Core host-step
ratios are 2.104, 2.104 and 2.194× over ordinary OA dispatch with serial loss.
That combined gain is not attributed entirely to the loss kernel.

The fresh-output build ablation alternates listing-enabled and binary-only
order over three rounds per worker. Complete native cubin bytes match in every
pair; bundle/source/text/resource contracts validate. The first loss cohort's
build ratios are 1.613, 1.571 and 1.576×; the repeat gives 1.684, 1.626 and
1.625×. This is worker-local compilation, excluding transport/queue. No persistent
cache is used. Default builds still emit listings; binary-only is explicit.

## Production paths and diagnostics

The admitted code lives in modelfront/replay.py, profiling.py and kernels_loss.py.
No production module imports exp/. Model configuration accepts:

    execution = "cuda_graph"
    sass_listings = false

Execution defaults to direct dispatch and listings default on. The public
CompiledModel.prepare_native_training method returns a run/close object.
Input/target values can change in place on the preparation stream. Shapes,
layout, storage, parameter bindings and learning rate are fixed. Externally
modified or replaced parameters require recapture. Returned tensors borrow
workspace storage until the next invocation.

Model reports retain separate worker-local CPU factory/capture, storage planning
and native bundle phases, graph preparation, host-step samples, CUDA stream
event intervals, profiled iteration indexes and per-kernel duration lists.
Profiler duration lists describe observed invocations, not independent timing
distributions. Detailed shuffled timing distributions remain in these experiments.
OA generation versus donor-tool times inside native_bundle remain a future
instrumentation refinement; queue/deployment startup are not folded into it.

The [promoted public-API replay](runs/20261112T134849Z/result.json) passes all
424 complete fields using CompiledModel.prepare_native_training with the
production ordered-loss selector. Both paths use the newly selected loss
kernel. All paired bits and independent activity audits pass. The promoted
method additionally rejects live parameter replacement through its owning
CompiledModel, external parameter mutation, stream changes and use after close.
The retained source identities match the production replay and loss files.

[Ordinary CLI build and separate saved replay](runs/production-cli/) each pass
93 complete fields for attention and 33 for the Tensor Core MLP. Both select
CUDA graph execution and omit textual listings; exact profiler categories and
counts pass, with zero vendor compute and zero replay code generation. The
attention CLI cohort was captured before the ordered-loss selector was promoted
and retains serial loss. The Tensor Core cohort uses the promoted ordered loss.
The public-API cohort above validates the current ordered loss in all three
models. Build phase, graph preparation and profile duration data are retained.

Final local regression is 948 passed with two optional Torch skips. The separate
Torch-enabled CPU graph/numerics/import collection is 94 passed; its host Torch
version is recorded separately and is outside the pinned GPU runtime. Focused
model/compiler/audit/inventory checks pass 137 tests. These scopes overlap and
are not added together. The final documentation and whitespace checks pass.
No live SSH destination was provided; Modal and SSH use the same native
implementation and existing transport tests remain in the full regression.

## Remaining work, in evidence order

1. **FP32 matrix kernels:** build tiled/coalesced ordered-accumulation forward,
   dX and dW candidates, including rank-3 flatten/layout costs and small-M/tails.
   Compare complete regions and same-PTX controls. TF32/FMA contraction cannot
   silently replace the FP32 separate-multiply/add contract.
2. **Tensor Core regions:** reduce repeated packing and launch overhead with
   versioned weight packs and common-operand reuse. Inference packs may persist
   while weights are unchanged; training must refresh after every update.
   Include transpose, bias, workspace and update costs in admission.
3. **Build generation:** profile native generation by kernel and add an exact
   content-addressed cache keyed by compiler/data/toolkit/architecture, source,
   precision, entry and resource contract. Revalidate executable text on every
   hit. Binary-only output already removes optional disassembly cost.
4. **Ownership validation:** retain disguised-name negative tests and raw
   activity categories. A missing/incomplete trace remains a failed gate.
5. **Comparators and coverage:** build the explicit-RNE whole-model mathematical
   framework comparator as a separately labelled research control, and expand
   causal/cache/batching/optimizer contracts independently. The supplied
   ptxas/cuBLAS deficits and half-model comparator gap remain open.

## Retained failures

The first replay attempt captured on the default CUDA stream and correctly
failed before execution. Its [raw result](runs/20261112T133106Z/result.json)
is excluded. The fix captures on a temporary non-default stream and replays
on the original bound stream without moving storage.

The first loss harness built correctly but read the older manifest key,
then failed with KeyError before any admitted timing. Its
[result and artifacts](runs/ablation-20261112T133533Z/result.json) remain.
An initial local test edit placed an existing bias-gradient assertion in the
wrong function; six NameError failures remain in final-focused.log. The
assertion was restored and final-focused-fixed.log passes. No numerical
tolerance or admission threshold was changed to admit these candidates.
