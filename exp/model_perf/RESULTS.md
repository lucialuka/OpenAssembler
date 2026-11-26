# Exact model kernel performance improvements

## November 26 kernel-research continuation

The first completed changes from the attached agenda are bounded forward pack
reuse and a cheaper allocator interval scan. The experiments started under
`exp/model_perf/` and were repeated through production after integration.
The checkout also contains concurrent Boolean-storage, documentation and
Conv2d work. Each remote envelope retains the exact compiler source hashes it
consumed. The final worker's production sources match the delivered sources.

### Repeated projection operand packs

The separate-pack runtime now retains a repeated row-major 128×128 FP16 pack
within one forward call. Its key includes the logical source, layout and
conversion signature. The runtime clears readiness on each forward, including
captured graph construction. The resulting graph refreshes every retained
pack on replay. The measured projection model shares its input across Q/K/V
products, eliminating two conversion launches. Unique packs, other matrix
shapes and backward/transposed layouts keep their existing execution paths.

The staged [A100-SXM4-40GB experiment](evidence/pack-reuse-20261126T074320Z.json)
measured 43.75→39.42 µs, with 14→12 launches. An independent
[integrated A100-SXM4-80GB replay](evidence/pack-reuse-integrated-20261126T074559Z.json)
measured 41.19→37.08 µs with storage views/reuse enabled. These workers retain
their separate absolute timings.

The final [A100-SXM4-40GB confirmation](evidence/pack-reuse-integrated-20261126T083532Z.json)
uses Torch 2.7.1+cu128, Triton 3.3.1, CUDA toolkit 12.8 and driver 620.70.06.
Each row below is the median of three round medians from randomized resident
CUDA Graph samples, fifteen samples per round and four invocations per graph.
All values are µs. Build, allocation, CPU capture, transfers and delivery are
outside these GPU intervals. Speedup is incumbent OA time / production OA time.

| Storage views | Buffer reuse | Incumbent OA | Production OA | Incumbent PTX + ptxas | Reuse schedule + ptxas | Triton | OA speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| On | On | 44.18 | 39.90 | 40.76 | 36.65 | 19.34 | 1.107× |
| On | Off | 44.11 | 39.72 | 40.87 | 36.58 | 19.35 | 1.111× |
| Off | On | 44.01 | 39.74 | 40.58 | 36.48 | 19.38 | 1.108× |
| Off | Off | 44.26 | 39.83 | 40.98 | 36.54 | 19.33 | 1.111× |

The matched ptxas arm compiles the identical OA PTX and uses the same pack
allocation/binding/launch schedule. Independent trace audits confirm twelve
launches for both selected schedules. Production OA remains about 2.06× slower
than direct Triton and 1.09× slower than matched-schedule ptxas on the first row.
The dense MLP has no repeated forward operand pack to remove and its 32-launch
training control remains near neutral. Workspace allocation bytes remain the
same on these plans.

Each final configuration passes six direct iterations and six actual captured
replays across incumbent OA, production OA, incumbent ptxas, scheduled ptxas and
Triton. Every replay changes inputs and targets. Inference also changes every
weight in place after capture. OA workspaces are poisoned before subsequent
checks. The gate compares complete predictions, losses, input/parameter
gradients and updated training weights against CPUReference, with the original
`rtol=0.002, atol=0.00002` policy, input immutability and allocation guards.
Independent profiler names/counts match the runtime ledger before timing.

A further [A100-SXM4-40GB replay](evidence/pack-reuse-integrated-20261126T084247Z.json)
passes the same eight configurations with per-field hashes and error counts.
Its [independent local audit](evidence/pack-reuse-complete-state-audit-20261126.json)
checks all **2,880 field comparisons** and reloads complete snapshots covering
**7,239,720 output elements** from one direct step and one captured replay per
arm/configuration. It also verifies native modules/text/registers, paired
sample order, all raw interval medians and profiler launch counts. The 64 MiB
raw snapshot/build archive remains in the ignored run directory, with its hash
retained in the audit and evidence index.

### Allocator compile cost

The interval scan traverses instructions in increasing order. A value's first
visit sets its start, and every subsequent visit sets its latest endpoint.
Streaming definitions, uses and CFG live-out values removes temporary lists,
repeated dictionary lookups and min/max comparisons. The spill selection,
pressure schedule and emitted native program retain their previous choices.

The [staged scan experiment](evidence/kernel-study-allocator-intervals-experiment.json)
preceded production changes. The [integrated warm-generation record](evidence/kernel-study-allocator-integrated.json)
retains three randomized paired samples with parsed PTX and a warm encoder on
the development Mac with Python 3.14.5. Concurrent host work limits throughput
comparisons. These are seconds of code generation.

| Fixture | Prior scan | Production scan |
|---|---:|---:|
| scalar96 | 2.360 | 1.574 |
| pred24 | 0.227 | 0.178 |
| group4_auto | 23.017 | 12.941 |
| no_pressure | 0.000420 | 0.000391 |
| impossible_groups | 0.000690 | 0.000594 |

The [cold subprocess observations](evidence/kernel-study-allocator-cold.json)
include fresh parsing/encoder creation, with one sample per arm. They measure
3.48→2.58 seconds for scalar96 and 29.96→16.13 seconds for group4_auto.
The [toolkit build worker](evidence/allocator-build-20261126T082620Z.json)
also compares four complete builds including ptxas/container packaging. Its
single baseline/current pairs measure 11.03→6.57 seconds for scalar96,
0.94→0.72 for pred24, 93.77→47.22 for group4_auto and 1.96→1.45 for pressure_tight.
All four complete cubins match byte-for-byte. These observations have separate
measurement boundaries and provide no kernel-execution speedup claim.

All [48 fixture/optimization-level comparisons](evidence/kernel-study-allocator-all-integrated.json)
preserve complete code, controls, labels, exit offsets, encoded words,
registers, local frame/spill resources and impossible-pressure diagnostics.
The [policy continuity audit](evidence/allocator-continuity-20261126.json)
additionally matches all 44 installed GEMM contracts and five rowwise policies
against 98 retained executable sessions. Dispatch identities were rebound only
after this audit, preserving measured policies and original timing evidence.

### Negative controls and remaining uncertainty

The [early pressure-scheduling experiment](evidence/kernel-study-allocator-experiment.json)
accelerated recoverable-pressure fixtures but increased compile cost for
spill-heavy controls. The streaming scan was selected independently of that
candidate. The native FP16x2 conversion experiment passed exact packed-byte
checks for signed zeros, subnormals, ties, infinities and signed canonical NaNs.
Its completed dense-training row showed only about 0.5 µs improvement. Its
overall envelopes failed before full campaign completion, so that conversion
remains experimental.

Retained failed attempts include a reserved PTX register name, a graph-capture
stream mismatch and incomplete profiler launch traces. Fresh workload
processes and explicit stream capture resolved those harness issues.
One [later incumbent dense-control failure](evidence/pack-reuse-integrated-20261126T082620Z.json)
reported a large prediction mismatch with views disabled and buffer reuse
enabled. Its original log lacks the iteration/input dump needed for complete
isolation. An [isolated replay](evidence/pack-reuse-debug-integrated-20261126T082958Z.json)
with the same seed/configuration passed, followed by the expanded final
six-direct/six-replay campaign. The earlier numerical failure remains
unexplained and excluded from performance admission. It is retained as an
open control issue alongside the successful final gates.

The final local full suite passes **4,654 tests with 16 explicit skips**. The CUDA
worker passes **118 focused tests**. Final schedule/runtime/dispatch checks pass
**75 tests with one explicit skip**, including duplicate-input identity and the unique/unmeasured-shape
boundaries. The [evidence index](evidence/kernel-study-20261126.json) preserves
raw envelopes, partial results from failed runs, compile samples and archive
hashes. Other agenda items, including tensor rescheduling, parallel embedding
gradients and tiled attention, remain open research.
The [final validation record](evidence/kernel-study-validation-20261126.json)
binds these completed scopes to current sources, tests, hardware envelopes and
the retained control limitation.

November 19, 2026. The user review correctly identifies dense backward and
reduction overhead. The frozen current-revision feature inventory now passes
both tests unchanged; the reported stale retired-application rule is absent.
The earlier ML suite prose was corrected: OA beats its manual PyTorch CUDA
operator schedule on dense FP16 and projection inference, while dense FP32
remains slower. Original numeric rows and comparator definitions are preserved.

## Research and selected implementation

The initial ablation keeps precision, shapes, complete task semantics and gates
fixed. On A100-SXM4-40GB, changing the existing 128x128 output tile to 64x64 for
four small FP16 products reduces the full dense training step from about
236 to 184 microseconds. The isolated forward interval falls from 41 to 27
microseconds, and backward from 117 to 91. Projection inference falls from
68 to 44. The separate 32x32 shared-memory transpose-pack candidate passes
correctness but is slightly slower (about 242 versus 236 microseconds for
training). It is rejected. Coalesced stores alone do not establish a speedup.

Two reduction changes then remove another bottleneck. Bias gradients of exactly
128x128 and 128x256 split rows among four warps and reduce their column partials
through shared memory. A 128x128 MSE uses 1,024 threads, warp partials, one CTA
barrier and a final warp reduction. The 256- and 512-thread variants were also
measured; 1,024 threads won the standalone sweep and passed complete training.
Neither reduction uses atomics or removes synchronization dependencies.

The production implementation lives in `modelfront/kernels_training.py`:

- FP16 M/N/K = 128/128/128, 128/256/128, 128/128/256 and 256/128/128 select
  64x64 only when the existing model plan would select the 128x128 baseline.
  Other shapes, precision and already selected GEMM policies remain guarded.
  This changes model recipe selection, not the global standalone GEMM table.
- Bias gradients select the parallel reduction only for the two admitted shapes.
- MSE selects the warp reduction only for the previously admitted 128x128 loss.
- All other matrix/loss/bias shapes keep their existing paths. Packing, crops,
  layouts, master state and separate/dual policy semantics are preserved.

Reduction association changes, so references use the existing FP32 tolerance,
not bitwise identity. FP32 task gates stay rtol=1e-4/atol=1e-5; explicit FP16
operand tasks stay rtol=0.002/atol=0.00002. No tolerance or optimizer policy was
changed to admit a candidate.

## Correctness and admission

Before promotion, five standalone patterns (normal, zero, constant,
cancellation and common offset) pass full outputs, changed outputs after graph
capture, allocation guards and input immutability for both bias shapes and all
loss variants: 40 checks. Four exact tensor shapes pass zero, integer, normal
and alternating patterns under both tiles: 32 checks, each including two
poisoned graph replays. Complete task admission checks four changing-input
direct steps and three poisoned changing-input replays, including prediction,
loss, dX, parameter gradients and updated weights. Independent profiler
activities must match OA launch names and counts before timing.

Randomized timing uses three rounds of 15 shuffled event samples, four resident
steps per graph. Compilation, setup, transfer and CPU delivery are outside
these intervals. Forward, backward and loss/seed are independently paired
intervals and are not summed to reconstruct the total. Standalone operator
samples use 16 repetitions; they do not represent entire model latency.

The first all-variant harness failed because the profiler audit had already
saved its trace before a second export. That harness error was fixed and rerun;
it is not a failed arithmetic result. Failed envelopes remain evidence.

## Integrated production confirmation

Final timings come from the isolated performance checkout based on `91e3f94`,
after unrelated edits appeared in the original shared checkout. Early production
runs that included those concurrent edits remain archived but do not serve as
the final admission. The final integrated run compares frozen pre-change
emitters against the integrated production implementation on the same worker.
All three paired rounds pass the unchanged complete-state/replay/ownership gates.

| Task | Frozen OA µs, round range | Production OA µs, round range | Paired speedup range |
|---|---:|---:|---:|
| Dense FP16 training | 234.77–235.14 | 127.99–128.23 | 1.835–1.835× |
| Projection inference | 67.79–68.25 | 44.14–44.21 | 1.534–1.545× |
| Dense FP32 training | 227.21–228.40 | 170.57–172.93 | 1.320–1.332× |

The FP16 independently paired backward interval is 150.21–150.46→74.54–74.74
microseconds, forward 51.82–52.00→33.73–33.97, and loss/seed
23.28–23.30→11.25–11.36. These intervals come from a different capture than the
full total; they cannot be added or compared as absolute values to the user's
older worker. Launch counts stay **32 / 14 / 18** for FP16 training, projections
and FP32 training. No reduction in host submissions is used to explain these
kernel improvements.

Final raw evidence: [paired integrated confirmation](evidence/integrated-20261119T150400Z.json)
and [plans/ledgers](evidence/isolated-integrated-plans-ledgers.json). Actual device:
A100-SXM4-40GB; UUID and driver 620.70.06 plus tool versions are retained. This
identifies the worker, not independent physical hardware for every prior session.

## Fresh four-arm comparison

This separate worker reruns the unchanged frozen task/comparator harness with
current production OA. Three rounds of 15 shuffled samples use four resident
steps per graph. Values are medians of round medians, microseconds. All four
arms pass four changing-input full-state checks before timing. TF32 is disabled.

| Task | OA | Direct Triton | New OA PTX + ptxas | Manual PyTorch CUDA ops |
|---|---:|---:|---:|---:|
| Dense FP16 training | 128.81 | 60.65 | 105.74 | 246.99 |
| Projection inference | 44.38 | 19.82 | 40.97 | 105.83 |
| Dense FP32 training | 168.85 | 98.61 | 86.87 | 115.22 |

OA remains **2.12× / 2.24× / 1.71× slower than Triton**, and
**1.22× / 1.08× / 1.94× slower than same-PTX ptxas**, respectively. Ptxas
compiles the newly selected OA PTX, so its latency also improves; comparing new
OA against only the old ptxas stream would overstate the backend improvement.
The manual operator schedule is not stock autograd or a direct cuBLAS search.
These results narrow the gap and do not establish OA superiority over NVIDIA or Triton.

| Dense FP16 region | OA | Triton | Same-PTX ptxas | Manual CUDA ops |
|---|---:|---:|---:|---:|
| Forward | 34.36 | 17.06 | 31.87 | 75.66 |
| Loss and seed | 11.27 | 7.98 | 8.68 | 11.72 |
| Backward | 75.25 | 30.32 | 59.26 | 153.94 |
| Update | 9.96 | 9.52 | 9.76 | 9.87 |

These remain independently paired component intervals. The largest remaining
FP16 gap is still backward products/conversions, followed by forward packing
and composition. The rejected transpose-pack result argues against promoting
coalescing alone. Next candidates are producer-side packing/fusion with an
explicit liveness/layout contract, avoiding repeated source regeneration during
cache validation, and joint model tile/launch planning. Each needs its own
complete-task gate; this patch does not extrapolate to new shapes or fuse
operators without evidence.

[Final four-arm result](evidence/comparisons-20261119T150402Z.json) retains every
sample, gate, per-kernel trace duration, complete plan and executable hashes.
The older suite's per-kernel traces remain diagnostic, not distributions.

## Cache and other review observations

Binary-only cache keys now query only ptxas; requests for SASS listings still
include nvdisasm. A real cache miss and validated hit run with every nvdisasm
call forced to fail. Changed weights are refreshed, a corrupted cached cubin
is rebuilt, executable/source verification remains enabled, and donor cubins
stay excluded. A mock-backed regression also checks that the optional tool
is absent from binary-only identity lookup. This removes an unnecessary tool
dependency; source/plan generation and full validation can still dominate hits.

Replay-input rebinding is a separate defensive API observation. Supported replay
usage still changes values in place; this performance patch does not add a
new rebinding policy. Broader coverage work is deferred as requested.

Per-kernel standalone reduction timings are distributions. The four-arm model
suite still records one profiled invocation per arm for diagnostic per-kernel
durations; it does not provide steady-state distributions for every model
kernel. Compile backend/packaging/disassembly subphases remain bundled in the
shipped model report. This work does not claim to resolve those measurement gaps.

## Evidence and reproduction

See [README.md](README.md) for the pinned commands and survey scope. Successful
and failed envelopes include source hashes; generated bundles stay in local
ignored runs while compact outcomes and plans/ledgers are retained. GPU variant,
UUID, driver and tool versions are recorded in the new native campaign. The
existing four-arm suite retains its own metadata limitations and comparator
scope. Absolute times from separate workers are not pooled into a speedup.

The isolated binary-only cache gate measured a **4,430 ms miss / 593 ms hit**
for the dense FP16 plan, with a cache hit required to invoke no backend build.
This is one worker-local pair, not a cache speed distribution. The
[raw cache result](evidence/cache-20261119T150416Z.json) confirms missing-disassembler
operation, changed weights, corruption rebuild and donor exclusion. The pinned
[targeted regression worker](evidence/tests-20261119T150416Z.json) passes 121 tests.

Final local regression: **900 passed, 146 skipped**. The isolated checkout lacks
some generated probe corpus fixtures, and optional Torch scopes are unavailable
in the local interpreter; skips are not claimed as passes. The pinned Torch
worker separately passes 121 focused model tests. The feature-inventory tests
pass 2/2 unchanged. Documentation checks pass with 144 Markdown files and 82
catalogue entries after copying the primary checkout's existing ignored
compiler evidence into the isolated checkout. A clean Git-only checkout of the
base revision already lacks 16 referenced targets under ignored `compiler_exp/`
and `docs/compiler/research-plan.md`; this inherited documentation portability
problem is outside this performance patch. [Validation record](evidence/validation.json).
