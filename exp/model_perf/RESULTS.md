# Exact model kernel performance improvements

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
