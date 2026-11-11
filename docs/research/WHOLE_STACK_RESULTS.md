# A100 whole-application and native-stage results

Measured November 6, 2026 on NVIDIA A100-SXM4-40GB (SM80), PyTorch
2.7.1+cu128, CUDA 12.8 and Python 3.11.13. All new code is isolated under
[`exp/whole_stack`](../../exp/whole_stack/README.md). No production compiler
code was changed or automatically substituted into user applications.

This is the original completed study, before the later compiler/operator follow-ups.
Current deployment and the newest admitted results are summarized in
[STATUS.md](../STATUS.md); per-feature methods and commands are in the
[research catalogue](README.md).

The later [November 7 native NN arithmetic campaign](ideas/nn-arithmetic-campaign.md)
is a separate experiment: 26 named operator families, 18 shape/precision tensor
cases and six custom networks with full-state training gates. Its new native
templates remain under `exp/nn_round2/`. Some composed training routes fail
even when isolated formulas pass; matched ptxas remains faster in the captured
GEMM-only inference comparisons. See that dossier for its own 40GB/80GB records,
conditional improvements, exclusions and reproduction. The tables below remain
the original November 6 study and are not overwritten or pooled with the new one.

The completed November 11 [production GEMM replay](../../exp/dispatch_tuning/RESULTS.md)
adds 42 exact native contracts: 34 faster OA policies and eight reference
retentions, with 22 tile64 and 20 copy-address-reuse selections. Five separately
held-out [rowwise contracts](ideas/shape-aware-rowwise-dispatch.md) select
measured CTA warp counts. Those isolated integrations do not replace the
application backend used here or establish a new whole-model measurement. The
latest same-source ptxas nominal-selected GEMMs remain faster than OA.

## What was compared

The study has three distinct attribution levels:

| Level | OpenAssembler side | Control | What the result establishes |
|---|---|---|---|
| Whole transformer | Production application frontend worker: AOTAutograd forward/backward graphs lowered by TorchInductor/Triton | Eager PyTorch using its CUDA operators and libraries | Performance of the existing application compilation workflow, not native OA SASS superiority |
| Whole residual MLP | Native OA CLI-generated GEMMs in every forward, dX and dW product; shared PyTorch CUDA surrounding operations | Same-PTX ptxas or default cuBLAS GemmEx with the identical prescribed mixed-precision derivative | Effect of replacing GEMM implementations within a complete hybrid NN step |
| Native stages | OA scalar kernels and native tensor GEMMs | Same-PTX ptxas and public PyTorch CUDA operators/cuBLAS | Compiler-versus-source-mapping gaps and shape-dependent kernel behavior |

There is no single CUDA implementation called “naive CUDA.” Eager PyTorch is
the practical application baseline here and uses optimized CUDA libraries.
Same-PTX ptxas isolates the compiler backend from the kernel algorithm. Default
GemmEx is a vendor library dispatch, not an exhaustive cuBLASLt tuning search.

An actual CLI capture and saved replay were also executed independently:
[`cli_proof/summary.json`](../../exp/whole_stack/cli_proof/summary.json).
Both commands returned zero and checked 77 tensors including 15 final parameter
gradients. Capture emitted 35 cubins labelled `torch-inductor-triton`; replay
executed five optimizer steps with zero new GPU code generation. This proves the bundle/replay path;
its three-sample timings are integration checks, not the primary estimates.

## Primary evidence

| Record | Contents |
|---|---|
| [Whole transformer raw](../../exp/whole_stack/runs/whole-stack-20261106T105757.119747Z.json) | Ten configurations, two process-order rounds each, raw samples, state checks, separate profiles, runtime/source hashes |
| [Derived transformer summary](../../exp/whole_stack/runs/summary-whole-stack-20261106T105757.119747Z.json) | Verification-gated ratios, per-round ranges, corrected kernel categories/launch counts and throughput |
| [Native hybrid MLP and isolated GEMMs](../../exp/whole_stack/runs/native-mlp-20261106T105944.492233Z.json) | Two full MLPs, forward/dX/dW, packing, canaries, state checks, shuffled timing and poisoned-output CUDA graphs |
| [Native GEMM counters](../../exp/whole_stack/runs/native-counters-20261106T110405.921716Z.json) | Six NCU profiles on two contrasted shapes |
| [Native SASS inventory](../../exp/whole_stack/native_static.json) | Six same-PTX OA/ptxas pairs, opcode counts, issue controls and unrolling caveats; register resources are in the native MLP raw record |
| [Native scalar neural stages](../../exp/whole_stack/runs/native-ops-20261106T110300.124164Z.json) | Five operator families at two sizes; complete normal/zero outputs, canaries and 600 shuffled event samples |
| [BF16 numerical diagnostic](../../exp/whole_stack/runs/bf16-diagnostic-20261106T110747.459824Z.json) | Initial logits, Q/K/V bias gradients, 29 updates and two separately fixed Adam epsilon policies |

The whole-model run wrote all 20 results and its final status before a local
proxy error during client teardown. A subsequent Modal stop query confirmed
that the application had already stopped. There are no case execution errors
in the saved run; two numerical comparisons failed as detailed below.

## Workloads and timing contract

These are synthetic decoder-transformer configurations from one controlled
architecture, not ten independently pretrained model families. The model
includes token/position embeddings, RMSNorm, QKV projections, causal scaled
dot-product attention, output projection, GELU FFN, residuals and a vocabulary
head. Training includes cross-entropy, backward, gradient clipping and AdamW.
The models range from about 85 thousand to 7 million parameters. This is a
workload coverage study, not a large-model pretraining convergence benchmark.

Each training pair uses eight warmup and twenty measured optimizer steps,
followed by one separately profiled diagnostic update. It uses the same seed,
source, dimensions and resident synthetic inputs. The rounds reverse process
order: eager then compiled, followed by compiled then eager. The processes are
fresh and execute on the same A100 worker. Two rounds give sensitivity evidence,
not a statistical confidence interval over machines, random seeds or datasets.

The CUDA-event `end_to_end` field is **resident-data application-step latency**.
It includes host enqueue gaps between its events, batch indexing, forward/loss,
backward, clipping and optimizer work. It excludes model construction, imports,
compilation, storage/data-loader/H2D costs, `zero_grad` at step start and final
verification copies. The initial two output snapshots occur before measured
steps. Worker/program wall times include initialization, compilation, profiling
and snapshots; they are not isolated compiler-pass timings.

FP32 disables TF32 and uses the highest float32 matmul precision. BF16 uses
autocast with FP32 master parameters. Tolerances are explicit: FP32 outputs and
gradients use `rtol=1e-3, atol=1e-4`; FP32 parameter/optimizer state uses
`rtol=2e-2, atol=2e-3`. BF16 uses `rtol=5e-2, atol=5e-3` for both groups. These
are tolerance checks, not raw-bit equality. The worker observes the first two
owning-model outputs and final parameters, clipped gradients and optimizer
state after the diagnostic update. It does not compare every intermediate.

Inference measures twenty prefill samples and three sequences of sixteen
greedy decode steps after warmup. Decode uses batch one and recomputes a fixed
window; it is not KV-cache serving. Generated sequences are checked for
presence, shape, range and exact eager/compiled token equality.

## Whole transformer results

Times below are medians of the two within-round medians. Speedup ranges are
computed within each matched round as eager/compiled. Training rows measure
the complete resident-data optimizer step; inference rows measure prefill.

| Workload | B / sequence / width / layers | Precision | Eager ms | Compiled frontend ms | Within-round ratio | Numerical gate |
|---|---|---|---:|---:|---:|---|
| Tiny training | 2 / 32 / 64 / 1 | FP32 | 3.513 | 2.841 | 1.20–1.27x | pass both |
| Base training | 8 / 128 / 128 / 2 | FP32 | 5.796 | 4.032 | 1.35–1.53x | pass both |
| Long-context training | 2 / 512 / 256 / 2 | FP32 | 5.461 | 3.880 | 1.39–1.42x | pass both |
| Wide training | 8 / 128 / 512 / 2 | FP32 | 5.800 | 4.380 | 1.24–1.42x | pass both |
| Deep training | 4 / 128 / 256 / 8 | FP32 | 17.202 | 11.007 | 1.53–1.59x | pass both |
| Single prefill | 1 / 128 / 256 / 4 | FP32 | 2.004 | 1.171 | 1.68–1.75x | pass both |
| Long prefill | 8 / 512 / 256 / 4 | FP32 | 4.296 | 3.765 | 1.13–1.15x | pass both |
| Batched prefill | 16 / 256 / 512 / 2 | FP32 | 5.245 | 4.953 | 1.04–1.08x | pass both |
| Wide training | 8 / 128 / 512 / 2 | BF16 | 7.235 | 4.778 | **not admitted** | fail both |
| Batched prefill | 16 / 256 / 512 / 2 | BF16 | 1.306 | 0.947 | 1.37–1.39x | pass both |

Nine workloads have two numerically admitted rounds. Faster-looking timings
for the BF16 training row remain diagnostic only. All four inference workloads
also passed exact generated-token comparison. Their batch-one sixteen-token
fixed-window decode totals were:

| Model configuration | Eager ms | Compiled frontend ms |
|---|---:|---:|
| Single-prefill model | 31.539 | 16.291 |
| Long-prefill model | 37.025 | 16.754 |
| Batched-prefill FP32 model | 17.973 | 10.681 |
| Batched-prefill BF16 model | 20.546 | 13.096 |

The compiled program took roughly 14–28 seconds for the FP32 training
invocations including compilation, profiling and snapshots, versus about
1–4 seconds eagerly. These short runs do not amortize cold compilation. Read
the raw wall-time fields rather than treating steady-step gains as startup
gains. Final peak-memory fields include profiling and are not a pure steady
memory-footprint comparison.

## Training-stage decomposition

Eager/compiled event medians in milliseconds. Each entry is the median of the
two round medians. Stage medians need not sum to the median of the full step.
The optimizer is eager/shared in the application frontend and has no consistent
speedup; most improvement occurs in compiled forward/backward graphs.

| Workload | Forward + loss | Backward | Clip + optimizer |
|---|---:|---:|---:|
| Tiny FP32 | 1.009 / 0.772 | 1.617 / 1.110 | 0.739 / 0.801 |
| Base FP32 | 1.631 / 1.059 | 2.924 / 1.842 | 1.048 / 0.953 |
| Long-context FP32 | 1.562 / 1.044 | 2.804 / 1.907 | 0.933 / 0.778 |
| Wide FP32 | 1.882 / 1.319 | 3.188 / 2.331 | 0.614 / 0.595 |
| Deep FP32 | 5.194 / 3.006 | 9.549 / 5.652 | 2.288 / 2.216 |

Every case has a separate operator/kernel profile. Raw profiler data includes
CUDA-side user annotations (`stage/*`, `Optimizer.*`) that overlap real work.
[`analyze.py`](../../exp/whole_stack/analyze.py) explicitly removes those
ranges, separates DMA events, keeps CPU attribution separate and quarantines
unknown names. All current names are classified; do not sum the unfiltered
`cuda_kernels` fields in the historical raw JSON.

In the base training profile, recognized kernel launches fell from **207 to
123**, with diagnostic kernel-time sums of about 1.746 to 0.996 ms. The
breakdown below uses recognized kernel-name families, not exact semantic
boundaries inside fused kernels:

| Base training kernel family | Eager microseconds | Compiled microseconds |
|---|---:|---:|
| GEMM and split-K kernels | 597.2 | 460.2 |
| Fused attention | 165.3 | 130.0 |
| GELU forward/backward | 22.1 | 18.8 |
| Standalone pointwise kernels | 324.7 | 19.3 |
| Standalone reductions | 255.2 | 18.1 |
| Softmax/loss kernels | 52.1 | 11.7 |
| Multi-tensor optimizer/clip kernels | 231.6 | 187.0 |
| Fused norm/residual/embedding groups | not a separate eager group | 27.7 |
| Other mixed Triton groups | not a separate eager group | 114.7 |

Work removed from standalone pointwise/reduction rows partly moves into fused
groups; it is not all eliminated. Fused attention includes attention matrix
products and softmax, so no independent attention-softmax latency is invented.
Profile sums can overlap across streams, differ from non-profiled timing and
are never substituted for application latency.

In deep training, launches fell from 615 to 367. In BF16 batched prefill, they
fell from 74 to 30. These support fusion and reduced launch/intermediate work
as important application benefits. Those benefits are attributable to the
Inductor/Triton workflow, not to a new OA MMA algorithm.

The FP32 batched-prefill replay is a useful counterexample: compiled GEMM
kernel time increased from roughly 3.328 to 3.593 ms and fused attention from
0.222 to 0.240 ms, while launches fell from 51 to 18 and whole prefill latency
still improved slightly. A whole-workload gain does not prove faster MMA.

## Native OpenAssembler whole-MLP comparison

The network is `x + W2(ReLU(W1(LayerNorm(x))))`. All methods prescribe the same
FP16 operand rounding, FP32 GEMM accumulation/output and explicit backward
packing. FP32 master parameters, LayerNorm, ReLU, residual, MSE and AdamW are
shared CUDA operations. Both forward GEMMs, both dX GEMMs and both dW GEMMs use
the selected native OA / same-PTX ptxas / cuBLAS implementation. Allocation,
casting, transposition, packing and optimizer work are included.

This is an experimental hybrid autograd contract, not a claim of equivalence
to unmodified PyTorch autocast. Both MLPs pass raw-bit equality of complete
OA/ptxas outputs, gradients, parameters and AdamW state after three updates.
All 108 verification GEMMs pass full FP32-reference checks and output canaries.

| Batch / width / hidden | Method | Training step ms | Inference ms |
|---|---|---:|---:|
| 512 / 512 / 2048 | OA | 1.6896 | 0.3635 |
| 512 / 512 / 2048 | ptxas | 1.6916 | 0.3451 |
| 512 / 512 / 2048 | cuBLAS | 1.6056 | 0.3318 |
| 1024 / 1024 / 4096 | OA | 2.1340 | 0.4567 |
| 1024 / 1024 / 4096 | ptxas | 2.0746 | 0.4065 |
| 1024 / 1024 / 4096 | cuBLAS | excluded by state gate | excluded |

Times use fifteen shuffled paired samples after five warmup steps. Earlier
complete runs reversed the tiny OA/ptxas small-training ordering; no reliable
OA whole-training superiority over ptxas is established. OA inference was
consistently slower. The larger cuBLAS comparison failed its unchanged 0.005
gradient-relative-L2 gate at `gradient/w1` (0.006364, maximum absolute
9.84e-6). It remains excluded; stateless GEMM controls are checked separately.

Separate diagnostic stages retain both linears' forward cast/pack, forward
GEMM, backward transposes, dX, dW, bias reductions, normalization, activation,
loss and optimizer timings. For the larger OA MLP, instrumented backward
packing cost about 0.171/0.179 ms per linear versus roughly 0.09–0.12 ms per
GEMM; instrumented AdamW was about 0.72 ms. Extra event scopes inflate these
short stages, so these are non-additive diagnostics, not an exact whole-step
budget. They nevertheless identify real surrounding integration work that a
pure GEMM comparison omits.

### Isolating the GEMM backend

Preallocated ten-launch CUDA graphs, nine shuffled replay samples, full-output
poison-after-capture checks and canaries isolate the actual kernels. cuBLAS is
bound to the capture stream. Packing and allocation are outside these timings.

| M / N / K | OA ms | Same-PTX ptxas ms | Default GemmEx ms |
|---|---:|---:|---:|
| 512 / 512 / 2048 | 0.05960 | 0.04157 | 0.01546 |
| 512 / 2048 / 512 | 0.01997 | 0.01393 | 0.02427 |
| 1024 / 1024 / 4096 | 0.11387 | 0.07956 | 0.07834 |
| 1024 / 4096 / 1024 | 0.08448 | 0.05990 | 0.05837 |
| 2048 / 512 / 512 | 0.02109 | 0.01454 | 0.02611 |
| 4096 / 1024 / 1024 | 0.08786 | 0.06236 | 0.06963 |

OA beats the selected default vendor dispatch on two small expanded/tall
shapes. ptxas beats OA on **all six identical-PTX shapes**, by about 29–31%
in time reduction (OA is about 41–45% slower). Those two vendor wins therefore
support shape/algorithm selection rather than superior OA SASS scheduling.
No exhaustive cuBLASLt selection or all-shape OA autotuning was performed.

### Counters and source-level explanations

All six NCU replays succeeded. On each contrasted shape all three methods
execute exactly **262,144 tensor instructions**. OA/ptxas show zero measured
shared-load bank conflicts. Representative counters are:

| M / N / K | Method | Tensor active | Short scoreboard | Long scoreboard | Total instructions |
|---|---|---:|---:|---:|---:|
| 512 / 512 / 2048 | OA | 48.16% | 10.00% | 11.38% | 577,856 |
| 512 / 512 / 2048 | ptxas | 69.67% | 2.75% | 7.05% | 433,408 |
| 512 / 512 / 2048 | cuBLAS | 52.88% | 5.21% | 4.79% | 889,088 |
| 512 / 2048 / 512 | OA | 37.61% | 8.34% | 19.04% | 680,192 |
| 512 / 2048 / 512 | ptxas | 55.19% | 1.74% | 9.42% | 432,640 |
| 512 / 2048 / 512 | cuBLAS | 26.21% | 4.02% | 13.78% | 1,521,101 |

The first OA/ptxas grid contains only 16 CTAs; cuBLAS selects a 64×64 tile and
64 CTAs. Better device coverage is a concrete shape/dispatch difference. On
expanded N, the selected cuBLAS grid is `(16,8,2)` and has much more instruction
work. NCU time is replay time, not the graph time above; these observations
support hypotheses rather than proving the cost of each cause independently.

Static SASS also shows OA's 282 MOVs and uniform stall8 on its 512 static HMMA
words. On K2048/4096, ptxas has one MOV and mixed HMMA controls, including many
stall1 words. OA uses fewer registers (216 versus 231/234), with zero reported
local-memory bytes, yet remains slower. Register count and MMA count alone
do not predict performance. For smaller K, ptxas unrolls substantially more,
so its larger static instruction counts are not more mathematical work.
Removing MOVs or shortening stalls is not automatically safe: proven
accumulator, async-copy, LDSM address-WAR and synchronization edges remain
requirements. Operand-collector causality is still unisolated.

## Native scalar neural stages

All ten cases pass both full normal/zero-output checks against FP64 formulas
and all OA/ptxas canaries. Twenty shuffled samples per method use the validated
nonzero inputs. ReLU is exact; GELU, softmax, LayerNorm and the Adam-like formula
have separate predeclared tolerances. Normal ReLU and softmax are OA/ptxas
bitwise equal; normal GELU, norm and Adam-like outputs are tolerance-only.

| Stage / shape | OA ms | Same-PTX ptxas ms | Public PyTorch CUDA ms |
|---|---:|---:|---:|
| ReLU / 1,048,576 elements | 0.02304 | 0.02355 | 0.02150 |
| ReLU / 4,194,304 elements | 0.04608 | 0.04608 | 0.03226 |
| Tanh GELU / 1,048,576 | 0.02253 | 0.02355 | 0.02560 |
| Tanh GELU / 4,194,304 | 0.04659 | 0.04710 | 0.03738 |
| Adam-like / 1,048,576 | 0.03994 | 0.03584 | 0.12288 |
| Adam-like / 4,194,304 | 0.11110 | 0.10138 | 0.36864 |
| Softmax / 128×128 | 0.11264 | 0.08448 | 0.02048 |
| Softmax / 1024×512 | 0.40704 | 0.30618 | 0.02150 |
| LayerNorm / 128×128 | 0.09830 | 0.08499 | 0.03482 |
| LayerNorm / 1024×512 | 0.35533 | 0.29901 | 0.03686 |

The serial reduction examples measured in this November 6 table assign a
whole row to one scalar thread and loop across columns. Adjacent lanes read
strided rows and these shapes launch only one or eight CTAs. ptxas is slow on
the same source too: this exposed a major source-mapping gap in addition to the
OA compiler gap. Subsequent production [softmax](ideas/cooperative-softmax.md)
and [population LayerNorm](ideas/cooperative-layernorm.md) use warp-per-row
cooperative reductions, and five exact rowwise contracts now choose measured
CTA geometry. Their separate cohorts do not overwrite the old serial table or
prove a complete training normalization interface. The cycle-level breakdown
of remaining reduction gaps still requires matched probes/counters.

The Adam-like formula is fused into one native kernel while its eager PyTorch
control launches several operations and materializes intermediates. ptxas's
fused version is faster than OA. This is a fusion benefit, **not** a result
against tuned fused AdamW, nor the bias-corrected optimizer used in the whole
transformer. PyTorch reduction outputs allocate whereas native outputs are
preallocated; PyTorch is still much faster on those reductions. Event spans
include allocation/host enqueue effects. Small GELU wins are size-dependent
and have differing numerical error; no universal activation win is claimed.

## BF16 training failure and diagnostic

Both whole-model BF16 training rounds fail at key-bias parameters in the first
QKV projection after 29 updates. Four of 1,536 QKV bias entries fail the
declared elementwise gate; maximum absolute drift is approximately 0.00707 and
0.00695. The run is not relabelled successful because its loss looks close.

The separate diagnostic uses identical initialization and data, the same
AOTAutograd/Inductor lowering and unchanged numerical gate. It compares all
524,288 initial logits, Q/K/V bias gradients, clipped gradients, first-step
updates and final parameters. Initial logits have relative L2 0.002706 and
maximum absolute difference 0.015625, within the declared mixed tolerance.
First losses are 6.3812256 versus 6.3815293.

At the first step, key-bias gradient L2 is about 1.8e-6, compared with query
bias 5.8e-4 and value bias 8.4e-3. Layer-zero key-bias gradient signs disagree
on 146/512 entries. With Adam epsilon 1e-8, the maximum key-bias parameter
difference reaches 0.006191 after 29 updates and again fails the gate. In a
separately fixed matched pair with epsilon 1e-4, key-bias drift falls to
4.67e-6/7.85e-6 and no parameter tensor fails that same gate.

In ideal arithmetic, a common key bias adds an equal offset to every score
for each query, which softmax cancels. Tiny finite-precision gradients in this
nearly invariant direction can therefore be amplified by Adam's small
denominator. The gradient measurements and epsilon intervention strongly
support that explanation. They do not prove every implementation difference
is benign. Epsilon changes the training policy and loss trajectory; the
diagnostic does not fix or override the failed original experiment and is not
promoted into production.

## What to optimize next

Subsequent work is documented separately in [direct pointer homes](ideas/direct-pointer-homes.md),
[cooperative softmax](ideas/cooperative-softmax.md),
[LayerNorm](ideas/cooperative-layernorm.md), and
[complete state diagnostics](ideas/state-verification-diagnostics.md).

The current-source BF16 diagnostic was rerun on November 9
([raw record](../../exp/whole_stack/runs/bf16-diagnostic-20261109-current.json)).
At the recorded checkpoints, step 1 passes the parameter gate and step 29
fails it in two QKV key-bias tensors with the original epsilon `1e-8`.
The diagnostic records only steps 1 and 29, so it does not locate the first
failing update or operation. With a separately matched epsilon `1e-4` policy,
no parameter tensor fails at those checkpoints. This compares eager execution
with AOTAutograd/Inductor, not native OA kernels; it remains a numerical and
optimizer-policy diagnosis rather than a native compiler correctness repair.

A separate [30-update boundary diagnostic](../../exp/whole_stack/runs/bf16-first-observed-divergence-20261109T111158.926633Z.json)
on November 9 records 151 required checkpoints on an A100 80GB PCIe with
PyTorch 2.7.1/CUDA 12.8. It keeps epsilon `1e-8` and the original mixed gates.
The first observed full-logit gate failure is **step 3 before the update**:
2 of 524,288 BF16 elements fail, despite relative L2 of only 0.002881. The first
parameter-gate failure is **step 22 after the update**, in both QKV bias tensors
(one key-bias entry each; maximum absolute differences 0.005915 and 0.005558).
Raw/clipped gradients, Adam moment/step tensors and optimizer settings stay
inside their declared gates at every recorded boundary; this does not make
their histories bitwise equal or repair the failed parameters. The diagnostic
completes all checkpoints and fails 53 of them, including repeated observations
of already divergent parameter state. It compares eager execution with
Inductor and adds separate untimed full-logit forwards. It neither identifies
the first arithmetic operation that caused a difference nor establishes native
OA correctness. See the [state-trajectory method](ideas/state-verification-diagnostics.md)
for reproduction and scope.
The measurements here describe the original study, not those later compiler changes.

1. Extend the implemented cooperative forward reductions to production
   backward/affine/saved-statistic contracts, and test load/tree/synchronization
   overlap. The old serial source-mapping gap was addressed by the forward
   rowwise kernels; five exact CTA-geometry wins do not close the remaining
   ptxas gap or supply training interfaces.
2. Reduce native tensor address/control/copy work and jointly search fragment
   ordering/register liveness. The equal-MMA, zero-conflict counter cases make
   this more targeted than another unconditional swizzle sweep.
3. Extend the exact-contract dispatch table beyond the completed 42-contract
   replay only with current binaries and independent held-out evidence.
   Twenty-two current contracts select tile64, 20 select copy-address reuse,
   two select Morton8 and eight retain reference. Keep earlier shape regressions
   visible and compare each frozen policy with its actual ptxas/library control;
   kernel gains still need a controlled graph and full-state integration.
4. Reduce dX/dW transposed packing and repeated conversion overhead, while
   measuring complete steps and keeping the same derivative/rounding contract.
5. Preserve numerical gates in multi-step training. Treat near-invariant
   gradients, reduction ordering and optimizer sensitivity as part of the
   contract; do not silently loosen tolerances or change epsilon to admit speed.
6. Add a stock compiled-PyTorch/tuned-library baseline, more model families,
   larger vocabularies, genuine KV-cache decode and sustained BF16 training.
   The present frontend wins are against eager execution and are not a native
   OA LLM training acceleration claim.

## Failed and superseded attempts

The first whole-model submission failed on a remote `__file__` path assumption
before workload execution. A tiny run then passed verification but failed a
JSON-reporting print call. Another early submission reached inference after
training, but failed its strict FP32/TF32-enabled output tolerance (maximum
absolute 0.000217); it did not publish a complete suite and is excluded.
The current FP32 run disables TF32 and records its own contract and hashes.

`base-profile.json` is an earlier smoke measurement with missing final-gradient
coverage and overlapping profiler rows; `smoke-fixed.json` is the subsequent
single-case validation. Neither is pooled into the final paired study.

Native setup initially lacked a materialized current CUDA context. An earlier
native MLP submission also aborted on the larger cuBLAS numerical gate before
partial-result handling was added. Both are documented in
[the native-hybrid-MLP dossier](ideas/native-hybrid-mlp.md). Two earlier
scalar-op records are superseded: one added artificial PyTorch output copies;
the next timed a different random input distribution from its checked inputs.
The primary scalar record fixes both. The larger MLP cuBLAS state failure and
both BF16 transformer failures remain failures under unchanged gates.

Reproduction commands, local notes, manifests, source hashes and raw samples
are indexed in [`exp/whole_stack/README.md`](../../exp/whole_stack/README.md).
