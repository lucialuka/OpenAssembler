# Fused local dot products for backward row reductions

Status: validated-not-integrated

This is an experimental source-template optimization under `exp/nn_round2/`.
It is **not installed in the production compiler or `rowwise` command**. The
measured gain is over an unfused native OpenAssembler template. Same-source
ptxas still runs faster in every measured candidate case.

## Idea in plain language

A backward normalization or softmax kernel must add several products within each
thread, then combine the partial sums across a warp. The first experimental
version computed every product into a different register and subsequently added
them. Those temporary products stay live together and use registers.

The candidate accumulates each new product directly with a fused multiply-add
(FMA). It needs fewer instructions and fewer simultaneously live products. It
then performs exactly the original warp reductions. No shuffle, synchronization,
address dependency, mask or output check is removed.

This is a local per-thread transformation, not fusion of different network
layers. Explicit FMA rounds its multiplication and addition once. The original
PTX multiply/add forms permit ptxas contraction, so the vendor control already
uses FMA in several places. We validate both compiler results independently
against the same mathematical reference; bitwise equality is not promised.

## Operators and numerical contract

All arrays are contiguous FP32. The backward operators return the input gradient
only: there are no learned normalization scale/bias parameters or their gradients.
For a row, `mean` below divides by its full, unpadded width:

- Softmax backward consumes probabilities `p` and upstream gradient `g`, and
  returns `p * (g - sum(p*g))`. Its gate is `rtol=1e-4, atol=1e-5`.
- RMSNorm uses `r = 1/sqrt(mean(x*x) + 1e-5)`. Its input gradient is
  `r * (g - x*r*r*mean(x*g))`. The backward gate is
  `rtol=2e-4, atol=5e-5`.
- Population LayerNorm centers `x` before its variance and dot-product reductions.
  Its backward gate is also `rtol=2e-4, atol=5e-5`.
- The forward LayerNorm/RMSNorm controls use `rtol=1e-4, atol=2e-5`.

References evaluate the formulas in FP64 on the exact FP32 inputs. All tolerances
were inherited from the initial arithmetic campaign and remain unchanged. The
contract covers the tested bounded finite inputs, not arbitrary NaN/Inf,
overflow or every representable floating-point value.

Every complete output and both 256-element guard regions are checked for normal,
zero and structured inputs. The structured LayerNorm case has a large common
offset and small variance. Input bits must remain unchanged. Captured graphs are
poisoned after capture, replayed, and checked again before timing. A failed
method pair is excluded from timing.

## Method and retained negative results

The main study contains five operator families, widths 128/256/513, four/eight
warps per CTA, an unfused and FMA template, 257/4096 rows and two reversed-order
rounds: **240 cases, all passing**. The extra lane/row masks exercise boundaries.
Four/eight warps mean 128/256 threads per CTA; there is no shared-memory scratch.

Inputs are seeded by operation, width and row count. The exact same tensors are
therefore used across template versions, CTA geometries and rounds. This repairs
a limitation of the initial broad operator sweep, whose compiler arms shared
inputs within each case but whose different source variants used different
random seeds.

Each source is independently compiled by native OA and by CUDA 12.8 ptxas `-O3`.
There is no compiler fallback. Native SASS is checked by nvdisasm. The worker
records cubin/source identities and driver-reported registers/resources. The
first run uses 17 shuffled OA/ptxas graph-event samples, 32 operations per graph,
and then reverses the overall candidate order for its second round. The two
templates are compared across those blocks; they are not alternated inside each
individual timing sample.

The hardware is **NVIDIA A100-SXM4-40GB**, SM80, 108 SMs, with PyTorch
2.7.1+cu128/CUDA 12.8 in the measured worker. Timings are resident-data graph
replays. Module loading, compilation and host launch overhead are outside them.

Fewer registers do not guarantee a faster kernel. Forward norms were mostly
neutral and some settings regressed by about 4%. Small backward shapes were
mostly neutral, with several small LayerNorm gains. There is no basis for
turning this on globally.

## Independently confirmed selected results

A separate Modal process repeated the two strongest backward families at
4096×513 with both CTA geometries: **16/16 cases pass**. It uses 31 shuffled
samples, 64 operations per graph, the same three input patterns and two reversed
candidate-order rounds. The original compiled cubins and mathematical gates are
unchanged.

Microseconds below are the two order rounds in that confirmation process. Ratios
are paired native-baseline/native-FMA within each round:

| Operation / threads | Native unfused OA µs | Native FMA OA µs | Native improvement | FMA same-PTX ptxas µs |
|---|---:|---:|---:|---:|
| RMSNorm backward / 128 | 16.176 / 16.480 | 11.808 / 12.416 | 1.370× / 1.327× | 9.504 / 9.536 |
| RMSNorm backward / 256 | 16.848 / 16.880 | 12.384 / 12.336 | 1.360× / 1.368× | 10.112 / 10.464 |
| Softmax backward / 128 | 15.232 / 15.216 | 12.240 / 12.256 | 1.244× / 1.242× | 9.536 / 9.520 |
| Softmax backward / 256 | 15.344 / 16.144 | 11.936 / 11.920 | 1.286× / 1.354× | 10.416 / 10.400 |

The first, broader process also found large gains for these settings: about
1.23–1.34×. Its absolute latencies are retained separately rather than pooled
with the confirmation process. The measured width is **513**, a boundary case,
so these results do not establish the same gain for standard 512/1024-wide
training layers.

For width-513 RMSNorm backward, OA registers fall from 61 to 44 and its generated
instruction count falls from 238 to 205. Those observations support a hypothesis
about fewer live products and better occupancy/issue behavior. They do **not**
measure active occupancy, operand-collector delays or a causal stall reduction;
those require hardware counters. Synchronization remains intact.


## Follow-up at standard model widths: November 7

A separate worker tested the same two backward operators at widths **512 and
1024**, with 1024/4096 rows, 128/256 threads, both templates and two reversed-order
rounds: **64/64 cases pass**. Input seeds, three patterns, full outputs, unchanged
inputs, guard checks and poisoned graph replay use the same rules. Tolerances and
epsilon are unchanged. The timing protocol is 17 shuffled compiler-pair samples
and 32 operations per graph. Its actual device is again A100-SXM4-40GB.

At **4096 rows**, the gains extend to these standard widths. These are fresh
within-worker pairs; absolute times are not pooled with the width-513 runs:

| Backward operator / width / threads | Unfused OA µs, two rounds | FMA OA µs, two rounds | Native improvement, two rounds | FMA same-PTX ptxas µs, two rounds |
|---|---:|---:|---:|---:|
| RMSNorm / 512 / 128 | 15.712 / 16.096 | 12.320 / 12.320 | 1.275× / 1.306× | 10.624 / 10.496 |
| RMSNorm / 512 / 256 | 16.000 / 16.128 | 12.384 / 12.352 | 1.292× / 1.306× | 10.688 / 10.688 |
| RMSNorm / 1024 / 128 | 63.840 / 63.808 | 51.744 / 51.808 | 1.234× / 1.232× | 40.832 / 40.832 |
| RMSNorm / 1024 / 256 | 64.416 / 64.064 | 52.320 / 52.320 | 1.231× / 1.224× | 40.576 / 40.832 |
| Softmax / 512 / 128 | 13.728 / 13.984 | 11.648 / 12.032 | 1.179× / 1.162× | 9.536 / 9.568 |
| Softmax / 512 / 256 | 14.048 / 14.688 | 10.720 / 11.712 | 1.310× / 1.254× | 9.216 / 9.696 |
| Softmax / 1024 / 128 | 62.272 / 62.048 | 49.856 / 51.072 | 1.249× / 1.215× | 40.320 / 40.480 |
| Softmax / 1024 / 256 | 62.592 / 62.496 | 51.904 / 51.552 | 1.206× / 1.212× | 40.864 / 40.736 |

At **1024 rows**, the changes were mostly neutral or slower, including about a
5–6% loss in the first 512-wide/128-thread round and no repeated large gain.
Those negative controls rule out a width-only global dispatch policy. Row count,
working-set size and scheduling all matter. The much larger absolute latency at
4096×1024 also merits L2/DRAM and occupancy counters; this measurement alone does
not isolate the cause.

OA registers fall from 58 to 42 for width-512 RMSNorm backward and from 106 to
74 at width 1024. Softmax backward falls from 53 to 41 and 101 to 73 respectively.
These resource reductions accompany the selected large-shape gains, but ptxas
still beats every FMA candidate. Standard-width gains have two order rounds in
one process; the separate-process confirmation above is specifically for 513.
A whole-training-step test and independent standard-width confirmation remain
necessary before promotion.

## Reproduce and inspect

From the repository root, use Python 3.12+ for native compilation, CUDA 12.8
tools and an authenticated Modal client for A100 execution. `python3` in the
Modal commands means the Python environment in which the Modal package is
installed; it is only the client transport, not the compiler interpreter.

```sh
.venv/bin/python -m pytest -q exp/nn_round2/test_operator_suite.py \
  exp/nn_round2/test_operator_candidates.py
.venv/bin/python exp/nn_round2/operator_candidates.py --build
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_candidates_modal.py \
  --samples 17 --rounds 2
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_fma_confirm_modal.py \
  --samples 31 --rounds 2
```

Both launchers save timestamped raw JSON under `exp/nn_round2/runs/`, including
failures. To regenerate the published summaries from their retained raw records:

```sh
python3 exp/nn_round2/operator_fma_analysis.py \
  exp/nn_round2/runs/operator-fma-20261107T143537.650425Z.json \
  exp/nn_round2/operator_fma_analysis.json
python3 exp/nn_round2/operator_fma_analysis.py \
  exp/nn_round2/runs/operator-fma-confirm-20261107T144005.532354Z.json \
  exp/nn_round2/operator_fma_confirmation_analysis.json
```

Generated source, both cubins, manifests and native disassembly are under
`exp/nn_round2/operator_fma_build/`. They are regenerable artifacts; the source
and evidence identities in the raw records define the measured revision.


For the separate standard-width follow-up:

```sh
.venv/bin/python -m pytest -q exp/nn_round2/test_operator_standard_fma.py
.venv/bin/python exp/nn_round2/operator_standard_fma.py
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_standard_fma_modal.py \
  --samples 17 --rounds 2
python3 exp/nn_round2/operator_fma_analysis.py \
  exp/nn_round2/runs/operator-standard-fma-20261107T144401.126097Z.json \
  exp/nn_round2/operator_standard_fma_analysis.json
```

Its generated artifacts are under `exp/nn_round2/operator_standard_fma_build/`.

The follow-up streamed load/FMA overlap candidate is documented in
[`exp/weaknesses/reduction_overlap_report.md`](../../../exp/weaknesses/reduction_overlap_report.md).
Its source-inclusive A100 run passed full output/tolerance, canary,
input-immutability and graph-poison checks for 4096-row RMSNorm and softmax
backward at widths 512, 513 and 1024 with four and eight warps. It reduced OA
instructions and registers and beat the unfused OA candidate across the tested
matrix, but matched ptxas remained faster in every case. Nsight counters for
the candidate are still required before any promotion decision.

## Code and evidence

- [Source transformation, exact formulas and stable-input worker](../../../exp/nn_round2/operator_candidates.py).
- [Original operator source generator](../../../exp/nn_round2/operator_suite.py),
  [full-output/guard/graph validator](../../../exp/nn_round2/operator_worker.py),
  [CPU precision and synchronization tests](../../../exp/nn_round2/test_operator_candidates.py).
- [Full 240-case run](../../../exp/nn_round2/runs/operator-fma-20261107T143537.650425Z.json),
  [its mechanical summary](../../../exp/nn_round2/operator_fma_analysis.json).
- [Standard-width source](../../../exp/nn_round2/operator_standard_fma.py),
  [64-case standard-width run](../../../exp/nn_round2/runs/operator-standard-fma-20261107T144401.126097Z.json),
  [standard-width summary](../../../exp/nn_round2/operator_standard_fma_analysis.json).
- [Independent confirmation](../../../exp/nn_round2/runs/operator-fma-confirm-20261107T144005.532354Z.json),
  [its mechanical summary](../../../exp/nn_round2/operator_fma_confirmation_analysis.json).

The next admission tests are independent confirmation of standard-width choices,
hardware-counter analysis and a composed training workload with matching
saved-state/gradient contracts.
No production implementation or whole-transformer speedup follows from this
standalone result.
