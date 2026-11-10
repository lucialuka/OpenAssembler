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

## Current-source revalidation: November 9

The current experiment runner was rebuilt against the present checkout and
executed for 64 standard-width cases on an A100-SXM4-80GB with CUDA 12.8. All
64 cases pass complete outputs, canaries, unchanged-input checks, poisoned graph
replay and exact OA/ptxas comparisons. The FMA templates still reduce selected
OA work, but the ptxas/OA timing ratios range from about 0.57 to 0.88 across
the retained widths and row counts, so ptxas remains faster. This confirms the
candidate is a useful source-level reduction experiment, not a production
rowwise dispatch rule. The source-hashed record is
[`operator-standard-fma-20261109T063211.248802Z.json`](../../../exp/nn_round2/runs/operator-standard-fma-20261109T063211.248802Z.json).

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
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_standard_fma_modal.py \
  --samples 7 --rounds 2
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

## Streamed load/FMA follow-up and counters

The source-inclusive [overlap run](../../../exp/weaknesses/reduction_overlap_run_20261108T051551.705065Z.json)
passes full normal/zero/structured outputs, guards, input immutability and
poisoned graph replay for 4096-row RMSNorm and softmax backward, widths
512/513/1024, and four/eight warps per CTA. The candidate loads one chunk of
each operand, immediately folds its products into local FMA accumulators, then
loads the next chunk. Tail masks, the five-step full-mask shuffle tree and
output stores stay unchanged. Normal and structured results pass the declared
FP32 tolerance but are not generally bitwise equal to ptxas; zero results are.

Representative RMSNorm baseline/overlap static instruction and register counts
are 232/58 → 200/43 at width 512, 240/61 → 208/45 at width 513 and 392/106 →
328/75 at width 1024. Softmax changes from 152/53 → 136/41, 160/56 → 144/43
and 264/101 → 232/73 respectively. The candidate beats unfused OA in this
matrix; ptxas remains faster in every case. These are historical source-bound
measurements, not a current production reduction policy.

The subsequent [Nsight record](../../../exp/weaknesses/reduction_overlap_ncu_20261108T061002.104721Z.json)
completed **all 12 cases**, with baseline and overlap sources independently
compiled by OA and ptxas and 21 accepted metric names. The worker profiles the
third exact launch, validating full outputs, canaries and input immutability
before and after replay. Barrier stalls are zero because these kernels have no
CTA barriers. Counter replay duration is never used as benchmark timing.

The overlap version cuts OA dynamic instructions by **10–16%** and registers by
**12–31**, but OA long-scoreboard fractions remain about **82–91%**, against
about **19–49%** for ptxas. DRAM/L2 traffic changes by only a few percent.
Eligibility changes are small: fewer registers and scalar instructions alone
do not solve memory/reduction dependency overlap. Stall percentages are
sampling diagnostics and are not additive time fractions or causal proof.

The complete retained per-case table follows. Inst/FMA are dynamic counts;
registers are per thread, DRAM traffic is bytes, and eligible warps is the
reported scheduler average. Absolute values stay within this A100-SXM4-40GB
counter process and are not pooled with newer devices or timing rounds.

| Op | Width | Warps | Method | Inst | FMA | Regs | Long SB % | Short SB % | Wait % | DRAM bytes | L2 hit % | Eligible warps |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rmsnorm_backward | 512 | 4 | baseline oa | 921600 | 532480 | 58 | 86.64 | 1.99 | 1.22 | 26915072 | 50.30 | 0.17 |
| rmsnorm_backward | 512 | 4 | baseline ptxas | 626688 | 331776 | 40 | 48.56 | 11.18 | 1.94 | 26200832 | 50.28 | 0.12 |
| rmsnorm_backward | 512 | 4 | overlap oa | 798720 | 409600 | 43 | 86.34 | 2.23 | 0.96 | 27186304 | 51.02 | 0.19 |
| rmsnorm_backward | 512 | 4 | overlap ptxas | 626688 | 331776 | 40 | 47.74 | 11.47 | 2.40 | 26072576 | 50.52 | 0.14 |
| rmsnorm_backward | 512 | 8 | baseline oa | 921600 | 532480 | 58 | 82.32 | 2.37 | 1.37 | 27106048 | 50.17 | 0.19 |
| rmsnorm_backward | 512 | 8 | baseline ptxas | 626688 | 331776 | 40 | 46.61 | 11.74 | 2.21 | 25986560 | 50.68 | 0.16 |
| rmsnorm_backward | 512 | 8 | overlap oa | 798720 | 409600 | 43 | 85.47 | 2.08 | 0.97 | 27013760 | 50.78 | 0.21 |
| rmsnorm_backward | 512 | 8 | overlap ptxas | 626688 | 331776 | 40 | 45.85 | 10.22 | 2.29 | 25823360 | 50.18 | 0.17 |
| rmsnorm_backward | 513 | 4 | baseline oa | 974848 | 561152 | 61 | 85.32 | 2.13 | 1.25 | 27439104 | 53.02 | 0.19 |
| rmsnorm_backward | 513 | 4 | baseline ptxas | 671744 | 352256 | 48 | 28.74 | 6.06 | 1.97 | 26003456 | 52.28 | 0.17 |
| rmsnorm_backward | 513 | 4 | overlap oa | 843776 | 430080 | 45 | 87.43 | 2.06 | 0.81 | 27517312 | 55.45 | 0.18 |
| rmsnorm_backward | 513 | 4 | overlap ptxas | 671744 | 352256 | 48 | 28.67 | 6.92 | 2.06 | 25956992 | 52.19 | 0.18 |
| rmsnorm_backward | 513 | 8 | baseline oa | 974848 | 561152 | 61 | 84.36 | 2.12 | 1.21 | 27488000 | 52.93 | 0.19 |
| rmsnorm_backward | 513 | 8 | baseline ptxas | 671744 | 352256 | 48 | 26.11 | 4.99 | 2.15 | 26589056 | 53.68 | 0.22 |
| rmsnorm_backward | 513 | 8 | overlap oa | 843776 | 430080 | 45 | 85.56 | 2.32 | 0.87 | 27323776 | 55.01 | 0.20 |
| rmsnorm_backward | 513 | 8 | overlap ptxas | 671744 | 352256 | 48 | 27.02 | 5.50 | 1.75 | 26281856 | 53.21 | 0.18 |
| rmsnorm_backward | 1024 | 4 | baseline oa | 1576960 | 991232 | 106 | 87.19 | 1.75 | 1.46 | 54762752 | 49.69 | 0.09 |
| rmsnorm_backward | 1024 | 4 | baseline ptxas | 1085440 | 593920 | 72 | 34.73 | 7.50 | 2.69 | 51831936 | 50.20 | 0.13 |
| rmsnorm_backward | 1024 | 4 | overlap oa | 1323008 | 737280 | 75 | 85.85 | 2.01 | 0.94 | 51736192 | 50.46 | 0.11 |
| rmsnorm_backward | 1024 | 4 | overlap ptxas | 1085440 | 593920 | 72 | 34.57 | 7.70 | 2.88 | 51564416 | 49.64 | 0.14 |
| rmsnorm_backward | 1024 | 8 | baseline oa | 1576960 | 991232 | 106 | 86.10 | 1.65 | 1.46 | 55053952 | 50.17 | 0.09 |
| rmsnorm_backward | 1024 | 8 | baseline ptxas | 1085440 | 593920 | 72 | 37.56 | 7.72 | 2.74 | 51968000 | 49.97 | 0.12 |
| rmsnorm_backward | 1024 | 8 | overlap oa | 1323008 | 737280 | 75 | 86.59 | 1.69 | 0.85 | 52047232 | 50.66 | 0.10 |
| rmsnorm_backward | 1024 | 8 | overlap ptxas | 1085440 | 593920 | 72 | 37.73 | 7.12 | 2.86 | 51939072 | 50.14 | 0.13 |
| softmax_backward | 512 | 4 | baseline oa | 610304 | 299008 | 53 | 89.22 | 1.35 | 0.89 | 26935168 | 50.45 | 0.11 |
| softmax_backward | 512 | 4 | baseline ptxas | 495616 | 229376 | 40 | 30.70 | 8.17 | 2.10 | 25818368 | 50.29 | 0.15 |
| softmax_backward | 512 | 4 | overlap oa | 548864 | 237568 | 41 | 91.53 | 0.97 | 0.67 | 26887552 | 50.79 | 0.09 |
| softmax_backward | 512 | 4 | overlap ptxas | 495616 | 229376 | 40 | 30.94 | 8.51 | 2.22 | 25815296 | 50.40 | 0.16 |
| softmax_backward | 512 | 8 | baseline oa | 610304 | 299008 | 53 | 87.07 | 1.36 | 1.04 | 26999296 | 50.10 | 0.10 |
| softmax_backward | 512 | 8 | baseline ptxas | 495616 | 229376 | 40 | 29.83 | 7.33 | 2.18 | 25898112 | 50.49 | 0.19 |
| softmax_backward | 512 | 8 | overlap oa | 548864 | 237568 | 41 | 87.54 | 1.08 | 0.82 | 26977664 | 50.84 | 0.12 |
| softmax_backward | 512 | 8 | overlap ptxas | 495616 | 229376 | 40 | 29.90 | 7.56 | 2.19 | 25854720 | 50.51 | 0.19 |
| softmax_backward | 513 | 4 | baseline oa | 651264 | 315392 | 56 | 89.87 | 1.21 | 0.76 | 27249408 | 53.82 | 0.12 |
| softmax_backward | 513 | 4 | baseline ptxas | 532480 | 245760 | 47 | 28.37 | 7.97 | 2.24 | 26114816 | 52.31 | 0.17 |
| softmax_backward | 513 | 4 | overlap oa | 585728 | 249856 | 43 | 90.05 | 1.23 | 0.79 | 27486976 | 55.65 | 0.11 |
| softmax_backward | 513 | 4 | overlap ptxas | 532480 | 245760 | 47 | 28.66 | 8.04 | 1.95 | 26055040 | 52.06 | 0.14 |
| softmax_backward | 513 | 8 | baseline oa | 651264 | 315392 | 56 | 86.87 | 1.37 | 0.97 | 27803904 | 53.03 | 0.12 |
| softmax_backward | 513 | 8 | baseline ptxas | 532480 | 245760 | 47 | 27.25 | 6.87 | 1.73 | 26346368 | 54.10 | 0.16 |
| softmax_backward | 513 | 8 | overlap oa | 585728 | 249856 | 43 | 89.37 | 1.10 | 0.74 | 27548032 | 55.12 | 0.11 |
| softmax_backward | 513 | 8 | overlap ptxas | 532480 | 245760 | 47 | 26.46 | 7.03 | 2.07 | 26418944 | 54.02 | 0.19 |
| softmax_backward | 1024 | 4 | baseline oa | 1069056 | 561152 | 101 | 89.64 | 1.22 | 1.24 | 55245824 | 49.81 | 0.05 |
| softmax_backward | 1024 | 4 | baseline ptxas | 888832 | 425984 | 72 | 18.79 | 6.76 | 1.92 | 52167552 | 49.66 | 0.11 |
| softmax_backward | 1024 | 4 | overlap oa | 942080 | 434176 | 73 | 88.62 | 1.18 | 0.81 | 51734400 | 50.37 | 0.06 |
| softmax_backward | 1024 | 4 | overlap ptxas | 888832 | 425984 | 72 | 18.73 | 6.61 | 2.12 | 52236032 | 49.76 | 0.12 |
| softmax_backward | 1024 | 8 | baseline oa | 1069056 | 561152 | 101 | 88.68 | 1.23 | 1.23 | 55189248 | 49.88 | 0.05 |
| softmax_backward | 1024 | 8 | baseline ptxas | 888832 | 425984 | 72 | 19.07 | 8.22 | 2.56 | 52169472 | 49.86 | 0.14 |
| softmax_backward | 1024 | 8 | overlap oa | 942080 | 434176 | 73 | 90.77 | 0.95 | 0.73 | 52353152 | 50.33 | 0.06 |
| softmax_backward | 1024 | 8 | overlap ptxas | 888832 | 425984 | 72 | 19.45 | 8.63 | 2.30 | 52443136 | 50.10 | 0.13 |

Reproduce the bounded source and hardware studies with:

```sh
.venv/bin/python -m pytest -q exp/weaknesses/test_reduction_overlap_codegen.py
modal run exp/weaknesses/reduction_overlap_run_modal.py
modal run exp/weaknesses/reduction_overlap_ncu_modal.py
```

The [source generator](../../../exp/weaknesses/reduction_overlap_codegen.py)
and [initial build gate](../../../exp/weaknesses/reduction_overlap_build_modal_result.json)
remain available. The latter checks PTX/disassembly only. Promotion still
requires current-source paired held-out timing and the applicable composed
training-state contract. The next candidates are bounded load windows,
balanced local/reduction trees, proven shuffle-reader tracking and resource-
aware geometry selection. A synchronization deletion requires an explicit
full-mask/visibility proof and full hardware checks.

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

The next admission tests are current-source held-out confirmation of the selected
standard-width choices, combined load/tree scheduling informed by the retained
counters, and a composed training workload with matching saved-state/gradient
contracts.
No production implementation or whole-transformer speedup follows from this
standalone result.
