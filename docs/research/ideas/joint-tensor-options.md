# Joint tuning of GEMM pointer, tile-order and fusion options

Status: tested-not-selected

The combinations remain experiment results rather than current compiler
defaults. Their correctness does not outweigh mixed or small timing changes,
and the candidates still lose to their matched ptxas controls.

This classification is conditional: combining direct pointers with a fused
FP32 epilogue gives small measured gains on selected shapes. The combined
tile-order experiment does not establish another broadly useful policy. All
new combinations remain in `exp/`; the production compiler's option checks
and defaults are unchanged.

## Idea in plain language

Two optimizations that help separately need not help together. Direct pointers
remove redundant address-register copies. Morton traversal changes the order
of output tiles so nearby work may reuse operands in cache. A fused epilogue
adds bias, a residual connection and ReLU before storing the final result.

The new experiment combines direct pointers with either Morton traversal or
the fused epilogue. The ordinary CLI currently permits neither combination.
An experiment wrapper selects the existing tensor backend's direct-pointer
policy on the corresponding legacy source, without removing a synchronization
dependency or changing MMA arithmetic. This tests whether the earlier option
restrictions conceal a useful combination; it does not weaken those production
restrictions before admission.

## Methods and complete checks

The November 7, 2026 run uses an **A100-SXM4-40GB**, 108 SMs, driver 13030,
PyTorch 2.7.1 with CUDA 12.8. Eighteen cases cover nine matrix shapes in FP16
and BF16: a small control; forward, input-gradient and weight-gradient products
for two MLP sizes; a 4096-cubed square; and a 16384×4096×4096 tall product.
These are isolated full-tile products, with already packed contiguous operands,
not complete transformer steps.

Each product is `C=A@B.T`, using FP16/BF16 inputs and FP32 accumulation/output.
The fused contract computes FP32 `(C + bias + residual).relu()`, with those
two additions in order. The tensor mainloop retains the existing two shared
stages, 128×128×64 tile, four warps, HMMA issue gap eight, epoch drain fifteen,
reuse hints, copy visibility and address/read/write dependency handling.

Before timing, every method runs zero, small-integer and normal input families.
Every output element is checked against a complete FP64 reference outside
timing. Each native method must additionally agree **bitwise** with ptxas
compiling its exact PTX source. NaN-poisoned outputs, surrounding canaries and
all unchanged input buffers are checked. Encoded native instruction/control
words and critical operands must also round-trip through `nvdisasm`.

All **402/402 full method/input checks pass**, covering **4,505,468,928 output
elements**. Another **134/134 poisoned CUDA-graph replay checks pass**, covering
1,501,822,976 output elements. The largest relative L2 error against FP64 is
6.61e-6. These are output checks across repeated methods and patterns, not
billions of statistically independent experiments.

Timing uses three within-process rounds, eleven shuffled paired samples per
round and five launches per CUDA graph. Round zero chooses a fixed candidate;
rounds one and two evaluate that choice. The separate ptxas traversal choice
also uses round zero. This is held-out measurement within one worker, not an
independent-process replication. Kernel capture, allocation and FP64 checking
are outside these steady-state timings.

## Findings and decisions

Ratios below are baseline time divided by candidate time in the two evaluation
rounds; above one favors the candidate. These compare fixed named options,
rather than selecting a new winner from each evaluation round.

| Combination and case | Evaluation ratios | Decision |
|---|---:|---|
| Direct+fused vs legacy+fused, 512×2048×512 FP16 | 1.021× / 1.021× | Small positive result worth a targeted follow-up |
| Direct+fused vs legacy+fused, 1024×4096×1024 FP16 | 1.027× / 1.023× | Conditional positive result; no production promotion |
| Direct+fused vs legacy+fused, 1024×4096×1024 BF16 | 1.020× / 1.018× | Small gain, below 2% in one evaluation round |
| Direct+fused vs legacy+fused, 512×2048×512 BF16 | 0.993× / 1.014× | Mixed; no repeatable gain established |
| Direct+fused vs legacy+fused, 4096³ FP16/BF16 | About 1.000–1.004× | Too small/mixed to justify selection |
| Direct+Morton vs direct, MLP and square controls | About 0.995–1.005× | Mostly tied or slower; no additive locality gain established |
| Direct+Morton vs Morton, tall FP16 | 0.996× / 0.995× | Slower than retaining legacy pointers with Morton |
| Direct+Morton vs Morton, tall BF16 | 0.999× / 0.993× | No repeatable additional benefit |

The large improvement of direct+Morton over a legacy Morton MLP is primarily
the already established direct-pointer effect. It must not be advertised as a
new Morton gain: the relevant direct-only control is tied or slightly faster.
Conversely, on the tall case, Morton helps both pointer policies but the
legacy-pointer Morton control remains better than the new combination.

The new direct+fused FP16 cases still lose to their exact-source fused ptxas
controls. For example, the wide forward case takes about **135.8–136.4 µs**
with the new native combination versus **85.2 µs** with fused ptxas. The smaller
forward case takes 29.5 µs versus 20.9–21.1 µs. A gain over an older OA option
therefore does not establish superiority over CUDA compilation.

The cuBLAS control is `GemmEx` algorithm 99 with FP32 compute/output and
reduced-precision reduction disallowed. Its matching unfused post-epilogue
uses preallocated FP32 add/add/ReLU operations. This is a concrete library
dispatch, not an exhaustive cuBLASLt/CUTLASS search. The native fused square
can beat this separate library-plus-pointwise pipeline while still losing to
the same-source fused ptxas kernel; both comparisons must remain visible.

No counter diagnosis is inferred from these timing results alone. The separate
[early pointer-advance experiment](early-gemm-pointer-advance.md) changes
address placement while keeping the operation multiset fixed.

## Reproduction and commands

Run from the repository root with the compiler test environment, CUDA 12.8
tools and an authenticated Modal client. All builds stay under `exp/nn_round2`.
The central campaign launcher is the tested GPU transport; the experimental
suite's separate `run` convenience path is not the reproduction route.

```sh
.venv/bin/python -m pytest -q exp/nn_round2/test_tensor_suite.py
.venv/bin/python exp/nn_round2/tensor_suite.py matrix --profile full
.venv/bin/python exp/nn_round2/tensor_suite.py build --profile full
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase tensor \
  --payload exp/nn_round2/tensor_build_20261107T142912.570956Z.json \
  --rounds 3 --samples 11
.venv/bin/python exp/nn_round2/tensor_analyze.py \
  exp/nn_round2/runs/tensor-20261107T143201.758879Z.json
```

The shown timestamped paths replay/analyze the retained snapshot. A fresh
build prints its own manifest name; use that name for a new campaign. The
payload loader checks compiler-source and binary identities and refuses drift.
Timing may differ with clocks, allocation/cache state and worker identity;
compare paired samples within that actual worker.

## Code and evidence

- [Candidate compiler and full GPU validator](../../../exp/nn_round2/tensor_suite.py)
- [Candidate/contract CPU gates](../../../exp/nn_round2/test_tensor_suite.py)
- [Exact build metadata](../../../exp/nn_round2/tensor_build_20261107T142912.570956Z.json)
- [Full GPU record](../../../exp/nn_round2/runs/tensor-20261107T143201.758879Z.json)
- [Held-out summary](../../../exp/nn_round2/tensor_summary_20261107T143201.json)
- [Summary generator](../../../exp/nn_round2/tensor_analyze.py)
- [Existing direct-pointer option](direct-pointer-homes.md),
  [Morton traversal](morton-cta-order.md) and [fused epilogue](fused-gemm-epilogues.md)

## What remains open

Repeat the small direct+fusion gains in independent workers, inspect counters
and test them inside a complete network including packing and backward work.
A shape policy should choose from separately validated options on one tuning
partition and then pass a held-out whole-workload test. This experiment does
not add tails, arbitrary transposes, TF32, another architecture or an automatic
compiler policy.
