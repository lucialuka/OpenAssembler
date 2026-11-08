# Streamed load/FMA reduction candidate

This experiment adds a source-only candidate for the long-scoreboard gap in
4096-row backward reductions. It is under `exp/weaknesses/` and is not imported
by the production reduction backend.

## Candidate

`reduction_overlap_codegen.py` generates RMSNorm and softmax backward kernels
for widths 512, 513 and 1024 with four or eight warps per CTA. It keeps the
existing tail handling, five-step warp shuffle tree, full-lane mask and output
stores. It changes the order of work:

1. load one chunk of each input;
2. immediately fold the chunk into a local accumulator with explicit FMA;
3. load the next chunk;
4. reduce the local accumulator with the unchanged shuffle tree.

The purpose is to put arithmetic between global-load completions and reduce the
number of live product registers. It does not remove synchronization or weaken
the numerical gate.

## Local checks

```bash
pytest -q exp/weaknesses/test_reduction_overlap_codegen.py
```

The tests verify the generated FMA order, removal of the old per-chunk product
adds, tail guards, shuffle offsets and source-only scope. They do not prove GPU
correctness or speed.

The Modal build gate also accepted representative width-513 RMSNorm and
softmax sources with `ptxas -arch=sm_80 -O3` and `nvdisasm`. Its retained
summary is [`reduction_overlap_build_modal_result.json`](reduction_overlap_build_modal_result.json).
This gate validates PTX syntax and NVIDIA disassembly only; OpenAssembler
encoding and GPU execution remain separate checks.

## Required A100 validation

Compile each generated source with both OpenAssembler and CUDA 12.8 `ptxas`,
then run full outputs, canaries, poisoned graph replay and paired timing at
4096x513, 4096x512 and 4096x1024. Collect register counts and Nsight Compute
metrics for total instructions, global-load sectors, L2/DRAM traffic, short and
long scoreboard stalls, wait/barrier stalls and eligible warps.

The existing FMA-only study already showed 1.24--1.37x improvement over
unfused OA for selected 4096-row cases, but ptxas remained faster. This
candidate is intended to test whether load/FMA overlap reduces the remaining
long-scoreboard component. A lower long-scoreboard percentage without a lower
kernel time is not a promotion signal.

## Source-inclusive A100 validation

The source-inclusive Modal run is recorded in
`reduction_overlap_run_20261108T051551.705065Z.json`. It compiled both the
ordinary and overlap sources through the OA encoder, retained the matching
ptxas cubins, and ran 4096-row RMSNorm and softmax backward cases at widths
512, 513 and 1024 with four and eight warps.

Every case passed normal, zero and structured input checks, output tolerances,
canaries, input immutability, graph-poison replay and ptxas execution. Normal
and structured FP32 outputs are not bitwise identical because the overlap
candidate uses explicit FMA contraction; the declared tolerance contract
passed. Zero cases were bitwise equal.

The overlap candidate reduced OA instructions and registers in every case. The
representative RMSNorm reductions were:

| Width | OA baseline | OA overlap | Registers | OA overlap time | ptxas time |
|---|---:|---:|---:|---:|---:|
| 512 | 232 instructions / 58 regs | 200 / 43 | -15 | 0.01408 ms | 0.01229 ms |
| 513 | 240 / 61 | 208 / 45 | -16 | 0.01434 ms | 0.01178 ms |
| 1024 | 392 / 106 | 328 / 75 | -31 | 0.05402 ms | 0.04122 ms |

Softmax overlap reduced the corresponding OA streams from 152/53 to 136/41
registers at width 512, 160/56 to 144/43 at width 513, and 264/101 to
232/73 at width 1024. The overlap candidate was faster than the unfused OA
candidate in every tested case, but ptxas remained faster in every case. No
Nsight counter pass has yet been collected for this candidate, so the desired
long-scoreboard reduction is still unmeasured.

The candidate remains experiment-only. Its next gate is a matched NCU replay
for global-load sectors, DRAM/L2 traffic, long/short scoreboard, wait/barrier
stalls and eligible warps. Promotion requires a counter improvement together
with the existing full-output and held-out-shape gates.

