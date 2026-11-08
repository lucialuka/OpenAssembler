# CUTLASS-style load-next / compute-current fragment schedule

Status: validated-not-integrated

This experiment adapts a public CUTLASS SM80 mainloop idea to the current OA
GEMM source. It does not copy cuBLAS device code and it does not change the
production tensor backend.

## Idea in plain language

The current OA mainloop loads the current A and B fragments, then performs the
current HMMA group. The candidate uses two register windows. It computes the
first N-fragment half of the current K slice, loads the next K slice into the
other window, then finishes the current slice. This gives memory work a chance
to overlap tensor work while every accumulator keeps its original K order.

The candidate preserves the existing `cp.async`, commit/wait-group, barrier,
shared-memory addresses, HMMA count and LDSM count. It adds register pressure:
the measured candidate used 24 more registers per thread on the six-shape
FP16 run.

## A100 findings

The source-inclusive run is
[`load-next-compute-current_20261108T061236.040495Z.json`](../../../exp/tensor_scheduler/load-next-compute-current_20261108T061236.040495Z.json).
All full-output, canary, poisoned-graph, bitwise ptxas, SASS and resource gates
passed on six FP16 shapes.

| Shape | OA baseline ms | Candidate ms | Candidate / baseline |
|---|---:|---:|---:|
| 128×128×128 | 0.009664 / 0.009728 | 0.009600 | 0.993–0.987 |
| 512×512×2048 | 0.063360 | 0.061760 / 0.061824 | 0.975–0.976 |
| 1024×4096×1024 | 0.087744 / 0.087680 | 0.080512 / 0.080384 | 0.918–0.917 |
| 2048×512×512 | 0.020992 | 0.020608 | 0.982 |
| 4096×1024×1024 | 0.087488 / 0.088000 | 0.079872 / 0.079936 | 0.913–0.910 |
| 4096×4096×4096 | 0.755520 / 0.641536 | 0.829760 / 0.705216 | 1.098–1.099 |

The result is a real shape-specific OA improvement on five non-square cases.
The square regression prevents a global promotion. The higher register count
also requires occupancy and counter checks before a production rule is safe.

## Reproduce

```bash
pytest -q exp/tensor_scheduler/test_load_next_compute_current.py
modal run exp/tensor_scheduler/load_next_compute_current_modal.py \
  --rounds 2 --samples 8
```

The runner compiles the baseline and candidate through the OA CLI, builds
matching ptxas controls, validates normal/integer/zero inputs, poisons graph
outputs after capture, and times shuffled CUDA-Graph replays.

## Scope and next work

This is a validated research method, not an automatic compiler optimization.
The next step is to profile the candidate against OA baseline and ptxas on
held-out FP16/BF16 shapes, then add a register/occupancy threshold to the
shape-aware manifest. Alignment, workspace, tails, transpose layouts and
fused epilogues remain separate contracts.

Public design reference: [CUTLASS SM80 multistage mainloop](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_multistage.h).

