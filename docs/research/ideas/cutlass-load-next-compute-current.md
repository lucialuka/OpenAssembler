# CUTLASS-style load-next / compute-current fragment schedule

Status: implemented

This experiment adapts a public CUTLASS SM80 mainloop idea to the current OA
GEMM source. The validated schedule is now available as an explicit production
`GemmPlan(fragment_schedule="load_next")` / `openassembler gemm
--fragment-schedule load_next` option for two exact wide shapes. It does not
change the default source-order tensor schedule or copy cuBLAS device code.

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

The result is a real shape-specific OA improvement on five of the six shapes.
The square regression prevents a global dispatch rule. The higher register
count is therefore behind an exact-shape opt-in contract rather than automatic
selection.

A November 9 repeat used two shuffled rounds of eight graph replays on the same
six shapes. All full-output, canary, poisoned-graph, bitwise-ptxas and resource
checks passed. Averaging the two round medians, candidate/OA-baseline ratios
were 0.993, 0.975, 0.911, 0.982, 0.920 and 1.075 in table order. The 128-cubed
case ranged from a 1.3% gain to a tie; the 4096-cubed case remained 5.3–9.7%
slower. A current-source held-out replay added three large FP16 cases and three
BF16 cases: all gates passed, the three large FP16 cases regressed by about
5.1–5.4%, and the two exact wide 1024×4096×1024 / 4096×1024×1024 cases retain
the measured gains in both dtypes. The production option is restricted to
FP16/BF16 at both 1024×4096×1024 and 4096×1024×1024 and remains opt-in. The
large held-out FP16 shapes regress. See the [held-out raw record](../../../exp/tensor_scheduler/load-next-compute-current_20261109T045855.363914Z.json).

After the option was wired into `GemmPlan` and the CLI, a source-hashed
production-path confirmation reran 1024×4096×1024. The FP16 ratios were 0.9045
and 0.9084; BF16 ratios were 0.9260 and 0.9176. The fresh
4096×1024×1024 confirmation on an A100-SXM4-80GB measured FP16 ratios 0.9004
and 0.9002, and BF16 ratios 0.8954 and 0.9082. All checks pass in both raw
records. These are paired ratios within each device, not pooled absolute times:
[40GB confirmation](../../../exp/tensor_scheduler/load-next-compute-current_20261109T072411.844342Z.json),
[80GB confirmation](../../../exp/tensor_scheduler/load-next-compute-current_20261109T060813.116274Z.json).

## Reproduce

```bash
pytest -q exp/tensor_scheduler/test_load_next_compute_current.py
modal run exp/tensor_scheduler/load_next_compute_current_modal.py \
  --rounds 2 --samples 8
openassembler gemm --m 1024 --n 4096 --k 1024 \
  --fragment-schedule load_next --out-dir output_load_next
```

The runner compiles the baseline and candidate through the OA CLI, builds
matching ptxas controls, validates normal/integer/zero inputs, poisons graph
outputs after capture, and times shuffled CUDA-Graph replays.

## Scope and next work

The completed November 11 [42-contract production replay](../../../exp/dispatch_tuning/RESULTS.md)
includes this option wherever `GemmPlan` admits it. No frozen winning entry
selects load-next: other complete plans are faster or meet the selection gate
more reliably. The option therefore remains explicit and available to bounded
on-device tuning; this new result does not invalidate its older gains against
the then-reference source-order plan. Generalized shape-dependent dispatch is
implemented separately, with 34 other nonbaseline selections and eight
reference retentions. Held-out large load-next shapes remain rejected by this
option's admission rule.
Alignment, workspace, tails, transpose layouts and fused epilogues remain
separate contracts.

Public design reference: [CUTLASS SM80 multistage mainloop](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_multistage.h).

