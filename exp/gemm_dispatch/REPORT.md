# A100 CTA and fragment dispatch experiments

This report records source-inclusive A100 tests for traversal and fragment
schedule candidates. The experiments compile each source through the
OpenAssembler CLI and through matching `ptxas -O3`, then compare full outputs,
canaries, poisoned graph replay, SASS disassembly, registers and paired CUDA
Graph timings. No result in this directory changes the production dispatcher.

## CUTLASS-style load-next / compute-current

`exp/tensor_scheduler/load_next_compute_current.py` keeps the existing
`cp.async`, `wait_group`, barrier and shared-memory address protocol. It uses
two alternating fragment register windows and inserts the next K-fragment
loads between the first and second N-fragment HMMA groups. The per-accumulator
K order is checked before source generation.

The latest six-shape FP16 A100 run is
`../tensor_scheduler/load-next-compute-current_20261108T061236.040495Z.json`.
All 72 output/canary/graph/bitwise checks passed. The candidate kept the OA
HMMA/LDSM counts unchanged and increased registers by 24 per thread.

| Shape | OA baseline ms | Candidate ms | Candidate / baseline | OA regs | Candidate regs |
|---|---:|---:|---:|---:|---:|
| 128×128×128 | 0.009664 / 0.009728 | 0.009600 | 0.993–0.987 | 184 | 208 |
| 512×512×2048 | 0.063360 | 0.061760 / 0.061824 | 0.975–0.976 | 216 | 240 |
| 1024×4096×1024 | 0.087744 / 0.087680 | 0.080512 / 0.080384 | 0.917–0.916 | 216 | 240 |
| 2048×512×512 | 0.020992 | 0.020608 | 0.982 | 216 | 240 |
| 4096×1024×1024 | 0.087488 / 0.088000 | 0.079872 / 0.079936 | 0.913–0.910 | 216 | 240 |
| 4096×4096×4096 | 0.755520 / 0.641536 | 0.829760 / 0.705216 | 1.098 / 1.099 | 216 | 240 |

The candidate wins the five non-square cases in this run and loses the square
case. It is therefore a shape-specific dispatch candidate only; it is not a
global replacement. The square regression and higher register pressure require
additional held-out and counter validation before any production promotion.

## Grouped-N, Morton and m-fast traversal

`cta_swizzle.py` models the public CUTLASS grouped-N mapping. The A100 runner
uses the required launch grid `grid.x = m_tiles × width` and
`grid.y = n_tiles / width`, then compares it with the current m-fast and
existing Morton-8 mappings. The complete FP16 run is
`grouped-n_20261108T062456.797718Z.json`.

All 36 layout checks passed. Grouped-N and Morton preserve the 512 HMMA work
and the resource count; Morton adds 24 integer prologue instructions and
grouped-N adds four.

| Shape | m-fast OA ms | Morton OA ms | Grouped-N OA ms | Best OA choice |
|---|---:|---:|---:|---|
| 1024×4096×1024 | 0.087744 / 0.088128 | 0.089536 / 0.088320 | 0.088640 / 0.088256 | m-fast or grouped-N within noise |
| 4096×4096×4096 | 0.645632 / 0.640128 | 0.644160 / 0.642496 | 0.647232 / 0.642112 | Morton on first round; near tie later |
| 16384×4096×4096 | 2.642240 / 2.648832 | 2.475072 / 2.498176 | 2.472576 / 2.467648 | grouped-N, narrowly ahead of Morton |

The tall case is the strongest result: grouped-N improves OA by roughly 6–7%
versus m-fast and is slightly ahead of Morton in the second round. The wide
case does not justify a grouped-N rule, and the square case is effectively a
tie. These results support a shape-aware traversal manifest rather than a
universal swizzle.

## Shape, alignment and workspace manifest

`shape-alignment-workspace-manifest.json` enumerates 108 candidates across
tall, wide and square FP16/BF16 shapes, 16/128-byte input alignment and
0/1 MiB/64 MiB workspace. `grouped_n8` rows were initially marked `untested`.
The measured records above can now mark only the exact FP16 shapes as
`validated-experiment-only`; alignment and workspace variants remain
unmeasured and must not inherit the timing claim.

Promotion requires full output, canaries, poisoned input/graph replay, valid
disassembly, resource reports and paired held-out timing. A candidate that
beats OA on one traversal or one shape is not a general cuBLAS comparison.

