# CTA traversal and dispatch experiments

This directory contains experiment-only models for the A100 GEMM launch
policy.  It is deliberately separate from `openassembler/kernels/gemm.py` so
that a traversal candidate cannot become a production dispatch rule before it
has been compiled and timed on held-out A100 shapes.

`cta_swizzle.py` implements three logical traversals:

* `m_fast`: the current OA launch, with `blockIdx.x` selecting the M tile;
* `morton8`: the existing OA 8×8 Morton traversal;
* `grouped_n8` (and narrower widths): the public CUTLASS grouped-N mapping,
  where a physical X group maps to one M tile and a consecutive group of N
  tiles.

The grouped-N rewrite changes both the CTA coordinate arithmetic and the
required launch grid.  The experiment caller must launch
`grid.x = m_tiles * width` and `grid.y = n_tiles / width`; reusing the current
`m_tiles × n_tiles` grid is invalid.  The rewrite does not change the GEMM
math, shared layout, tensor instruction count or output epilogue.

Run the CPU model and generate the manifest with:

```bash
python3.13 -m pytest -q exp/gemm_dispatch/test_cta_swizzle.py
python3.13 exp/gemm_dispatch/heuristic_manifest.py
```

The generated `shape-alignment-workspace-manifest.json` is a candidate matrix,
not a benchmark result.  Every grouped-N row remains `untested` until a
source-inclusive A100 run verifies full output, canaries, poisoned-input
execution, valid disassembly, register/shared-memory resources and paired
held-out timing against the current OA traversal.

The grouped mapping follows the public CUTLASS implementation in
[`threadblock_swizzle.h`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h).

## A100 source-inclusive validation

Run the grouped-N, Morton and m-fast comparison with:

```bash
modal run exp/gemm_dispatch/grouped_n_modal.py --rounds 2 --samples 8
```

The latest record is `grouped-n_20261108T062456.797718Z.json`. It covers
1024×4096×1024, 4096×4096×4096 and 16384×4096×4096 in FP16. Every numerical,
canary, graph-replay, disassembly and resource check passed. The tall shape
favored grouped-N and Morton by about 6–7% over OA m-fast; the wide and square
cases did not produce a stable grouped-N win.

`measured_dispatch_manifest.json` records exact shape/alignment/workspace
entries from this run. A `validated-experiment-only` status means both timing
repeats beat the current OA policy for that exact case. It does not extend to
unmeasured alignment, workspace, dtype or neighboring shapes.
