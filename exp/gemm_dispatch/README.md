# CTA traversal and dispatch experiments

This directory retains the research models and source-inclusive tests for A100
GEMM traversal. Production implementations live in
`openassembler/kernels/gemm.py`; this directory is not imported by the compiler.
The [grouped-N dossier](../../docs/research/ideas/grouped-n-cta-swizzle.md)
owns its current integration status, while the files here preserve the exact
historical experiment contracts.

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
not a benchmark result. Its `untested` rows must not inherit results from a
different shape, dtype, alignment or workspace. A dispatch entry requires a
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

The original record is `grouped-n_20261108T062456.797718Z.json`. It covers
1024×4096×1024, 4096×4096×4096 and 16384×4096×4096 in FP16. Every numerical,
canary, graph-replay, disassembly and resource check passed. The tall shape
favored grouped-N and Morton by about 6–7% over OA m-fast; the wide and square
cases did not produce a stable grouped-N win.

Later November 9 held-out and production-path confirmations are linked in the
[canonical dossier](../../docs/research/ideas/grouped-n-cta-swizzle.md).
`measured_dispatch_manifest.json` records exact shape/alignment/workspace
entries from this run. A `validated-experiment-only` status means both timing
repeats beat the current OA policy for that exact case. It does not extend to
unmeasured alignment, workspace, dtype or neighboring shapes.
