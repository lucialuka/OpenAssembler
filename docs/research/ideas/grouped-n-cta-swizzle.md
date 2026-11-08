# CUTLASS-style grouped-N CTA traversal

Status: validated-not-integrated

This experiment tests a public CUTLASS CTA traversal policy against OA's
m-fast and Morton-8 traversals. It changes logical tile order and launch-grid
mapping only; the GEMM arithmetic, shared layout and tensor instructions stay
the same.

## Idea in plain language

Grouped-N puts a consecutive group of N tiles behind one M tile. For width 8,
the physical launch grid is `grid.x = m_tiles × 8` and
`grid.y = n_tiles / 8`. The kernel maps
`logical_m = blockIdx.x >> 3` and
`logical_n = (blockIdx.y << 3) | (blockIdx.x & 7)`. This can improve reuse of
the A tile and the order in which CTAs touch B tiles, but it can also hurt
other grids.

## A100 findings

The source-inclusive run is
[`grouped-n_20261108T062456.797718Z.json`](../../../exp/gemm_dispatch/grouped-n_20261108T062456.797718Z.json).
All numerical, canary, poisoned-graph, disassembly and resource gates passed.

| Shape | m-fast OA ms | Morton OA ms | Grouped-N OA ms |
|---|---:|---:|---:|
| 1024×4096×1024 | 0.087744 / 0.088128 | 0.089536 / 0.088320 | 0.088640 / 0.088256 |
| 4096×4096×4096 | 0.645632 / 0.640128 | 0.644160 / 0.642496 | 0.647232 / 0.642112 |
| 16384×4096×4096 | 2.642240 / 2.648832 | 2.475072 / 2.498176 | 2.472576 / 2.467648 |

The tall case favors grouped-N and Morton by about 6–7% over m-fast. The wide
and square cases do not produce a stable grouped-N win. Grouped-N adds four
integer prologue instructions and keeps the measured 216-register resource
count; Morton adds 24 integer instructions for its coordinate decode.

## Reproduce

```bash
pytest -q exp/gemm_dispatch/test_cta_swizzle.py
modal run exp/gemm_dispatch/grouped_n_modal.py --rounds 2 --samples 8
```

`measured_dispatch_manifest.json` records the exact FP16, 16-byte-aligned,
zero-workspace entries. Unmeasured alignment, workspace, BF16 and neighboring
shapes remain unclassified.

## Scope and next work

The result supports a shape-aware traversal manifest, not a universal swizzle
rule. A production entry would need held-out repeats, counter evidence and an
explicit launch-grid contract. The public design reference is [CUTLASS
threadblock swizzle](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h).

