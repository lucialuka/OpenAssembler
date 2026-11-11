# CUTLASS-style grouped-N CTA traversal

Status: implemented

This experiment tests a public CUTLASS CTA traversal policy against OA's
m-fast and Morton-8 traversals. The validated mapping is now an explicit
`--cta-order grouped_n8` production option with a checked launch-grid contract.
It changes logical tile order and launch-grid mapping only; the GEMM arithmetic,
shared layout and tensor instructions stay the same.

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

The current-source held-out run
[`grouped-n_20261109T051142.571274Z.json`](../../../exp/gemm_dispatch/grouped-n_20261109T051142.571274Z.json)
passed all output, canary, poisoned-graph, disassembly and resource gates on
four additional shapes. The 32768×4096×4096 FP16 case improved by about 29%,
16384×2048×2048 by about 10%, and 16384×4096×4096 BF16 by about 7%; the
8192×4096×4096 control was neutral. The repeated two-round 16384×4096×4096
source-inclusive production run also passed and measured 6.9–7.3% lower time.
These data justify an explicit option, not automatic dispatch across all grids.
After integration into the production `GemmPlan` and launch contract, another
two-round current-source confirmation measured FP16 grouped-N/m-fast ratios
0.9269/0.9379 and BF16 ratios 0.9249/0.9376. Full outputs, canaries,
poisoned-graph replay, matched-layout ptxas checks, disassembly and resource
gates pass; source and toolkit identities are retained in the
[production-path record](../../../exp/gemm_dispatch/grouped-n_20261109T061426.814828Z.json).

## Reproduce

```bash
pytest -q exp/gemm_dispatch/test_cta_swizzle.py
modal run exp/gemm_dispatch/grouped_n_modal.py --rounds 2 --samples 8
openassembler gemm --m 16384 --n 4096 --k 4096 \
  --cta-order grouped_n8 --out-dir output_grouped_n
```

The historical `measured_dispatch_manifest.json` records its original FP16,
16-byte-aligned, zero-workspace research entries. Later FP16/BF16 production
confirmations are separate records, not additions inferred from that old
manifest. Unmeasured alignment, workspace and neighboring shapes remain
unclassified.

## Scope and next work

The completed November 11 [42-contract production replay](../../../exp/dispatch_tuning/RESULTS.md)
includes grouped-N where its checked launch/plan is representable. It selects
no grouped-N row: the two large tall contracts freeze Morton8 and other shapes
select other plans or retain the reference. Grouped-N remains a valid explicit
and tuning candidate; that finite outcome does not make its prior measured
gains false or predict every new grid. Automatic exact-contract dispatch is
implemented separately and must not be confused with universal grouped-N
selection. The public design reference is [CUTLASS
threadblock swizzle](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h).

