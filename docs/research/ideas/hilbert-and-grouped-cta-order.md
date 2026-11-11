# Hilbert, grouped-M and alternative CTA orders

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Try other logical orders for output tiles—Hilbert curves, N-fast traversal or strips/groups—to improve operand locality without altering the matrix multiplication itself.

## Findings and implementation decision

The phase-five study tested N-fast, grouped-M, Morton8 and Hilbert8 while retaining the physical HMMA/LDSM/LDGSTS words and controls. The first and independent runs selected different curve winners. Picking the largest apparent improvement from one sweep was rejected. Hilbert and the other alternatives were not promoted; the later corrected Morton path has its own dossier and validation.

A software tile order is only a locality hint. It cannot force physical SM adjacency or create hardware thread-block clusters.

## Method and validity checks

Check the full tile permutation, unchanged arithmetic kernel and output/guard correctness. Repeat paired timing and inspect cache counters separately. Distinguish a reproducible locality benefit from noise and different GPU clocks.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase5/layout-results.json
python3 -m json.tool docs/evidence/a100-mma/phase5/layout-counters.json
.venv/bin/python -m openassembler gemm --m 16384 --n 4096 --k 4096 --dispatch manual --out-dir output/order-control
```
The explicit manual production command builds m_fast; without that flag, the
current exact selector chooses Morton8 for this shape. There is no `--cta-order hilbert8` option. The complete original alternative-order sweep was retired; inspection of its JSON is not a fresh timing replay.

## Code and retained evidence

- [Alternative-order runs](../../evidence/a100-mma/phase5/layout-results.json)
- [Counters and replay limits](../../evidence/a100-mma/phase5/layout-counters.json)
- [Current Morton option](morton-cta-order.md)

## What remains open

The current bounded selector already measures m-fast, Morton8 and grouped-N
where admitted; the November 11 replay selects two Morton8 contracts and no
grouped-N row. Extending that search to Hilbert or grouped-M still needs a
checked rectangular-grid permutation and independent confirmations. Preserve
negative/neutral results rather than choosing the best isolated sample.
