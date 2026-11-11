# Deeper copy pipelines and larger tiles

Status: tested-not-selected

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Use more shared-memory stages or a larger matrix tile to hide memory delay and reuse more operands. Extra buffering can also consume the shared memory/registers needed to keep enough work active.

## Findings and implementation decision

For the tested 64-wide K tile, two stages beat three/four: approximately 1.003 ms versus 1.178/1.180 ms in the archived run. Larger 128×256 and 256×128 CTA variants passed but were slower. Three/four/five-stage follow-ups did not establish a repeatable win over the selected schedule. A smaller 64×32 warp tile increased active warps/tensor activity but still lost to the selected 64×64 tile on the tested large GEMMs.

These are rejected implementations on measured workloads, not proof that
deeper pipelines or alternate tiles can never win. A distinct later
[64×64 CTA family](small-grid-tensor-cta-tiles.md), with four 32×32 warp tiles
and two stages, wins 22 exact contracts in the November 11 production replay.
It does not share the older 64×32 warp-tile design or establish a deeper-stage
win.

## Method and validity checks

Account for every asynchronous tile, initial fill and final drain. Check complete outputs and guards, dynamic shared memory, register count, local memory and launch resources. Compare same-source ptxas and the current OA control with matched output precision and shuffled graph timing.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --out-dir output/two-stage-control
```
Production exposes the selected two-stage plan, not a `--stages` option. Historical multistage/alternate-tile generators were retired; their raw records are retained. Reimplement a candidate under `exp/` before rerunning it.

## Code and retained evidence

- [Phase-two stage/tile results](../../evidence/a100-mma/phase2/summary.json)
- [Later search outcomes](../../evidence/a100-mma/phase3/search-summary.json)
- [Occupancy/counter follow-up](../../evidence/a100-mma/phase3/profile-followup.json)
- [Selected pipeline tests](../../../tests/test_gemm_codegen.py)

## What remains open

Small grids, different K depths and a joint layout/register search may select different tiles. More occupancy or more stages is not itself a performance objective.
