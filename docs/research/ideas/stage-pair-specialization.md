# Two-tile stage-pair specialization and peeled drain

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Handle alternating shared-memory stages as a fixed pair of K tiles. Encode known stage offsets directly instead of repeatedly selecting a dynamic stage, and stop prefetching before the end of the matrices.

## Findings and implementation decision

`--schedule stagepair` is the selected production plan. It specializes alternating stage-zero/stage-one work into a two-tile loop and peels the final pair so no speculative out-of-range copy occurs. `--schedule combined` keeps an earlier dynamic-stage reference for comparison.

This specialization is one component of the historical 16.13–26.69% mainloop
bundle improvement. In the November 11 [42-contract replay](../../../exp/dispatch_tuning/RESULTS.md),
all installed plans retain stagepair; no combined-loop candidate wins the
pre-frozen selection and independent replay. Combined remains an admissible
comparison where the full plan allows it. HMMA8/drain15 and explicit copy
completion are retained; dispatch does not relax proven dependency edges.

## Method and validity checks

Track every K tile through issue, completion and consumption; test K=128 and odd/even numbers of larger stage pairs. Validate the final tile and guard regions. Compare the whole kernel under identical input/output precision, not isolated instruction counts.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 384 --schedule stagepair --out-dir output/stagepair
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 384 --schedule combined --out-dir output/combined
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_backend_tensor.py
```
For GPU measurements give both plans matching buffers and compare full outputs before timing. The generic `run` command is an exact-integer smoke, not the original phase-three search.

## Code and retained evidence

- [Stage-pair generator](../../../openassembler/kernels/gemm.py)
- [Tile-accounting tests](../../../tests/test_gemm_codegen.py)
- [Historical selected schedule](../../evidence/a100-mma/phase3/evidence-summary.json)

## What remains open

Larger stage counts require a separately correct copy/consume protocol and resource model. They cannot be obtained by changing a single loop constant and assuming the same safety edges.
