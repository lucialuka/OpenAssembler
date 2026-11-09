# Two-slot asynchronous barrier overlap

Status: tested-not-selected

The old single-CTA protocol is retained as historical capability evidence. Its
runner is retired and there is no current dense-GEMM lowering or A100 result
showing a benefit inside the OpenAssembler compiler.

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Use separate arrive and wait phases so a producer warp can prepare a future tile while a consumer works on the current one. Two buffer slots can avoid making both sides wait on the same storage each phase.

## Findings and implementation decision

A correct two-slot SM80 protocol passed phase-varying complete outputs, count/parity accounting and canaries. The one-CTA experiment measured 0.332 versus 0.455 ms steady (1.370×) and 0.989 versus 1.183 ms under artificial stress (1.197×). It used one producer warp, one consumer warp and a 16×16×16 product per phase.

The single-slot protocol was correct but slower: 0.553–0.557 versus 0.355 ms in its steady comparison. No general mbarrier lowering or dense-GEMM acceleration was promoted. The positive result is a tiny controlled overlap experiment, not a transformer-training result.

## Method and validity checks

Keep barrier objects separate from operands, count arrivals exactly, track phase parity through wraparound, and validate full phase-coded outputs. Two initial implementation mistakes were repaired: overlapping barrier/operand storage and using the A lane map for B. Those failures were not hardware limitations.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase5/barrier-two-slot.json
python3 -m json.tool docs/evidence/a100-mma/phase5/barrier-accounting.json
python3 -m json.tool docs/evidence/a100-mma/phase5/barrier-layout-check.json
```
The complete original assembly/protocol runner was retired. These commands inspect preserved results and accounting, not rerun the protocol. Current `gemm` uses its established copy/CTA-barrier model; no `--mbarrier` switch exists.

## Code and retained evidence

- [Two-slot full-output record](../../evidence/a100-mma/phase5/barrier-two-slot.json)
- [Single-slot negative result](../../evidence/a100-mma/phase5/barrier-single-slot.json)
- [Arrival/parity accounting](../../evidence/a100-mma/phase5/barrier-accounting.json)
- [Layout repair checks](../../evidence/a100-mma/phase5/barrier-layout-check.json)
- [Static protocol checks](../../evidence/a100-mma/phase5/barrier-pipeline-static-check.json)

## What remains open

Rebuild an auditable protocol driver and show it wins inside a competitive dense GEMM before adding a compiler lowering. A raw arrival instruction without initialization, parity and lifetime handling is insufficient.
