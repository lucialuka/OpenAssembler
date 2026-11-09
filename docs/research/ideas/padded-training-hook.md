# Padded forward/backward integration prototype

Status: tested-not-selected

The original hook is preserved as a historical integration result. The
0.98 ms versus 0.68 ms measurement is a regression, and its runner is retired;
the current native GEMM CLI does not automatically pad or substitute model
operations.

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Wrap the experimental full-tile GEMM in an autograd-style training hook, padding dimensions where necessary, so forward and backward can exercise the same kernel family.

## Findings and implementation decision

The phase-two FP16/BF16 hook passed forward/backward and three-step state checks under its explicit padding/rounding policy. It was slower end to end, about 0.98 ms versus 0.68 ms for the explicit reference. It establishes a functioning bounded integration prototype, not acceleration, automatic native-kernel substitution or an autocast-equivalent production path.

The later whole-stack/native hybrid MLP work is separate and has stronger stage/state accounting. It must not be combined with this small earlier hook as if they were the same experiment.

## Method and validity checks

Include padding, packing, conversion, launches and all backward derivatives in the measured call. Match input/output precision and optimizer state across multiple steps. A forward-output check alone cannot validate a training integration.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase2/integration-results.json
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_fastmm.py
```
The original hook driver was retired; the tests cover current prerequisites rather than reproduce its historical 0.98/0.68 ms measurement. Use the later native-hybrid-MLP dossier for the retained whole-step driver.

## Code and retained evidence

- [Hook numerical/state/timing result](../../evidence/a100-mma/phase2/integration-results.json)
- [Phase-two context](../../evidence/a100-mma/phase2/summary.json)
- [Retained newer hybrid MLP](native-hybrid-mlp.md)

## What remains open

General transposed/tail dX/dW, saved statistics, graph integration and actual transformer steps remain separate contracts. Padding cannot be omitted from total time merely because the kernel is faster.
