# Fused FP32 bias, residual and ReLU epilogues

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Apply bias, residual addition and optional ReLU while the matrix result is still available in registers. This can avoid an extra output write/read and a separate pointwise kernel.

## Findings and implementation decision

The explicit `bias_residual` and `bias_residual_relu` plans compute `(A@B.T + Bias) + Residual`, then optional finite-value ReLU, with FP32 inputs to the epilogue and FP32 output. Bias is [N], residual is [M,N]; both pointers are required. The measured stage-pair mainloop prefix remains unchanged and registers remain 216.

The November 5 square 4096³ full-output tests show 4.95–5.34% less total OA time versus OA GEMM plus OA pointwise work. Same-source ptxas fusion is faster. Small smoke regressed about 20%, and tested MLP fusion was slightly slower. The option is explicit; it is not universal training fusion.

The November 11 [42-contract dispatch replay](../../../exp/dispatch_tuning/RESULTS.md)
only promotes unfused `epilogue=none` GEMMs. It adds no automatic fused row or
full training interface. Explicit fused contracts remain available; an
unfused timing win cannot be reused as evidence for their complete cost.

## Method and validity checks

Match arithmetic order, rounding and saved/backward behavior. Use nonzero auxiliaries, full outputs, canaries and dReLU checks. Compare complete fused versus separate operations, not fused GEMM versus GEMM alone. cuBLASLt already supports applicable epilogues; a vendor baseline must use equivalent semantics.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --epilogue bias_residual_relu --out-dir output/fused
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_cli_gemm.py
modal run docs/evidence/a100-mma/current-validation/validate_modal.py
```
Automatic CLI smoke uses zero bias/residual and is not the nonzero accuracy/performance study. Supply the real five-pointer ABI and nonzero buffers in a separate launch specification or retained validator.

## Code and retained evidence

- [Fused generator](../../../openassembler/kernels/gemm.py)
- [Nonzero full-output current record](../../evidence/a100-mma/current-validation/validation_20261105T074142.327676Z.json)
- [Small-shape regression](../../evidence/a100-mma/phase5/fusion-smoke-results.json)
- [MLP fusion regression](../../evidence/a100-mma/phase5/fusion-mlp-results.json)
- [Independent square confirmation](../../evidence/a100-mma/phase5/fusion-square-confirmation.json)

## What remains open

Bias gradients, backward fusion, saved activations and larger graph partitions need whole-step measurements. Eliminating bytes is useful only if it does not damage the mainloop or increase launch/register costs more.
