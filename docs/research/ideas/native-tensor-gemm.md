# Native SM80 tensor GEMM compiler and mainloop

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Generate a complete matrix-multiplication kernel whose main work executes on A100 Tensor Cores, while OpenAssembler controls native instruction choice, registers, copies and dependencies.

## Findings and implementation decision

The production `gemm` path computes C=A@B.T with contiguous FP16/BF16 operands
and FP32 accumulation/output. The reference is a 128×128×64 CTA tile with four
64×64 warp tiles, 128 threads and 64 KiB dynamic shared memory; a checked
64×64×64 CTA family uses four 32×32 warp tiles and 32 KiB. Exact-contract
dispatch chooses a current promoted policy, while unknown/stale contracts keep
the representable reference family. Both combine asynchronous copies, paired fragment
loads, 7/7 shared swizzle, the protected HMMA8/drain15 schedule and vector
stores.

The historical schedule bundle reduced time 16.13–26.69% versus the earlier
direct schedule across six square/MLP/tall cases and reached 79.76–98.91% of
matched default cuBLAS. Those historical ratios are not current universal
performance estimates.

The completed November 11 [production-source dispatch replay](../../../exp/dispatch_tuning/RESULTS.md)
validates 42 exact contracts in independent selection and held-out sessions:
34 select faster OA plans, eight retain reference, 22 select tile64 and 20
select copy-address reuse. Improved baseline/selected ratios span
1.033–2.460× across six paired rounds. Every session's nominal selected OA
kernel is still slower than its same-source ptxas control; seven held-out
nominal selections differ from the pre-frozen installed policy, so their
supplemental library ratios are not the installed policy's timings. None of
these isolated GEMM results establishes cuBLASLt parity, arbitrary GEMM support
or a native transformer compiler.

## Method and validity checks

Verify every output against a declared FP32 reference, use zero/integer/normal inputs and guards, and preserve the complete dependency graph. Retain full encoded stream hashes; disassemble generated cubins; time matched methods on one worker with randomized order. Profiling replay is a separate experiment.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler gemm --m 4096 --n 4096 --k 4096 --dtype f16 --out-dir output/native-gemm
.venv/bin/python -m openassembler run output/native-gemm --backend modal --budget 300 --job-timeout 120
.venv/bin/python -m pytest -q tests/test_backend_tensor.py tests/test_gemm_codegen.py tests/test_cli_gemm.py
modal run docs/evidence/a100-mma/current-validation/validate_modal.py
```
`run` uses an exact-small-integer smoke contract, not the archived normal-input performance test. The retained validator supplies broader checks. Original phase-three candidate generators were retired; raw historical results remain available.

## Code and retained evidence

- [PTX kernel generator](../../../openassembler/kernels/gemm.py)
- [Native tensor selector/scheduler](../../../openassembler/backend/tensor.py)
- [Bundled encoder](../../../openassembler/backend/tensor_encoding.py)
- [Historical schedule evidence](../../evidence/a100-mma/phase3/evidence-summary.json)
- [Current-source validator](../../evidence/a100-mma/current-validation/validate_modal.py)

## What remains open

Same-source ptxas remains faster in the latest nominal-selected GEMM cohort.
Transposed storage, general strides/tails, arbitrary alpha/beta, TF32 generation
and a full native forward/backward/optimizer graph need new contracts. The
[dispatch dossier](shape-aware-native-gemm-dispatch.md) specifies current
conditional routing; it is not an all-shapes optimum guarantee.
