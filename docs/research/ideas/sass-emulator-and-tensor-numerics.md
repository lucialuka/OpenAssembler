# SASS execution emulator and fitted tensor arithmetic

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Run the generated assembly in a software model before allocating a GPU. Model per-lane registers, memory, predicates and warp operations, then fit the unusual Tensor Core arithmetic against actual A100 outputs.

## Findings and implementation decision

The [exact-division investigation](../../compiler/gaps-verification.md) resolves
the retained one-ULP mismatch by measuring A100 FCHK slow-path selection and
repairing the emulator's small-dividend guard. All 3,409 exact quotient controls
pass across the four OA optimization levels; general hardware FCHK predicate
reconstruction remains incomplete. Emulator sampling now uses stable seeds and
retains an always-running fixed-counterexample regression.

The emulator supports the recorded scalar and tensor families, including HMMA/IMMA/DMMA, LDSM and asynchronous-copy probes. Recorded finite tensor cases match 512/512 lanes per tested family, including FP16, BF16, TF32, integer and FP64 tensor operations. This does not mean the production compiler generates all those modes; its native GEMM input contract remains FP16/BF16.

Earlier faults exposed uniform operations executed once per thread instead of once per warp, truncated wide-register reads, shared-memory races, uninitialized transpose inputs and special floating-point cases. Those faulty probes are excluded explicitly in replay tests.

## Method and validity checks

Use saved real GPU input/output bytes, replay the matching SASS in software, and compare every recorded lane. Poison otherwise-uninitialized registers to expose missing definitions. The model reproduces arithmetic, not instruction timing: an emulator pass cannot prove that scoreboard/stall settings are safe on hardware.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
PYTHONPATH=. .venv/bin/python tools/build_corpus.py sm_80
.venv/bin/python -m pytest -q tests/test_emu.py tests/test_gpu_truth.py tests/test_backend_tensor.py
```
The corpus is generated build output. Hardware replay tests report a skip when its SASS inputs have not been built.

## Code and retained evidence

- [Machine model](../../../openassembler/emu/machine.py)
- [Tensor arithmetic model](../../../openassembler/emu/tensor.py)
- [Recorded tensor hardware outputs](../../../corpus/sm_80/gpu_a100_40gb_tensor.raw.json.gz)
- [Ground-truth replay tests](../../../tests/test_gpu_truth.py)
- [Scalar model tests](../../../tests/test_emu.py)

## What remains open

NaN payloads, infinities, subnormals, cancellation extremes and all rounding modes need broader adversarial hardware coverage. Functional emulation must remain separate from the clocked tensor scheduling model.
