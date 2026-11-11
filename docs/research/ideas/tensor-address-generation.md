# Carry-linked pointer updates and folded shared addresses

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Spend fewer instructions calculating where the next matrix tile lives. Keep a loop-carried 64-bit pointer in an allocated pair, update it with a correct carry chain, and fold known shared-stage offsets into the selected copy/address form.

## Findings and implementation decision

The earlier address work reduced one selected kernel from 1,898 to 1,812 static non-NOP instructions; six same-worker cases improved about 0.9–5.4%. Folded shared-copy displacements and compact addresses were later part of the larger promoted schedule. Both halves of a 64-bit pointer and the carry predicate remain explicit dependencies.

This original in-place pointer-update feature is distinct from the newer `--pointer-mode direct`, which removes temporary transfer-source copies and has different shape-dependent results.

The current tile64 generator also supports checked copy-base reuse through
`--copy-address-init reuse`. The completed [42-contract independent replay](../../../exp/dispatch_tuning/RESULTS.md)
selects reuse in 20 exact contracts. On 128×512×512 FP16, frozen reuse/direct/early
time is about 3.8–4.8% below the prior exact tile64/direct incumbent across six
paired rounds. The 2.237–2.261× gain versus tile128 baseline combines geometry,
pointer and address changes; it must not be described as isolated CSE benefit.
The address certificate includes 64-bit carry boundaries, copy/shared/LDSM
identity and unchanged async dependencies; per-policy current PTX, cubin and
SASS are retained.

## Method and validity checks

Test both low-half wrap and carry into the high half, exact address ranges, stage identity, peeled final tiles and complete output coverage. A shorter address expression is not valid if it changes alignment, wraps a 32-bit element offset or overwrites an in-flight reader.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend_loops.py tests/test_backend_tensor.py tests/test_gemm_codegen.py
.venv/bin/python -m openassembler gemm --m 128 --n 128 --k 384 --dtype f16 --out-dir output/tensor-address
```
The archived address-confirmation JSON describes the original isolated experiment; its temporary generator was retired. Current tests reproduce the selected address contract.

## Code and retained evidence

- [Tensor pointer selection](../../../openassembler/backend/tensor.py)
- [Generator stage addressing](../../../openassembler/kernels/gemm.py)
- [Address confirmation](../../evidence/a100-mma/phase3/address-confirmation.json)
- [Async-copy encoding/round trips](../../evidence/a100-mma/encoding/address-ldgsts-roundtrips.json)
- [New transfer-pointer mode](direct-pointer-homes.md)

## What remains open

Address simplification must preserve asynchronous source collection and stage boundaries. Test further variants under full address/output checks and paired timing rather than counting operations alone.
