# Morton8 logical CTA traversal

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Assign neighboring output tiles in a Z-shaped Morton order so nearby launches may reuse nearby operands in cache. This changes which logical matrix tile a CTA computes; it does not tell the hardware which SM must execute it.

## Findings and implementation decision

`--cta-order morton8` is an explicit option for complete 8×8 CTA groups, requiring M and N divisible by 1024. The name means an eight-by-eight local CTA group: three coordinate bits from each axis, interleaved into six local-order bits. “Morton6” could refer to those six bits, but is not a different supported policy without a concrete alternative mapping; there is no `morton6` CLI value. This traversal is not the row/column layout of a hardware MMA instruction. The current mapping was corrected to decode each aligned physical group independently; the earlier flattened rectangular-grid mapping is superseded.

The November 5 full-output run measured tall 16384×4096×4096 FP16 at 2.528 versus 2.728 ms (7.34% less time), and BF16 at 2.478 versus 2.652 ms (6.54%). Same-source ptxas remained faster at 2.369/2.327 ms. MLP Morton candidates regressed and square results were near noise. The [current independent production replay](../../../exp/dispatch_tuning/RESULTS.md) now automatically selects Morton8 for the two exact 16384×4096×4096 FP16/BF16 contracts, with 1.051–1.115× baseline/selected ratios across six paired rounds. Other exact shapes select different policies or retain baseline. It remains an explicit option for other admitted shapes and is neither a universal row-major replacement nor an MMA-instruction layout mode.

## Method and validity checks

Prove a bijection of logical tiles, check every output/canary on zero/integer/normal inputs, and compare the identical mainloop under another CTA assignment. Keep the older mapping's results labeled superseded. Paired timing controls cache/clock differences within a worker.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m openassembler gemm --m 16384 --n 4096 --k 4096 --dtype f16 --cta-order morton8 --out-dir output/morton
.venv/bin/python -m pytest -q tests/test_gemm_codegen.py tests/test_cli_gemm.py
modal run docs/evidence/a100-mma/current-validation/validate_modal.py
```
The validator contains the corrected mapping and matched controls. `run output/morton` is only its exact-integer smoke if used without a separate floating validator.

## Code and retained evidence

- [Morton coordinate generator](../../../openassembler/kernels/gemm.py)
- [Bijection tests](../../../tests/test_gemm_codegen.py)
- [Current full-output/timing record](../../evidence/a100-mma/current-validation/validation_20261105T074142.327676Z.json)
- [Superseded earlier production result](../../evidence/a100-mma/phase6-validation-20261104-superseded.json)

## What remains open

Measure other group sizes and actual training shapes. Names such as Morton6/8 must identify the concrete mapping, not imply hardware support or guaranteed SM co-location.
