# Cooperative population LayerNorm

Status: implemented

The explicit `rowwise --op layernorm` generator is validated for bounded FP32
rows. It has no affine weight/bias, saved statistics or backward implementation,
so it is not a drop-in replacement for a complete training LayerNorm module.

## Idea in plain language

Normalize a row by its population mean and variance using all 32 threads of a
warp instead of a serial thread. Consecutive threads load consecutive columns;
the reference launch groups four rows in four warps. Butterfly shuffles
combine each thread's partial sum, without shared-memory scratch. The
[shape-aware rowwise selector](shape-aware-rowwise-dispatch.md) now chooses
eight warps for two exact LayerNorm contracts, still one warp per row.

The implementation first subtracts the row's first element, computes the mean
of these differences, then reduces squared centered differences in a second
pass. Subtracting a common offset before summation avoids losing a small
variance while summing a large row mean in the tested cases. This is different
from unstable `E[x²]-E[x]²` variance reconstruction. Tail columns contribute
zero to the relevant sums; every lane participates in synchronization.

## Contract and tests

The operation returns `(x-mean)/sqrt(population_variance+epsilon)` in contiguous
FP32 `[rows,width]` output. Width is 1–1024; row/column tails are supported.
Arrays are four-byte aligned. The launch is `ceil(rows/warps_per_cta)` blocks
of `32*warps_per_cta` threads with no shared memory; four warps remain the
baseline, and manual choices are 1, 2, 4 or 8. Row comparisons fit u32 and the
total element count is at most 2³². Partial overlap is unsupported. Inputs, differences, sums and
squared differences must remain finite. NaN/Inf and arithmetic overflow are
outside the validated domain.

Epsilon must round to a **positive normal finite FP32** value. Merely accepting
a positive Python number is insufficient: it could round to zero or be flushed
to zero by the chosen reciprocal-square-root instruction. Values below 2⁻¹²⁶,
overflow, zero and nonfinite epsilon are rejected before writing artifacts.

The original November 6 nine-shape/five-pattern study uses FP64 formulas on exact FP32 inputs,
`rtol=1e-4, atol=2e-5`, all outputs and both canaries. Normal, zero, wide,
alternating and large-offset/small-variance inputs pass. The two reduction
families together check 17,408,740 OA outputs. The shifted candidate and its
same-source control pass the stringent large-offset gate where the old serial
and PyTorch controls miss it on eight nontrivial shapes; this is a finding
for this benchmark/gate, not a universal accuracy ranking.

## Results

Original November 6 four-warp **A100-SXM4-80GB** graph medians, microseconds
per operation:

| Shape | New OA | Same-source ptxas | Old serial OA | PyTorch CUDA |
|---|---:|---:|---:|---:|
| 128×128 | 2.240 | 1.792 | 64.624 | 3.520 |
| 1024×512 | 3.808 | 2.816 | 266.816 | 6.304 |
| 1024×1024 | 5.952 | 4.256 | 530.688 | 7.328 |

Twenty shuffled paired graph samples contain 32 operations each. Outputs are
poisoned after capture and validated after replay. Timing uses checked normal
inputs. Native outputs are preallocated; PyTorch's public LayerNorm allocates
its graph-managed output during capture, with no Python allocation on replay.
This compares those exact y-only APIs, not every possible tuned library kernel.

Same-source ptxas remains faster. Width-one is about 0.2 µs slower than legacy.
The main gain over the old OA source is cooperative work/data mapping, not a
general proof of superior code generation. No whole-step training result is
inferred from this standalone table.

## Current shape-specific geometry

The later November 10 selection and independent replay promotes eight-warps-
per-CTA launches for LayerNorm 4096×1 and 4096×32 at epsilon `1e-5`. Their
held-out OA-W4/selected ratios are 1.106–1.125× on an A100-SXM4-40GB. Formula,
per-row reduction order and numerical gates are unchanged. LayerNorm
129×1023/W2 and 4096×1024/W1 passed numerical checks but missed the frozen 3%
per-round timing cutoff, so they retain four warps. See the
[rowwise dispatch dossier](shape-aware-rowwise-dispatch.md) for all five-pattern,
canary, poisoned-graph, bitwise-control and identity evidence. These later
geometry measurements do not replace the original 80GB table or add affine,
saved-statistic or backward interfaces.

## Reproduce

Requirements: Python 3.12+, CUDA 12.8 tools; a real driver/GPU or authenticated
Modal client for execution. Run from the repository root:

```sh
openassembler rowwise --op layernorm --rows 1024 --width 512 --epsilon 1e-5 \
  --out-dir output_norm
openassembler compile output_norm/.build/generated/layernorm.ptx --out-dir output_norm_again
openassembler run output_norm --backend modal --budget 180 --job-timeout 60
.venv/bin/python -m pytest -q tests/test_reduction_codegen.py tests/test_cli_rowwise.py
```

The auto-run verifies analytic zero output from zero input, starting output
storage as NaNs so unwritten slots fail. It also requires bitwise equality with
ptxas. That smoke is separate from five-pattern floating-point validation:

```sh
PYTHONPATH=. python3 -c 'from exp.whole_stack.native_ops import build_payload; build_payload(rebuild=True)'
OA_REDUCTION_VARIANT=production PYTHONPATH=. .venv/bin/python exp/weaknesses/reductions_a100.py --build-only
OA_REDUCTION_VARIANT=production PYTHONPATH=. modal run exp/weaknesses/reductions_a100.py
```

Fresh serial-source controls use the current compiler; historical pre-fix
controls need their recorded baseline/payload. Actual GPU variant and hashes
must be read from each new result rather than assumed from the requested tier.

## Source, evidence and future work

- [Generator](../../../openassembler/kernels/reductions.py),
  [checked source/API](../../../openassembler/rowwise.py),
  [tests](../../../tests/test_reduction_codegen.py).
- [Final production run](../../../exp/weaknesses/reductions_runs/a100-20261106T133842.082727Z.json),
  [exact stream identity](../../../exp/weaknesses/reductions_production_identity.json),
  [hardware validator](../../../exp/weaknesses/reductions_a100.py).
- [Required compiler fixes](scalar-dependency-correctness.md),
  [softmax companion](cooperative-softmax.md).

Affine parameters, saved mean/inverse-standard-deviation, backward reductions,
wider rows and model-graph integration require their own accuracy and whole-step
tests. They are not supplied by choosing this forward generator.
