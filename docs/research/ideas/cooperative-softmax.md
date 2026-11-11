# Cooperative FP32 row softmax

Status: implemented

The explicit `rowwise --op softmax` generator passes its declared numerical
contract. It replaces the old serial row mapping when the caller chooses this
operator; it is not automatic PyTorch attention/softmax substitution.

## Idea in plain language

Softmax needs the maximum and sum of a row before producing probabilities.
The old example gave one entire row to one thread, which walked all columns
serially. The new kernel lets a complete 32-thread warp share a row. Adjacent
threads load adjacent values, keep their assigned columns in registers, and
exchange partial results through synchronized butterfly shuffles. The reference
launch groups four rows in four warps, with no shared-memory scratch. The
[shape-aware rowwise selector](shape-aware-rowwise-dispatch.md) now chooses two
or eight warps for three exact softmax contracts; each warp still owns one row.

The kernel subtracts the row maximum, computes approximate FP32 exponentials,
reduces their sum and applies a reciprocal. Out-of-range tail columns contribute
neutral values while every lane participates in each shuffle. A tail row exits
uniformly for the whole warp. Native synchronization and source-collection
dependencies are documented in [the compiler-correctness dossier](scalar-dependency-correctness.md).

## Contract and methodology

Input/output are contiguous FP32 `[rows,width]` arrays, four-byte aligned, with
ABI `(X,Y)` and no partially overlapping buffers. Width is 1–1024, rows fit an
unsigned 32-bit comparison, and total element count is at most 2³². The launch
uses `32*warps_per_cta` threads, `ceil(rows/warps_per_cta)` blocks and zero
shared memory. Four warps are the baseline for unknown/stale contracts;
manual selection permits 1, 2, 4 or 8. Inputs are bounded
finite FP32: extreme finite arithmetic overflow and NaN/Inf are not covered.

The original November 6 four-warp study tested nine shapes, including widths 1,31,33,127,128,512,513,1023,1024 and row tails.
Each is checked on normal, zero, wide, alternating and large-offset/small-variance
inputs against FP64 formulas evaluated on the exact FP32 inputs. The unchanged
softmax gate is `rtol=5e-5, atol=1e-6`. Every native output and both guard regions
are checked. Softmax and LayerNorm together check **17,408,740 OA elements** in
the final production run, all passing.

Graph timing uses 20 shuffled paired samples with 32 operations per replay.
Outputs are poisoned after capture, then replayed and checked in full, proving
that a graph actually performs its work. Timing uses the same checked normal
inputs; a failing candidate is not eligible for a speedup claim.

## Results and limits

Original November 6 four-warp paired graph medians on the actual
**A100-SXM4-80GB**, in microseconds:

| Shape | New OA | Same-source ptxas | Old serial OA | PyTorch CUDA |
|---|---:|---:|---:|---:|
| 128×128 | 2.816 | 2.208 | 95.824 | 3.168 |
| 1024×512 | 3.968 | 2.912 | 306.384 | 4.512 |
| 1024×1024 | 6.080 | 4.256 | 618.496 | 7.280 |

These large old/new gains mostly repair the algorithm's thread/data mapping;
ptxas compiling the same cooperative PTX remains faster. Width-one is about
0.2 µs slower than the old serial kernel, and 7×31 does not beat the PyTorch
control. No universal library or transformer-training acceleration is claimed.
An earlier integrated candidate ran on 40 GB; all 18 native streams of both
operator families match the promoted build byte for byte. Absolute times from
the two device variants are not pooled.

## Current shape-specific geometry

The later November 10 selection and independent replay covered 58 rowwise
operator/shape contracts. Three softmax entries passed the per-round 3% gate:
129×1023 selects two warps; 4096×1 and 4096×32 select eight. Held-out gains
against the four-warp OA control range from 1.034–1.126× on an A100-SXM4-40GB.
All five-pattern full outputs, canaries, input immutability, poisoned graphs and
bitwise OA-control checks passed. These are geometry gains, not changes to the
softmax formula or a whole-model result. The
[rowwise dispatch dossier](shape-aware-rowwise-dispatch.md) owns the exact
records and rejected near-threshold candidates. Its 40GB times are not pooled
with the original 80GB table.

## Reproduce

Run from the repository root with Python 3.12+, CUDA 12.8 tools and an
authenticated optional Modal client for hardware. The CLI smoke is independent
of the larger numerical experiment:

```sh
openassembler rowwise --op softmax --rows 1024 --width 512 --out-dir output_softmax
openassembler compile output_softmax/.build/generated/softmax.ptx --out-dir output_softmax_again
openassembler run output_softmax --backend modal --budget 180 --job-timeout 60
.venv/bin/python -m pytest -q tests/test_reduction_codegen.py tests/test_cli_rowwise.py
```

The auto-run uses zero input and NaN-poisoned output, checks every analytic
constant `1/width` and retains strict OA/ptxas bitwise comparison. It is a smoke
test, not the five-pattern accuracy result. To rerun both full reduction families:

```sh
PYTHONPATH=. python3 -c 'from exp.whole_stack.native_ops import build_payload; build_payload(rebuild=True)'
OA_REDUCTION_VARIANT=production PYTHONPATH=. .venv/bin/python exp/weaknesses/reductions_a100.py --build-only
OA_REDUCTION_VARIANT=production PYTHONPATH=. modal run exp/weaknesses/reductions_a100.py
```

The first command prepares a serial-source control; its current compiler is
not automatically the historical pre-fix compiler. Preserve recorded controls
or reconstruct them at the recorded baseline commit when reproducing old/new
compiler comparisons. Fresh runs record their own actual device and hashes.

## Source and evidence

- [Generator](../../../openassembler/kernels/reductions.py),
  [CLI/checked annotation](../../../openassembler/rowwise.py),
  [runner](../../../openassembler/runner.py),
  [tests](../../../tests/test_reduction_codegen.py).
- [Final full run](../../../exp/weaknesses/reductions_runs/a100-20261106T133842.082727Z.json),
  [stream identity](../../../exp/weaknesses/reductions_production_identity.json),
  [ordinary CLI smoke](../../../exp/weaknesses/cli_rowwise_smoke.json),
  [validator](../../../exp/weaknesses/reductions_a100.py).

Masking, arbitrary layouts/dtypes, wider rows, backward and attention fusion
remain separate work. A valid standalone y-only kernel does not supply those
training interfaces by itself.
