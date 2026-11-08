# Direct pointer homes for asynchronous GEMM copies

Status: implemented

The later [joint option search](joint-tensor-options.md) and
[earlier pointer-placement experiment](early-gemm-pointer-advance.md) do not
extend this production contract. They use isolated experiment compilers, retain
the protected dependencies and show conditional gains plus tall-shape losses.

This is an explicit `--pointer-mode direct` tuning option. The default remains
`legacy`: the idea improves the measured MLP matrix products but regresses on
the tall control. “Implemented” does not mean selected automatically or faster
for every shape.

## Idea in plain language

The native tensor compiler already keeps each loop-carried 64-bit input pointer
in an aligned pair of registers. The old asynchronous-copy selection copied
that pair into two temporary registers before each staged transfer. Direct mode
lets the transfer read the existing pair, eliminating redundant MOV instructions.
The instruction-dependency model then sees the true pointer reads and protects
later updates. It does not remove a barrier or shorten the proven MMA timing.

For measured looped kernels, static non-NOP instructions fall **1,579→1,451**,
the full code shrinks **1,600→1,472 words**, and registers remain **216**.
Static HMMA/LDSM counts stay 512/128. The tiny control has no corresponding
looped-home opportunity and is unchanged. Existing HMMA issue gap 8, epoch
drain 15, async-copy visibility and matrix-load address dependencies remain.

The checked generated contract permits direct mode only with `stagepair`,
`m_fast` CTA order and no fused epilogue. FP16/BF16 inputs and FP32 accumulation/
output retain the normal full-tile GEMM layout/dimension contract. The source
annotation and manifest record the choice; recompiling the exact annotation
preserves it. Edited/unannotated PTX uses the legacy policy.

## Validation and measured results

Three independent 18-case runs cover nine shapes in both dtypes, zero/integer/
normal inputs, all outputs, matched-PTX bitwise equality and guard canaries:
**486 full method/input checks pass**. The public-CLI rebuild matches the entire
experimental native stream before independent replay. First/final runs are
40 GB A100s; the middle run is 80 GB. Ratios below aggregate paired comparisons,
not absolute timings across devices.

| M×N×K | FP16 legacy/direct ratio | BF16 ratio |
|---|---:|---:|
| 128×128×128 | 1.000× | 1.000× |
| 512×512×2048 | 1.046× | 1.046× |
| 512×2048×512 | 1.035× | 1.029× |
| 1024×1024×4096 | 1.065× | 1.064× |
| 1024×4096×1024 | 1.117× | 1.120× |
| 2048×512×512 | 1.035× | 1.036× |
| 4096×1024×1024 | 1.112× | 1.111× |
| 4096³ | 0.999× | 1.000× |
| 16384×4096×4096 | **0.955×** | **0.963×** |

The timing method uses ten launches per validated CUDA graph, fifteen shuffled
paired samples per round, with 3/3/2 rounds. A shorter instruction stream is
not enough: the tall case loses even though it removes the same copies.

Nine separate Nsight Compute profiles compare legacy/direct/ptxas on a winning,
neutral and losing shape. At 512×512×2048, dynamic instructions fall 577,856→512,320
and tensor activity rises 48.27%→51.71%, with the same 262,144 tensor instructions
and zero measured shared-bank conflicts. On the tall case, dynamic instructions
also fall, but tensor activity drops 76.83%→75.22%, long-scoreboard/barrier waits
increase and L2 hit rate falls 63.26%→60.58%. These associate the regression
with changed pipeline/cache behavior; one counter replay is not a causal proof
of every contribution and its duration is not benchmark timing.

Whole [hybrid MLP](native-hybrid-mlp.md) results are smaller and mixed: the large
model gains about 1.15–1.31% in training and 2.97–6.01% in inference; the small
model's training is neutral/slower and inference loses 2.48–5.22%. A kernel gain
does not become an application claim automatically.

## Reproduce

Run from the repository root with Python 3.12+, CUDA 12.8 tools, and an
authenticated Modal client for A100 tests:

```sh
openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 \
  --pointer-mode direct --out-dir output_direct
openassembler compile output_direct/.build/generated/gemm.ptx --out-dir output_direct_again
openassembler run output_direct --backend modal --budget 300 --job-timeout 120
.venv/bin/python -m pytest -q tests/test_tensor_pointer_homes.py tests/test_cli_gemm.py
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_homes.py --mode production --rounds 2
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_homes_counters.py
```

The ordinary run is an exact small-integer smoke, not the complete normal-input
test. The experiment driver checks frozen full-word hashes before timing and
refuses changed controls. Historical controls may require their retained cache
or reconstruction at commit `3507e6198a9323060241ab05c4558052d7bac2b3`; a new
compiler's output cannot silently impersonate that snapshot. Counter access
requires a working Nsight installation/permissions in the worker environment.

## Source and evidence

- [Tensor selector/configuration](../../../openassembler/backend/tensor.py),
  [plan validation](../../../openassembler/kernels/gemm.py),
  [annotation](../../../openassembler/gemm.py),
  [regressions](../../../tests/test_tensor_pointer_homes.py).
- [Production replay](../../../exp/weaknesses/gemm_pointer_homes-20261106T131150.445587Z.json),
  [initial](../../../exp/weaknesses/gemm_pointer_homes-20261106T130015.386450Z.json),
  [repeat](../../../exp/weaknesses/gemm_pointer_homes-20261106T130456.021475Z.json),
  [paired summary](../../../exp/weaknesses/gemm_pointer_homes_summary.json),
  [static identities](../../../exp/weaknesses/gemm_pointer_homes_static.json),
  [counters](../../../exp/weaknesses/gemm_pointer_homes_counters-20261106T130748.311735Z.json).

The [post-allocation control](post-allocation-pointer-forwarding.md) yields
identical streams, ruling out a changed physical register layout for these
cases. Global selection, Morton/fusion combinations and more aggressive
schedules require new matched tests before admission.


The later [early pointer-placement policy](early-gemm-pointer-advance.md) is integrated separately as an explicit shape-guarded option. It does not change the legacy default or imply that direct pointers are universally faster.
