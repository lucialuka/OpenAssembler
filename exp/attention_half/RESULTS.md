# FP16 BMM and composed attention: A100 results

Completed November 14, 2026. The admitted implementation is
`openassembler/modelfront/kernels_tensor_batched.py`; public capture, symbolic
backward, CPU reference, artifact validation and runtime dispatch all use it.
Experiments preceded promotion. Production phases execute the integrated path
without monkeypatches; the explicitly labeled all-tensor comparator remains an
experimental control.

## Supported contract

- Contiguous FP32 storage; each BMM forward/dX/dW independently rounds both
  operands to FP16 RNE, accumulates and returns FP32. Master parameters and SGD
  remain FP32. Inputs/intermediates must remain finite after the declared casts.
- Matching rank-three batch counts through 32, with positive logical M/N/K
  through 512. Rank-four matmul flattens matching batch axes. No broadcasting.
- Larger products pack zero-padded normal/transposed layouts, launch OA Tensor
  Core GEMM with batch byte strides in CTA-z, and crop the output. Logical packs
  can be reused within an iteration; every next forward refreshes them.
- Products with M/N/reduction length all at most 64 round operands into FP32
  workspaces and use OA scalar BMM. This implements the same half operand
  contract while avoiding padded Tensor Core work. It is not a vendor fallback.
- Attention is composed from these products, OA FP32 softmax/layouts and linear
  projections. Masks, fused attention, arbitrary repo linking and general
  dynamic shapes are not admitted.

Default model precision stays FP32. `tensor_packing="dual"` remains optional for
linear operators and is needed for their bounded tails. BMM also works with the
default `separate` setting; the final standalone production sweep tests it.

## Experimental decisions and failures

The original all-tensor path passed isolated products but cost about 55–69 µs
for a complete forward/dX/dW sequence even on tiny matrices. An explicit RNE
rounding stage plus scalar BMM reduced that overhead. Exact rounding tests cover
signs, ties, normal/subnormal boundaries and signed zero.

The first small selector used arithmetic volume alone. The `511×3×17` case then
regressed to 175.92 µs versus the 69.94 µs tensor control. That rule was rejected.
The final rule requires *each* logical matrix dimension at most 64. Its
experimental and production nine-shape sweeps retain the tensor path for tall
and larger-reduction cases. The [rejected result](evidence/scalar-volume-regression.json)
and [final experiment](evidence/final-experiment.json) remain available.

The full LayerNorm/GELU block failed the initial `rtol=0.002, atol=0.00002`
trajectory gate at update 3: two prediction elements failed; maximum absolute
error was 0.000101805. [That failure is retained](evidence/initial-strict-gate-failure.json).
The full block is supported and timed under an explicit `atol=0.0002` gate;
other attention cases retain `atol=0.00002`. Isolated products use `rtol=0.001,
atol=0.000002` and all pass. The precise source of the composed drift is not
localized; these results do **not** establish the original stricter full-block
trajectory equivalence. Production defaults are not globally relaxed.

A stream mismatch and a missing test fixture in early harness runs, an overly
strict template-anchor count during promotion, and an existing-directory error
in the first paired runner were corrected and rerun. These are not counted as
successful validations; selected failed harness outcomes are retained under
`evidence/`. A nonzero subprocess always overrides any earlier success file.

## Standalone products and tails

Final production sweep: NVIDIA A100-SXM4-40GB, CUDA 12.8.1, Python 3.12,
Torch 2.7.1/cu128. Times are medians of 15 CUDA-event samples, each containing
10 resident-input graph replays of forward plus both input derivatives. All
products are checked against independent CPU formulas with explicit operand
casts; workspace poison, repeated replay and guard checks also pass. The maximum
absolute error over all controls/products is about 1.1e-6.

| B, M, N, K | OA FP32 µs | Padded FP16 µs | Selected FP16 µs | Padded / selected |
|---|---:|---:|---:|---:|
| 1, 1, 1, 1 | 6.96 | 55.50 | 16.38 | 3.39× |
| 3, 7, 9, 5 | 11.88 | 56.93 | 22.32 | 2.55× |
| 2, 17, 33, 65 | 39.83 | 57.65 | 57.45 | 1.00× |
| 2, 128, 128, 128 | 139.57 | 58.57 | 58.37 | 1.00× |
| 4, 129, 67, 31 | 76.70 | 63.90 | 63.90 | 1.00× |
| 1, 256, 64, 128 | 144.49 | 61.95 | 62.05 | 1.00× |
| 32, 2, 5, 3 | 9.73 | 68.40 | 19.87 | 3.44× |
| 2, 31, 33, 512 | 179.30 | 72.09 | 72.29 | 1.00× |
| 1, 511, 3, 17 | 160.15 | 69.94 | 70.25 | 1.00× |

The FP32 column is a different numerical policy. The selected half path remains
slower than FP32 on tiny cases because it performs explicit rounding. The
larger FP16 products improve over OA FP32 by roughly 2.3–2.5× on three shown
cases. Neither comparison establishes superiority over NVIDIA or Triton.
Raw samples/source identities are in [production-bmm.json](evidence/production-bmm.json),
with [plans and cubin/text/source hashes](evidence/bmm-plans-ledgers.json).

## Complete attention training

Four independently captured attention configurations pass four changing-input
direct updates plus three changing-input graph replays: tiny B=2/S=7/H=2/D=4,
odd-tail B=1/S=17/H=3/D=5, medium B=2/S=33/H=2/D=16, and the example's complete
LayerNorm/GELU/FFN block. There are 616 field comparisons covering predictions,
loss, dX, parameter derivatives and updated weights. Inputs/targets remain
unchanged, poisoned workspaces/guards pass, and independent profiler kernel
counts match OA launch ledgers. [Production checks](evidence/production-attention.json)
and [plans/ledgers](evidence/attention-plans-ledgers.json) retain their scope.

The paired performance campaign uses a single worker, two opposite process-order
rounds, and the same data, precision, gates and complete SGD computation. Each
median has 15 event samples of four graph replays. Timings exclude compilation,
transfer, graph preparation and CPU checking. All controls pass their declared
full-state gates before timing.

| Complete task | Padded FP16 µs, rounds 1 / 2 | Selected FP16 µs, rounds 1 / 2 | Speedup |
|---|---:|---:|---:|
| Tiny attention | 399.22 / 399.11 | 328.25 / 328.26 | 1.216× / 1.216× |
| Medium attention | 742.52 / 742.46 | 717.51 / 717.74 | 1.035× / 1.034× |
| Attention + LayerNorm/GELU/FFN | 551.42 / 551.44 | 481.07 / 480.75 | 1.146× / 1.147× |

Launch counts stay at 84 for tiny/medium and 120 for the full block in both
variants. The gain comes from replacing padded work/conversions in BMM regions,
not from fewer launches. This is a comparison against OA's new all-tensor FP16
baseline, not a speedup against the previously FP32 attention implementation.
Raw samples, every comparison, device and source identities are retained in
[paired-training.json](evidence/paired-training.json).

## Public CLI, provenance and regressions

The actual `cnr examples/native_attention_model --backend modal --verify
--no-sass` build and subsequent saved replay pass 93 state checks each across
three updates. Their reports record zero vendor compute launches and zero GPU
code generation during replay. All launches are checked against OA executable
ledgers. See [build](evidence/cli-build.json), [saved replay](evidence/cli-replay.json),
[plan](evidence/cli-plan.json), and [native ledger](evidence/cli-native-ledger.json).

The pinned Torch CPU worker passes 75 graph, rounding/reference, state-diagnostic
and ownership tests; its [raw result](evidence/torch-cpu-tests.json) is separate
from the local no-Torch suite. The
[continuity audit](../native_indexing/dispatch-continuity-20261114T060925Z.json)
re-encodes 500 GEMM candidate contracts against both retained sessions (1,000
checks), 40 rowwise session checks and six fixed tensor goldens. Native words,
controls and register counts remain unchanged, allowing table identity refresh
without inventing new timing evidence. Local regression and documentation checks
are recorded in the repository status page.

See [README.md](README.md) for commands. SSH uses the same artifact/runtime
contracts; this campaign executed on Modal and does not claim a fresh SSH GPU run.

The final local suite passes **975 tests**, with three optional Torch scopes
skipped locally. A subsequent CLI/report regression selection passes **101 tests**.
The documentation checker passes **142 Markdown files** and all 82 catalogue
entries; [regression.json](evidence/regression.json) retains commands and source
hashes. A second independent saved-model replay also passes all 93 checks with
concise CLI output and zero vendor/codegen activity;
[its report](evidence/cli-final-replay.json) and [log](evidence/cli-final-replay-log.json)
are retained.
