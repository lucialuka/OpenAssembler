# Current production A100 shape replay — November 11, 2026

The completed production-source campaign audits 42 exact GEMM contracts in two independent sessions. It installs **34 nonbaseline shape policies and eight measured baseline retentions**. Every promoted improvement exceeds 2% against the reference in each of six paired rounds; the observed OA baseline/selected range is **1.033–2.460×**. These are improvements over the previous OpenAssembler reference, not a claim of CUDA-toolchain superiority.

## Scope and method

- Contiguous unbatched `C=A@B.T`; FP16 or BF16 inputs, FP32 accumulation/output, alpha one, beta zero, aligned full tiles, no workspace or tails.
- 32 QKV/output/FFN role-shape contracts plus ten legacy/tile64 contracts. The candidate inventory was fixed before selection. It retains earlier shape losers: 64/128 tiles, copy-address reuse, direct/early pointers, combined loops, Morton8/grouped-N CTA order, and bounded load-next where representable.
- Each session builds 500 eligible production-source native candidates and matched ptxas controls. It checks 1,500 complete candidate/input outputs, canaries, unchanged inputs, finite/reference errors, bitwise ptxas/native and reference-policy equality, poisoned graph replay, resources, disassembly and protected tensor controls. Archives retain all PTX, native/ptxas cubins and SASS.
- Three rounds of eight shuffled paired samples use 16-launch resident-input CUDA graphs. Selection choices were frozen before independent held-out apps; promotion rechecks the frozen choice, not the held-out nominal winner.
- Selection workers: 28 A100-SXM4-40GB, ten A100-SXM4-80GB and four A100-80GB PCIe. Held-out workers: 41 A100-SXM4-40GB and one A100-SXM4-80GB. Ratios are computed within a paired worker/round; absolute runtimes are never pooled.
- Current GEMM compiler identity: `a010742ea1aef39d0a251aeeb547cf206ef21e6b0ecb2c1941642741488327ae`. Unrelated rowwise/runtime orchestration changes have a scoped continuity audit; raw protocol and earlier records remain untouched.

## Exact installed policies

`baseline/selected` is a speed ratio over all six native paired rounds. A baseline entry records no qualifying frozen improvement, not proof that every alternative is always slower. Policies specify the complete generated plan and launch; the installed selector also requires exact contract, current compiler and PTX body hashes.

| M × N × K | Input | Frozen installed policy | Baseline / selected range |
|---|---|---|---|
| 1024 × 1024 × 1024 | bf16 | `tile64_reuse_direct_early` | 1.374–1.385× |
| 1024 × 1024 × 1024 | f16 | `tile64_reuse_direct_early` | 1.367–1.387× |
| 1024 × 1024 × 4096 | bf16 | `tile64_reuse_direct` | 1.446–1.455× |
| 1024 × 1024 × 4096 | f16 | `tile64_direct` | 1.449–1.456× |
| 1024 × 1536 × 512 | bf16 | `tile64_reuse_direct` | 1.073–1.101× |
| 1024 × 1536 × 512 | f16 | `tile64_reuse_direct` | 1.072–1.101× |
| 1024 × 2048 × 512 | bf16 | `pointer_direct_early` | 1.035–1.044× |
| 1024 × 2048 × 512 | f16 | `pointer_direct` | 1.033–1.041× |
| 1024 × 3072 × 1024 | bf16 | `pointer_direct_early` | 1.062–1.071× |
| 1024 × 3072 × 1024 | f16 | `pointer_direct_early` | 1.063–1.072× |
| 1024 × 4096 × 1024 | bf16 | `pointer_direct_early` | 1.117–1.145× |
| 1024 × 4096 × 1024 | f16 | `pointer_direct_early` | 1.122–1.286× |
| 1024 × 512 × 2048 | bf16 | `tile64_reuse_direct_early` | 1.980–1.998× |
| 1024 × 512 × 2048 | f16 | `tile64_direct_early` | 1.962–1.972× |
| 1024 × 512 × 512 | bf16 | `tile64_reuse_direct_early` | 1.722–1.744× |
| 1024 × 512 × 512 | f16 | `tile64_reuse_direct_early` | 1.725–1.744× |
| 128 × 1536 × 512 | bf16 | `tile64_reuse_direct_early` | 2.274–2.287× |
| 128 × 1536 × 512 | f16 | `tile64_reuse_direct_early` | 2.273–2.287× |
| 128 × 2048 × 512 | bf16 | `tile64_reuse_direct_early` | 2.163–2.177× |
| 128 × 2048 × 512 | f16 | `tile64_reuse_direct_early` | 2.162–2.182× |
| 128 × 512 × 2048 | bf16 | `tile64_reuse_direct_early` | 2.449–2.459× |
| 128 × 512 × 2048 | f16 | `tile64_reuse_direct_early` | 2.450–2.460× |
| 128 × 512 × 512 | bf16 | `tile64_reuse_direct_early` | 2.237–2.261× |
| 128 × 512 × 512 | f16 | `tile64_reuse_direct_early` | 2.237–2.261× |
| 16384 × 4096 × 4096 | bf16 | `morton8` | 1.051–1.115× |
| 16384 × 4096 × 4096 | f16 | `morton8` | 1.060–1.114× |
| 192 × 320 × 384 | bf16 | `tile64_reuse_direct_early` | 1.070–1.079× |
| 192 × 320 × 384 | f16 | `tile64_reuse_direct_early` | 1.070–1.079× |
| 4096 × 1024 × 1024 | bf16 | `pointer_direct_early` | 1.135–1.172× |
| 4096 × 1024 × 1024 | f16 | `pointer_direct_early` | 1.101–1.141× |
| 4096 × 2048 × 2048 | bf16 | `pointer_direct_early` | 1.111–1.122× |
| 4096 × 2048 × 2048 | f16 | `pointer_direct_early` | 1.112–1.118× |
| 4096 × 2048 × 8192 | bf16 | `baseline` | 1.000–1.000× |
| 4096 × 2048 × 8192 | f16 | `baseline` | 1.000–1.000× |
| 4096 × 4096 × 4096 | bf16 | `baseline` | 1.000–1.000× |
| 4096 × 4096 × 4096 | f16 | `baseline` | 1.000–1.000× |
| 4096 × 6144 × 2048 | bf16 | `baseline` | 1.000–1.000× |
| 4096 × 6144 × 2048 | f16 | `baseline` | 1.000–1.000× |
| 4096 × 8192 × 2048 | bf16 | `baseline` | 1.000–1.000× |
| 4096 × 8192 × 2048 | f16 | `baseline` | 1.000–1.000× |
| 64 × 192 × 256 | bf16 | `tile64_reuse_direct` | 1.073–1.083× |
| 64 × 192 × 256 | f16 | `tile64_reuse_direct` | 1.067–1.083× |

Copy-address reuse is selected in 20 contracts, Morton8 in the two large tall contracts, and direct/early pointer combinations in other exact shapes. No combined-loop, grouped-N or load-next policy wins this frozen campaign; they remain legal explicit/tuning candidates. A prior loss does not remove a candidate from future shape-specific measurement.

## NVIDIA controls and reporting limits

Every session also compares its nominal selected native policy with same-source ptxas and a matched cuBLAS GemmEx call. OA is slower than same-source ptxas in **all 42 contracts in both sessions**; per-round native/ptxas ratios span about 1.038–1.297×. The nominal OA policy beats this cuBLAS call in all three rounds for six contracts in each session; this is not a tuned cuBLASLt algorithm search or a training-step comparison.

Seven held-out nominal selections differ from the pre-frozen policy. Their supplemental ptxas/cuBLAS comparisons describe the nominal selection only; they must not be attributed to the frozen installed policy. Promotion uses all-candidate native timing and exact per-policy ptxas correctness gates, which remain available for every frozen policy.

Compilation, input transfer, packing, graph/model integration and a complete training step are outside these resident-kernel timings. A100 variant, driver/toolkit, cache residency and clocks may change rankings. Unknown contracts keep the reference; explicit on-device tuning measures current admitted candidates.

## Evidence and reproduction

- [Campaign source and command instructions](README.md)
- [Frozen selection](runs/a100-production-shapes-copyinit-v5-20261110/frozen-selection.json)
- [Promotion decisions, per-round ratios and archive identities](runs/a100-production-shapes-copyinit-v5-20261110/promotion-decisions.json)
- [Selection progress](runs/a100-production-shapes-copyinit-v5-20261110/selection-progress.json) and [held-out progress](runs/a100-production-shapes-copyinit-v5-20261110/heldout-progress.json)
- [Scoped source continuity audit](runs/a100-production-shapes-copyinit-v5-20261110/selection-source-continuity-audit.json)
- [Installed exact table](../../openassembler/kernels/sm80_gemm_dispatch.json)

Each raw `selection/<case>/result.json` and `heldout/<case>/result.json` has a companion `tuning.artifacts.zip`, stdout and stderr. Earlier interrupted campaigns are retained as historical/failed evidence and are not automatic dispatch authority. Modal volume checkpoints preserve completed records across connection failures.

```sh
.venv/bin/python -m exp.dispatch_tuning.promote \
  exp/dispatch_tuning/runs/a100-production-shapes-copyinit-v5-20261110
```

Without `--table`, the reporting command may reuse saved decisions when record/protocol hashes agree. A `--table` invocation bypasses that cache and rechecks every archive and current native encoding. Use an experiment output path for a fresh offline audit, or `--table openassembler/kernels/sm80_gemm_dispatch.json` to install the complete audited campaign. Neither path changes raw evidence or starts a GPU job. To measure again, use a new campaign name and follow selection → freeze → held-out order in the README.
