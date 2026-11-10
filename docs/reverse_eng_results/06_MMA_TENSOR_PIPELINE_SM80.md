# SM80 MMA tensor pipeline

## Production tensor contract

The native GEMM path targets full-tile contiguous RCR GEMM:

* FP16 or BF16 inputs;
* FP32 accumulation and output;
* 128×128×64 CTA tile;
* four 64×64 warp tiles;
* 128 threads;
* 64 KiB dynamic shared memory;
* `cp.async` staging, paired LDSM fragments and vector stores.

The path computes the declared GEMM contract; it does not imply arbitrary
strides, transpose modes, tails, alpha/beta, TF32 or full training graph
integration.

## Dependency classes

The reverse-engineering work separates four tensor concerns:

1. dependent accumulator RAW latency;
2. independent HMMA initiation spacing;
3. operand readiness from LDSM, async-copy visibility, barriers and addresses;
4. observer dependencies from stores, scalar consumers, barriers and control.

The old production policy compresses too much into HMMA8 and a post-chain
drain. The experiment graph keeps the classes separate but refuses to cross
basic-block or proven dependency edges.

## Clocked A100 result

The corrected `%clock64` probe subtracts an empty-loop control and measures:

| Probe | Cycles per HMMA |
|---|---:|
| One dependent accumulator chain | ~24.02 |
| Two independent accumulator chains | ~12.01 |
| Three independent accumulator chains | ~8.01 |

This supports independent-chain interleaving, but does not authorize universal
S1 controls. The correctness sweep still found S7-safe controls for several
dependent/observer layouts and S1 only for the tested three-chain layout.

## Current decision

The production tensor backend retains HMMA8, drain15, async-copy visibility,
LDSM address-WAR and observer protections. The next scheduler must model
register pressure, exact fragment ownership and observer edges together before
promotion.

Two bounded alternatives now have explicit production contracts. The
load-next/compute-current fragment schedule is admitted only for
1024×4096×1024 and 4096×1024×1024, with FP16/BF16, stagepair, legacy pointers,
late advancement and no fused epilogue. Grouped-N CTA traversal is admitted as
`grouped_n8` when the N tile count is divisible by eight. These options remain
explicit unless an exact current dispatch-table entry admits them. The general
dependency scheduler remains unpromoted; the current table uses exact
source-identity-locked tile64 and pointer policies and falls back on unknown or
stale contracts.

The newer [fragment placement study](../../exp/shape_coverage/a100-role-gemm_20261109T111245.728446Z.json)
tests split-2/4/6 load-next placement with the same HMMA8/drain15 controls and
per-accumulator K order. All full FP16/BF16 checks pass on the medium FFN-up
role, but split-2/6 do not beat split-4 there. Register counts are 244/240/236;
fewer registers are not sufficient evidence of a better schedule.

The [current lowered-region audit](../../exp/tensor_scheduler/actual-block-audit_20261109.json)
also supersedes the earlier model's region acceptance. All 45 audited actual
GEMM regions now fail closed because the adapter cannot fully represent their
visibility, predicate or implicit state. A source-hashed `a100_clocked_24_8`
profile models the measured accumulator and initiation classes; all remaining
readiness/observer distances are explicitly uncalibrated. No general reordered
stream is encoded, and predicted model cycles remain distinct from measurements.

See [A100 tensor dependency scheduler](../research/ideas/a100-tensor-dependency-scheduler.md) and
[Native SM80 tensor GEMM](../research/ideas/native-tensor-gemm.md).

