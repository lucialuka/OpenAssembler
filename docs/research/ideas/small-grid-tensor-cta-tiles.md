# Smaller tensor CTA tiles for short transformer GEMMs

Status: implemented

This status covers the bounded production generator and measured cases. It is a
shape-aware native option, not a claim of general ptxas/cuBLAS superiority.

## Idea in plain language

A small GEMM can launch too few independent blocks when every block computes a
large output tile. A smaller tile gives the GPU more independent blocks and
shortens the work inside each one. That can help even if it repeats more
address calculations and operand reads overall.

The production generator can generate a 64×64×64 CTA tile with four 32×32 warp tiles,
FP16/BF16 operands, FP32 accumulation/output and two shared stages. Each thread
owns 32 accumulator values instead of 128. The tested source uses 88 allocated
registers and 32 KiB shared memory, versus 216 and 64 KiB in the default tile.
It retains HMMA8/drain15, async visibility, barriers and each accumulator's K
order. It changes no production compiler file.

## Contract and integrity gates

The standalone source plan requires M/N divisible by 64 and K divisible by 128,
contiguous A[M,K] and B[N,K], FP32 C[M,N]=A@B.T, alpha=1 and beta=0. It rejects
other precision, address overflow and unsupported policy combinations. Current
hardware evidence uses 128-aligned model cases; it does not establish all
possible 64-aligned shapes, tails, strides or transpose handling.

The public `GemmPlan(tile="64x64")` generates the checked production PTX and
launch annotation. Explicit plan fields enable direct pointer homes, early
pointer advancement and `address_init="reuse"`, which derives copy bases using
64-bit increments of `32*K` bytes. The default (original address setup) source
remains byte-identical to its hardware-validated golden. Tests check logical
copy coverage, shared swizzle/LDSM boundaries, K-order, and exact emitted-source
identity. Hardware gates cover zero/integer/normal outputs, canaries, poisoned
graphs, immutable inputs, same-policy/baseline ptxas bits, encoded controls,
resources, and paired timing.

## Measured small output-projection case

For M/N/K=128/512/512 FP16, the output grid grows from four to 16 CTAs.

| Independent worker | Default OA | OA tile64 | Same-source tile64 ptxas | cuBLAS GemmEx |
|---|---:|---:|---:|---:|
| A100-SXM4-40GB selection | 21.25–21.38 µs | 11.01 µs | 8.45 µs | 7.68 µs |
| A100-SXM4-80GB frozen replay | 19.97 µs | 10.24 µs | 7.94 µs | 7.17 µs |

Each row contains three paired rounds with 12 samples. The candidate reduces
OA runtime by about 48–49%. It also beats the old 128-tile ptxas control, but
not ptxas compiling the smaller-tile source. The replay includes direct+early
pointer OA at 19.20 µs, so the gain survives that stronger OA control. Absolute
times are not pooled across devices.

The selection record passes 15 method/input and five graph checks. The
independent replay record passes 33 method/input and 11 graph checks. Its
recorded source identity matches the selection run. These records are
historical evidence for the source family, not a current binary attestation.

## What the counters do and do not explain

The [separate counter study](../../../exp/shape_coverage/a100-small-tile-counters_20261109T113502.971650Z.json)
passes all 15 output checks and collects all 20 requested metrics for six
profiles. Dynamic tensor instructions stay at 16,384, while OA total
instructions rise from 42,512 to 75,328 and global-load sectors double from
32,768 to 65,536. Shared-bank conflicts are zero for both tiles. Thus the gain
is not a reduction in total work or bank conflicts.

The register-limited block ceiling rises from two to five, but achieved
active-warp percentages remain about 6.2%; the counters do not demonstrate
higher achieved occupancy. More grid parallelism and shorter per-CTA work are
supported explanations, not an isolated causal proof. Wait stalls fall from
50.00% to 40.43%, while both short and long scoreboards increase. These
percentages are not additive fractions of runtime.

At the same 64-tile geometry, OA executes 58,944 non-tensor instructions versus
ptxas's 24,256 and has more wait stalls. This remains a separate compiler gap.
The observed cuBLAS invocation launches 64 CTAs and is faster despite 243,200
total dynamic instructions. Its kernel name is an observation, not proof of an
extracted proprietary algorithm.

## Reproduce

Run from the repository root with the local test environment and authenticated
Modal client:

```sh
.venv/bin/python -m pytest -q exp/shape_coverage/test_small_tile_gemm.py \
  exp/shape_coverage/test_gemm_policy.py
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --heldout --model-case small_fp32_context --role attention_output_projection \
  --dtype f16 --tile-sweep --pointer-sweep --rounds 3 --samples 12
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --heldout --tile-sweep --pointer-sweep --rounds 3 --samples 12
modal run --detach exp/shape_coverage/a100_small_tile_counters_modal.py
```

The [explicit generator](../../../exp/shape_coverage/small_tile_gemm.py),
[layout/encoding tests](../../../exp/shape_coverage/test_small_tile_gemm.py),
and [evidence inventory](../../../exp/shape_coverage/evidence-ledger.json) retain
the implementation, methods and limitations. NCU replay duration is never used
as timing. Kernel timings exclude packing, graph integration and whole training.

## Broader role/shape validation

The [32-case A100-SXM4-40GB sweep](../../../exp/shape_coverage/a100-role-gemm_20261109T113850.802215Z.json)
covers four model geometries, QKV/output/FFN-up/FFN-down and FP16/BF16. All
1,296 method/input checks and 432 poisoned-graph checks pass; source identity
remains unchanged. Tile64 is faster than default OA in all three rounds on
18 cases and slower on 14.

| Cohort / role | Tile64 runtime change versus default OA |
|---|---:|
| Small-token QKV/output/FFN-up/FFN-down, both dtypes | About 45–54% lower |
| Base-report output projection | About 37–39% lower |
| Base-report FFN-down | About 46–47% lower |
| Medium output projection | About 21–22% lower |
| Medium FFN-down | About 27–28% lower |
| Medium QKV | About 33–35% higher |
| Medium FFN-up | About 5–9% higher |
| All large-role cases | About 24–49% higher |

Base-report QKV gains are small and can lose against the pointer control;
base-report FFN-up regresses. The best OA option remains slower than the best
measured ptxas option in every case. Selected default-library cases do lose to
OA, but this is not a tuned cuBLASLt comparison or whole-training result.

## Decision and remaining work

Keep this family shape guarded. The broader sweep establishes real gains and
regressions, so the small output case cannot justify a universal tile change.
`--dispatch auto` only selects a current exact table entry; an unknown or stale
entry uses the 128×128 reference. `--tile 64x64` remains available for explicit
manual experiments, and `--autotune` can test the currently admitted candidates
on an A100. Unseen alignment classes, toolkit versions and whole-model
substitution still need validation. A native model route and full
forward/backward/optimizer state check remain separate integration gates.

The production dispatch table currently fails closed to the 128×128 baseline:
its fingerprint predates the expanded candidate set. The replacement 42-case
campaign freezes the current source identity and compares tile64 with original
and reused copy addresses, direct pointer homes, early/late advancement, the
128×128 policies, load-next, Morton, and grouped-N where each plan is admissible.
No new automatic entry is enabled until independent selection and held-out
records pass the promotion auditor.

The production table
[`sm80_gemm_dispatch.json`](../../../openassembler/kernels/sm80_gemm_dispatch.json)
retains the previous compiler identity and is not active for this source tree.
Unknown/stale contracts use the reference. The current pilot records remain
useful exact-shape history; matched ptxas and cuBLAS remain faster.

```sh
.venv/bin/python -m exp.shape_coverage.audit_role_evidence
```
