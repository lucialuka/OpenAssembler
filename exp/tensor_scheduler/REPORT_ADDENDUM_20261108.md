# Tensor scheduler experiment addendum — November 8, 2026

## Full scalar control sweep

The retained scalar cubins under `corpus/sm_80/latsweep` were replayed on an
A100-SXM4-40GB with two random trials per variant. The raw source-inclusive
record is `control-sweep-a100_20261108.json` and the compressed worker record
is `control-sweep-a100_20261108.raw.json.gz`. It covers all 42 producer/consumer
families exposed by `tools/lat_sweep.py`, including integer, floating-point,
predicate, store and wide-address consumers.

The generated `control-inventory.json` records 42 scalar families and 14 tensor
probe families. The standalone SM80 PTX corpus contains 541 forms, but those
forms are inventory-only until a meaningful dependency context is defined. The
measured scalar minima are mostly S4/S5 for fixed-pipe
RAW edges, S13 for the three predicate-to-guard probes, and S1/S2 for the
tested store/address cases. These are pair-specific hardware observations;
they are not a license to replace the production tensor HMMA8/drain15 policy.

## Lowered scheduler on actual OA blocks

`actual_block_schedule.py` lowers real `GemmPlan` kernels through the current
OA selector and allocator, splits the physical stream at barriers and control
edges, and feeds each basic block into `lowered_adapter.py`. The output is
`actual-block-schedule.json`; `test_actual_block_schedule.py` checks that HMMA
blocks are accepted only when every physical accumulator group retains its
K-order, while terminator blocks fail closed. The experiment model finds many
movable source positions, but it has not yet been encoded or timed as SASS.
This is evidence that a virtual-register scheduling phase is technically
reachable, not evidence that the current model is safe to promote.

## Reduction counters

Nsight Compute coverage for the reduction-overlap candidate is recorded in
`../weaknesses/reduction_overlap_ncu_report.md`. The candidate cuts OA dynamic
instructions by 10–16% and registers by 12–31, but OA long-scoreboard stalls
remain about 82–91% while matched ptxas is about 19–49%. The result rejects a
promotion based on liveness reduction alone and points to load/reduction-tree
dependency overlap as the next target.

## Current promotion state

The experiment layer now has measured fragment and traversal candidates, but
no production schedule or dispatch rule has been changed. The only candidate
that is close to promotion is a shape-specific load-next schedule for the
non-square FP16 cases above; it still needs independent held-out repeats,
counter comparison and an explicit register/occupancy threshold before a
production change is justified.

