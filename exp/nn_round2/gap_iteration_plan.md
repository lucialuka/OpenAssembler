# Four-iteration performance-gap plan

This is an experiment plan derived from the current OpenAssembler source and
the retained A100 evidence. It does not promote any candidate into production.

## Current causal observations

The scalar build path in `openassembler/build.py` is:

```text
Selector -> coalesce/DCE -> physical allocation -> lower to SASS -> list schedule
```

The allocator is therefore upstream of the main scalar list scheduler. The
allocator in `openassembler/backend/ra.py` is a linear scan with aligned groups;
it has no general spill policy or schedule-aware occupancy objective. The list
scheduler in `openassembler/backend/sched.py` uses a small fixed latency table,
preserves memory/barrier order through one `last_mem` chain, and does not model
per-pipeline issue capacity, register-bank pressure, occupancy, or the tensor
operand collector.

The native tensor path keeps additional safety dependencies: async-copy address
WAR, LDSM visibility, barriers, HMMA accumulator use, HMMA8 issue spacing and
drain15. Those edges must remain present in every candidate. Existing clocked
probes are not yet a complete tensor latency model.

The strongest retained counter observations are:

| Case | OA observation | ptxas observation | Likely direction |
|---|---:|---:|---|
| Wide GEMM, OA legacy | 4.90M dynamic instructions; 57.7% tensor active; 41.5% wait | 3.23M; 72.0% tensor active; 35.1% wait | Reduce address/control work and improve copy/tensor overlap |
| Wide GEMM, OA direct | 4.37M instructions; 60.8% tensor active | 3.23M; 72.0% tensor active | Pointer homes help, but do not close the gap |
| Tall OA legacy → Morton | DRAM 3.05 GB → 1.09 GB; L2 64.2% → 85.8% | — | Traversal locality is a major shape-specific lever |
| Tall ptxas m-fast → Morton | DRAM 4.53 GB → 1.22 GB; L2 52.2% → 83.4% | — | Dispatch must consider CTA traversal and cache locality |
| RMSNorm backward FMA | 61 → 44 registers; 0.975M → 0.840M instructions; active warps 39.7% → 53.5% | Long scoreboard remains ~85% | Contraction/live-range reduction helps but does not solve memory/scoreboard waits |
| Softmax backward FMA | 56 → 43 registers; 0.651M → 0.586M instructions; active warps 42.1% → 54.7% | Long scoreboard remains ~88% | Same conclusion |

All values are diagnostic Nsight replay measurements with uncontrolled clocks;
they explain candidate mechanisms but are not replacements for paired graph
timing.

## Iteration 1 — instruction selection and virtual-register scheduling

**Hypothesis.** Some OA arithmetic sequences retain extra moves, address
formation and scalar operations because scheduling happens after physical
allocation. Scheduling a virtual-register IR before allocation should expose
more legal movement and allow copy/FMA coalescing without committing to poor
physical homes.

**Experiment.** Build an isolated backend variant under `exp/` that:

1. lowers PTX to the existing virtual IR;
2. applies only proven local rewrites (`mul` + `add` to an explicit FMA where
   the declared rounding contract permits it, address `mad` folding and dead
   copy elimination);
3. schedules virtual values using dependency, critical-path and live-range
   cost;
4. allocates aligned groups after scheduling;
5. lowers and encodes with the existing SM80 encoder.

Test ReLU/GELU/SiLU, integer div/rem and backward reductions. Require full
outputs, guards, unchanged inputs, disassembly and no register spill. Report
instruction counts, registers, active-warps/occupancy counters and paired OA /
ptxas timing. Keep the explicit FMA reduction template as a reference candidate,
not proof that every multiplication/addition can contract.

**Promotion rule.** Promote only a bounded rewrite or a per-kernel dispatch
rule that wins at least two fixed holdout shapes and does not change any failed
numerical gate. A global scheduler replacement needs scalar regression coverage
and no resource regression on the existing ten-case suite.

## Iteration 2 — protected tensor scheduling and pointer placement

**Hypothesis.** Address generation and copy-pointer lifetime can overlap with
tensor work if pointer updates move immediately after the next async-copy commit.
The operation can also be shape-sensitive because cache traversal and operand
traffic differ between square and tall grids.

**Experiment.** Reuse `tensor_followup.py` to compare legacy, direct, early-pointer,
Morton and fused combinations. Preserve the complete copy/LDSM/HMMA/barrier
multiset and all known dependency edges. Test two stages first, then three/four
stages only when the full output and canary checks pass. Use square, wide and tall
FP16/BF16 forward/dX/dW cases, and compare same-source ptxas for each PTX source.

**Promotion rule.** A candidate may become a shape-specific option only when it
wins repeated graph timings on a fixed holdout and loses no correctness check.
Early pointer placement currently helps the wide case but loses on tall; this is
evidence for dispatch, not a universal schedule.

## Iteration 3 — joint scheduling/allocation and resource-aware costing

**Hypothesis.** Register choices constrain post-allocation movement. A schedule
that minimizes local latency can increase live ranges and register pressure,
which reduces active warps or increases operand-collector stalls.

**Experiment.** Compare the current order (allocate then schedule) with an
isolated virtual schedule/allocate loop. Add a bounded cost function:

```text
critical-path latency
+ live-register peak penalty
+ aligned-group fragmentation penalty
+ occupancy bucket penalty (128/256-thread launch)
+ memory/async dependency violation = infinity
```

Use 2–4 bounded iterations of schedule → allocate → reschedule only where
physical register assignments change the dependency graph. Reject any spill,
implicit register reader loss, vector-group misalignment, or source-WAR/HMMA
dependency deletion. Test scalar FMA/backward reductions and a representative
tensor mainloop; inspect NCU counters rather than instruction count alone.

**Promotion rule.** Require a paired timing improvement on at least one large
shape, equal or lower register count, no new numerical failures, and no slowdown
on the retained scalar regression set. Keep the algorithm behind an experiment
flag until a broader workload confirms it.

## Iteration 4 — reduction trees, geometry and whole-step dispatch

**Hypothesis.** `ptxas` is selecting contraction, reduction trees, vector width,
warp count and shape-specific synchronization that the current row kernels do
not model. The existing FMA result shows that local contraction and lower live
ranges can matter, but row count and width change the outcome.

**Experiment.** For softmax, RMSNorm and LayerNorm forward/backward, sweep:

- scalar versus four-element loads;
- four versus eight warps;
- pairwise/tree reduction versus the current shuffle sequence;
- FMA versus separate multiply/add;
- width-tail paths for 128, 256, 512 and 1024;
- synchronization placement only when a full-mask dependency proof exists.

Then route the best admitted operator candidates through the six explicit
network workloads. Compare full training state, raw and clipped gradients,
optimizer moments, inference graph output and host/event timing. Shared Torch,
cuBLAS, and native OA reference groups must remain distinct.

**Promotion rule.** Promote only a shape/width dispatch entry with complete
forward and backward state checks. A standalone operator win is insufficient if
the composed optimizer trajectory diverges.

## Cross-iteration safeguards

Every candidate needs exact source/cubin hashes, a complete output reference,
canaries, input immutability, disassembly and resource reports. Timing uses
paired shuffled samples with fixed holdout methods; NCU runs are diagnostic and
retain clock/cache metadata. Failed candidates remain in `exp/` with their
failure reason. No candidate is merged into the production compiler until the
experiment has passed its numerical contract and a regression suite.
