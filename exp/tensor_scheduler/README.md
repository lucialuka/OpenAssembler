# Experiment: four-edge A100 tensor dependency scheduler

This directory contains an experiment-only scheduler model inspired by the
issue-rate separation discussed in [`gpu-unlock`](https://github.com/Zagitta/gpu-unlock).
It is intentionally isolated from `openassembler/backend/tensor.py` until it
has passed A100 SASS probes, full-output checks, disassembly checks and paired
timing.

## What this tests

The production tensor path currently uses a fixed HMMA stall of 8 cycles and a
15-cycle drain at the end of each HMMA run. Those values are useful as a
correctness baseline, but they conflate four different controls:

1. dependent accumulator latency;
2. independent HMMA initiation spacing;
3. operand readiness from `cp.async`, `wait_group`, LDSM and shared-memory
   visibility; and
4. observer dependencies from stores, scalar consumers, barriers and control
   flow.

`dependency_scheduler.py` builds a small dependency graph with separate edge
classes and list-schedules multiple independent accumulator chains. The graph
does not assume that every HMMA can issue at S1, and it never removes a known
dependency edge just to shorten the instruction listing.

## Run the local checks

From the repository root:

```bash
PYTHONPATH=exp/tensor_scheduler pytest -q exp/tensor_scheduler/test_dependency_scheduler.py
```

The checks are model-level only. They prove that the scheduler keeps the
accumulator, operand and observer constraints distinct. They do not prove a
faster A100 kernel.

## A100 validation gate

Before any production integration, generate tiny SASS probes for:

* a dependent HMMA chain reusing one accumulator group;
* two and three independent HMMA chains using disjoint accumulator groups;
* LDSM-to-HMMA with changing shared addresses;
* `cp.async` -> commit -> wait-group -> LDSM;
* HMMA -> scalar consumer, store, barrier and control-flow edges.

For every candidate, require:

* complete output, canaries and bitwise/tolerance checks;
* valid `nvdisasm` output and unchanged opcode counts;
* register, shared-memory and occupancy reports;
* matched ptxas control kernels;
* Nsight Compute counters for tensor pipe utilization, short/long
  scoreboard, barrier waits, shared-bank conflicts and issue rate; and
* repeated paired timings on the same A100.

The first production-safe candidate should be an opt-in schedule policy. The
current source-order HMMA8/drain15 path remains the fallback until the new
policy wins held-out shapes without numerical or disassembly regressions.

## Exact-SASS Modal campaign

The ordered hardware probes are generated and run by
`hardware_probes.py`:

```bash
modal run exp/tensor_scheduler/hardware_probes.py --mode preflight
modal run exp/tensor_scheduler/hardware_probes.py --mode run --budget 700
modal run exp/tensor_scheduler/hardware_probes.py --mode run --budget 900 --profile
```

The campaign sweeps one, two and three accumulator-chain arrangements, LDSM
reuse/freshness, async-copy visibility, HMMA observers and address WAR. It
changes only SASS control bits after CUDA `ptxas` compilation, verifies each
variant with `nvdisasm`, checks full output equality, and records paired timing.
The current measured result is summarized in `REPORT.md`. The NCU preflight
accepts the requested metric names, but the tiny direct-driver worker has not
yet produced a profiled kernel; those counter values remain open.

## Candidate policy search

The graph model can rank bounded policies without touching production code:

```bash
PYTHONPATH=exp/tensor_scheduler python3 exp/tensor_scheduler/candidate_search.py
```

The search varies independent HMMA initiation spacing, targeted observer delay,
LDSM readiness and address-WAR distance. It reports edge counts alongside model
cycles so a lower-cost candidate that silently drops a proven dependency is
visible and rejectable. These are compiler-model results; promotion still
requires exact SASS, full outputs and A100 counters.

## Lowered-instruction adapter

`lowered_adapter.py` maps one lowered OA basic block into the graph model and
returns a candidate source-index order. It rejects branches, returns and
reconvergence markers rather than moving across control-flow boundaries. This
is the first opt-in bridge toward virtual-register scheduling; it does not
encode SASS or replace the production tensor backend.

## Clocked loop probe

The launch-floor limitation is addressed by
`clocked_loop_probes.py`:

```bash
modal run exp/tensor_scheduler/clocked_loop_probes.py --mode run
```

The kernel reads `%clock64` around a 128-iteration loop and subtracts an empty
loop control. The recorded A100 result is approximately 24 cycles per HMMA for
one dependent accumulator chain, 12 cycles per HMMA for two chains and 8 cycles
per HMMA for three chains. This is the measured separation between dependent
accumulator latency and independent initiation rate that the opt-in scheduler
must preserve.

