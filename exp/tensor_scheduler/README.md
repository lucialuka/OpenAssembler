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

