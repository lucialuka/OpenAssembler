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

The checks are model-level only. They check explicit accumulator, operand and
observer constraints; they do not establish a faster or generally safe A100
schedule. The default distances and the historical issue1/issue2 search are
synthetic hypotheses, not measured hardware control policies.

An explicit partially calibrated model can be selected with:

```bash
PYTHONPATH=exp/tensor_scheduler .venv/bin/python exp/tensor_scheduler/candidate_search.py \
  --profile a100_clocked_24_8
PYTHONPATH=exp/tensor_scheduler .venv/bin/python exp/tensor_scheduler/actual_block_schedule.py \
  --profile a100_clocked_24_8 --out exp/tensor_scheduler/actual-block-audit_20261109.json
```

`latency_profiles.py` records the exact retained clocked-evidence hash and uses
approximate dependent-chain24/three-chain issue8 distances. LDSM, async-copy,
observer, address-WAR and synchronization distances remain uncalibrated. These
are static model estimates, not measured kernel cycles or minimum safe SASS
stall fields. Neither profile authorizes encoding a reordered stream.

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
The current measured minima are summarized in the [scheduler dossier](../../docs/research/ideas/a100-tensor-dependency-scheduler.md#exact-sass-probe-envelope).
The NCU preflight accepts the requested metric names, but the tiny direct-driver
worker has not yet produced a profiled kernel; those counter values remain open.

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

`lowered_adapter.py` accepts only explicitly described register-only windows.
It rejects labels, branches, predication, implicit predicate/carry state,
async-copy/wait/barrier operations, general memory and unsupported scalar
forms. LDSM additionally needs an explicit caller-supplied shared-visibility
context. Complete HMMA fragment/accumulator register metadata is checked;
scalar readers of tensor outputs receive observer edges, while scalar
overwrites of unfinished tensor results are rejected. It does not encode SASS
or replace the production tensor backend.

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

## Full control inventory

The retained scalar cubins were replayed on an A100 with the same output
comparison used by the earlier latency campaign:

```bash
modal run tools/gpu_modal.py --which dir:corpus/sm_80/latsweep \
  --trials 2 --budget 500 --suffix full_20261108
python3.13 exp/tensor_scheduler/control_inventory.py
```

The source-inclusive record is `control-sweep-a100_20261108.json`; its
compressed worker payload is `control-sweep-a100_20261108.raw.json.gz`. The
generated `control-inventory.json` covers all 42 scalar dependency families
exposed by `tools/lat_sweep.py` plus the 14 tensor probes. The broader
`tools/ptx_probes.py` corpus contains 541 standalone SM80 PTX forms; those are
recorded as inventory-only because a standalone form does not define a
producer/consumer control edge. The canonical
[scheduler dossier](../../docs/research/ideas/a100-tensor-dependency-scheduler.md#exact-sass-probe-envelope)
records the measured minima and the limits of those pair-specific controls.

## CUTLASS-style fragment schedule

`load_next_compute_current.py` is an experiment-only source transformer based
on the public CUTLASS SM80 multistage pattern. It alternates two fragment
register windows, loads the next K fragment between current HMMA halves, and
checks every accumulator's K-order. The source-inclusive Modal runner is:

```bash
modal run exp/tensor_scheduler/load_next_compute_current_modal.py \
  --rounds 2 --samples 8
```

The retained six-shape run passed full output, canary, poisoned-graph,
disassembly and resource checks. It improved OA timing on five non-square
shapes and regressed the 4096×4096×4096 square case. The checked production
generator now includes the admitted exact-shape option; see the
[load-next dossier](../../docs/research/ideas/cutlass-load-next-compute-current.md)
for current integration and dispatch scope. This older experiment driver is a
control, not the current compiler policy.

## Actual lowered blocks

`actual_block_schedule.py` audits real lowered OA GEMM regions. Current regions
contain memory, visibility or implicit state outside the adapter contract and
are rejected with explicit reasons, including HMMA-bearing regions. Earlier
`actual-block-schedule.json` acceptance was a model-level permutation check
with omitted state and is superseded by the fail-closed audit. No reordered
stream is encoded or promoted. A recorded `accepted` field means only that a
bounded window fits the static adapter, never hardware validation.

