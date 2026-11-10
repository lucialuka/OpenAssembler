# A100 tensor dependency scheduler

Status: validated-not-integrated

Hardware probes validate bounded dependency measurements, including dependent
HMMA latency, independent-chain initiation and selected operand/observer
hazards. The generalized four-edge schedule has not been validated end to end
or promoted. This status applies to the measured methods, not every schedule
the experiment model could produce.

## Scope

OpenAssembler's A100 tensor path is correct on its validated contract, but its
schedule currently collapses several different hardware dependencies into one
fixed HMMA issue gap and one post-chain drain. This note records the next
research direction and the experiment-only implementation in
`exp/tensor_scheduler/`.

## What the linked project does, and does not do

[`gpu-unlock`](https://github.com/Zagitta/gpu-unlock) changes privileged
Ampere speed-selector registers before the NVIDIA driver binds a supported
GeForce card. Its documented SS0/SS1 writes remove a consumer-card tensor-rate
cap. That mechanism is firmware and product configuration, not a compiler
scheduler. A100 is already the datacenter GA100 product with its normal tensor
throughput exposed, so the project should not be ported into OpenAssembler or
treated as evidence of a hidden A100 rate switch.

The transferable lesson is methodological: measure the hardware rate ceiling
first, then distinguish a hardware cap from a compiler dependency schedule.
For A100, record clocks, power/thermal state, tensor instruction counts and
issue rate before attributing a gap to SASS.

## Four dependency classes

### Dependent accumulator latency

When an HMMA reads an accumulator group written by an earlier HMMA, the second
instruction must wait until the result is readable. This is a true RAW edge and
must be modeled per accumulator group.

### Independent HMMA initiation rate

Two HMMA instructions using disjoint accumulators and ready operands may be
able to issue closer together than the dependent-chain latency. This is a
tensor-pipe reservation constraint, not a register RAW edge. It should be
represented as a resource limit in the scheduler so independent chains can be
interleaved.

### Operand readiness

An HMMA can only issue when its fragments are ready. This includes LDSM
completion, shared-memory visibility, `cp.async` commit/wait-group state and
the address WAR created when a shared-address register is reused. These edges
must remain explicit even when HMMA order changes.

### Observer dependencies

Stores, scalar instructions, address calculations, barriers and control-flow
edges may observe tensor results. A fixed drain after every source-order HMMA
run is a safe approximation, but a graph can instead delay only the actual
observers. Removing a drain is legal only when the graph proves that no
observer or register reuse can see an incomplete value.

## Current measured motivation

The retained A100 counter evidence shows equal Tensor Core instruction counts
but substantially more OpenAssembler non-tensor work. A wide direct GEMM had
approximately 4.37 million dynamic OpenAssembler instructions versus 3.23
million for `ptxas`. In 4096x513 backward reductions, OpenAssembler had about
22--45% more dynamic instructions and much larger long-scoreboard fractions
with only about 5% more DRAM traffic. The first hypothesis is therefore
instruction overhead and insufficient overlap, not an absent firmware unlock.

## Experimental implementation

`exp/tensor_scheduler/dependency_scheduler.py` contains a deterministic graph
model with separate edge kinds and a list scheduler. It is deliberately not
imported by production code. The local tests cover:

* independent HMMA-chain interleaving with dependent-chain latency;
* separate async-copy, wait-group, LDSM and observer distances; and
* source-order retention when no independent work is available.

Run:

```bash
PYTHONPATH=exp/tensor_scheduler pytest -q exp/tensor_scheduler/test_dependency_scheduler.py
```

The first Modal exact-SASS sweep is retained in
[`hardware-probes_20261108T030653.043652Z.json`](../../../exp/tensor_scheduler/hardware-probes_20261108T030653.043652Z.json). It swept
S1--S11 on an A100-SXM4-40GB after compiling each probe with CUDA 12.8
`ptxas`. Dependent one-group HMMA chains and observer cases were incorrect
below S7 in this probe family, while three independent accumulator chains were
correct at S1. LDSM reuse required S2, repeated fresh LDSM and the tested
async-copy path passed at S1. These are conditional observations, not a
universal HMMA issue rule.

## A100 probe sequence

1. Measure dependent HMMA latency with one accumulator group.
2. Measure independent initiation spacing with two and three groups.
3. Measure LDSM-to-HMMA delay and address-WAR reuse separately.
4. Measure async-copy visibility through commit and wait-group.
5. Measure HMMA-to-store, HMMA-to-scalar, barrier and control-flow observers.
6. Fit these results into the experiment model and validate held-out shapes.
7. Compare register pressure, occupancy, short/long scoreboards, shared-bank
   traffic and tensor-pipe utilization against the current schedule and ptxas.

The metric preflight succeeded for the requested Nsight Compute counters, but
the tiny ctypes-only worker returned “No kernels were profiled.” Therefore no
per-probe counter value is claimed yet; the next pass must use a runtime-launch
wrapper or the existing native-GEMM counter harness.

The existing native-GEMM harness then completed a matched A100 counter pass
(`exp/a100_counters/counter-validation_20261108T032554.579563Z.json`). In the
tall FP16 controls, OA and ptxas executed the same 134,217,728 tensor
instructions, while OA emitted 287.13M total instructions versus ptxas's
217.38M. OA's m-fast run had 9.56% short-scoreboard and 14.25% long-scoreboard
stalls; the Morton run had 10.09% and 10.11%. ptxas was 1.00%/56.38% on
m-fast and 2.11%/3.93% on Morton. This is direct evidence that the remaining
gap is in non-tensor instruction work and dependency overlap, not missing
Tensor Core instructions.

The follow-up `clock64()` loop probe now measures the tensor timing classes
directly on the same A100: one dependent accumulator chain costs about 24.023
cycles per HMMA after empty-loop subtraction, two independent chains cost about
12.012 cycles per HMMA, and three independent chains cost about 8.008 cycles
per HMMA. The record is
`exp/tensor_scheduler/clocked-loop-probes_20261108T050653.662428Z.json`.
This supports separate dependent latency and independent initiation modeling,
while the earlier correctness sweep still forbids a universal S1 policy.

The follow-up scalar replay covers all 42 producer/consumer families exposed by
`tools/lat_sweep.py` and is merged with the 14 tensor probes in
`exp/tensor_scheduler/control-inventory.json`. Fixed-pipe minima are mostly
S4/S5, predicate-to-guard probes are S13, and the tested store/address cases
are S1/S2. These are pair-specific observations, not global scheduler
constants.

The [November 8 exact-SASS sweep](../../../exp/tensor_scheduler/hardware-probes_20261108T030653.043652Z.json)
and [clocked-loop probes](../../../exp/tensor_scheduler/clocked-loop-probes_20261108T050653.662428Z.json)
validate component measurements, but they do not validate the graph scheduler
on a reordered GEMM stream. The separate [November 9 load-next replay](../../../exp/tensor_scheduler/load-next-compute-current_20261109T041912.552241Z.json)
passes full-output, canary, poisoned-graph and bitwise-ptxas checks for six
FP16 shapes. It tests a particular two-window mainloop schedule, not the
general four-edge dependency model; no scheduler-produced order has been
encoded or promoted.

`exp/tensor_scheduler/actual_block_schedule.py` lowers real OA GEMM streams
through the selector and allocator, splits them at barriers/control edges,
and audits each region against the experiment adapter's explicit contract.
The earlier accepted-region result is superseded by the November 9
[strict dependency audit](../../../exp/tensor_scheduler/actual-block-audit_20261109.json).
The adapter now rejects unmodeled predication, implicit registers, control,
async/wait/barrier visibility, general memory and unsafe tensor overwrites.
All 45 audited actual GEMM regions are rejected; their earlier permutation
acceptance did not prove semantic safety. Supported scalar consumers receive
observer edges, but incomplete token modeling still blocks production use.
The `a100_clocked_24_8` model profile now identifies its measurement artifact
and hash; 24/8 are approximate model distances fitted to the bounded clocked
probes, not universal controls. Other edge distances remain uncalibrated.

Separately, [split-2/4/6 load-next placements](../../../exp/shape_coverage/a100-role-gemm_20261109T111245.728446Z.json)
pass full FP16/BF16 output/graph/control checks at 1024×4096×1024 while retaining
HMMA8/drain15 and K order. Split-4 remains better than split-2/6 on that role.
This validates particular source schedules, not the general graph scheduler.

## Promotion rule

The current HMMA8/drain15 schedule remains the correctness baseline. Promote an
opt-in policy only after it passes full output and canary checks, valid
disassembly, register/shared-memory reports, matched ptxas controls and
repeated A100 timing on both seen and held-out shapes. A shorter instruction
listing by itself is not evidence of a faster or safer schedule.

## Exact-SASS probe envelope

These minima describe the retained S1–S11 correctness sweep, not general
instruction latency or an authorization to shorten every HMMA control:

| Probe family | Lowest passing control | Tested context |
|---|---:|---|
| One dependent group, 8/16/32 HMMAs | S7 | Same accumulator RAW chain |
| Two groups, 16 updates each | S7 | Two chains do not cover the dependency in this layout |
| Three groups, 16 updates each | S1 | Three ready independent chains |
| Reused LDSM fragment | S2 | One LDSM feeds 16 HMMAs |
| Fresh LDSM fragment | S1 | A new load precedes each HMMA |
| Async-copy/wait/LDSM/HMMA | S1 | Explicit async-copy and visibility protocol retained |
| MOV, FFMA, store, barrier or branch observers | S7 | Tested consumers after HMMA |
| LDSM address WAR | S2 | Immediate rewrite of the shared address register |

The corresponding tiny kernels take roughly 4.4–5.7 microseconds, close to the
launch-timing floor. Those samples do not establish a GEMM speedup. The clocked
loops above supply separate cycle measurements. The first clocked implementation
used an eight-byte block stride for a 32-byte record and caused a misaligned
address fault; that run was discarded. Retained results use the corrected
32-byte stride. The experiment graph's synthetic cycle totals remain model
outputs, not hardware timings.

