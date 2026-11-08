# A100 tensor dependency scheduler

Status: untested-or-hardware-unavailable

Status applies to the stated experiment and contract, not to every possible
version of the idea.

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

## A100 probe sequence

1. Measure dependent HMMA latency with one accumulator group.
2. Measure independent initiation spacing with two and three groups.
3. Measure LDSM-to-HMMA delay and address-WAR reuse separately.
4. Measure async-copy visibility through commit and wait-group.
5. Measure HMMA-to-store, HMMA-to-scalar, barrier and control-flow observers.
6. Fit these results into the experiment model and validate held-out shapes.
7. Compare register pressure, occupancy, short/long scoreboards, shared-bank
   traffic and tensor-pipe utilization against the current schedule and ptxas.

## Promotion rule

The current HMMA8/drain15 schedule remains the correctness baseline. Promote an
opt-in policy only after it passes full output and canary checks, valid
disassembly, register/shared-memory reports, matched ptxas controls and
repeated A100 timing on both seen and held-out shapes. A shorter instruction
listing by itself is not evidence of a faster or safer schedule.

