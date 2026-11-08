# A100 tensor dependency probe report

## Run identity

The full sweep was executed on Modal with an NVIDIA A100-SXM4-40GB:

* compute capability: 8.0;
* SM count: 108;
* reported SM clock: 1.410 GHz;
* driver: 13030;
* CUDA image: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`;
* toolkit compiler: `ptxas -arch=sm_80 -O3`;
* control sweep: S1 through S11, changing only SASS control bits 105:108;
* every accepted candidate was disassembled with `nvdisasm` before execution.

Raw timing/correctness record:

`hardware-probes_20261108T030653.043652Z.json`

The profiling attempt is recorded separately in the later timestamped JSON
files. Nsight Compute reported no kernels for the tiny ctypes-only worker. The
metric preflight itself succeeded, and the existing A100 counter harness
remains the reliable path for profiling larger native kernels.

The larger-kernel counter pass completed with the repository harness:

`exp/a100_counters/counter-validation_20261108T032554.579563Z.json`

## Correctness envelope

The table reports the lowest control stall that remained bitwise equal to the
reference output in this probe family. “S7+” means S7 through S11 passed and
S1 through S6 failed. A passing value is a local hardware observation, not a
portable rule for every register layout or kernel shape.

| Probe | Tensor structure | Lowest passing control | Observation |
|---|---:|---:|---|
| `dependent_8` | 8 HMMA on one accumulator group | S7 | Same-group RAW dependency is exposed. |
| `dependent_16` | 16 HMMA on one accumulator group | S7 | Same result across the longer chain. |
| `dependent_32` | 32 HMMA on one accumulator group | S7 | Longer chain does not lower the safe control. |
| `independent_2x16` | 32 HMMA, two accumulator groups | S7 | Two chains still do not hide the tested dependency distance. |
| `independent_3x16` | 48 HMMA, three accumulator groups | S1 | Three ready chains can cover the tested hazard in this layout. |
| `ldsm_reused` | One LDSM feeding 16 HMMA | S2 | LDSM completion/address protection is visible. |
| `ldsm_fresh` | LDSM before each HMMA | S1 | Repeated fresh loads provide enough intervening work here. |
| `async_ldsm_hmma` | `cp.async`/commit/wait/LDSM/HMMA | S1 | Async-copy/barrier protocol dominates this small test. |
| `observer_mov` | HMMA then scalar MOV/store | S7 | Scalar observer cannot consume the result too early. |
| `observer_ffma` | HMMA then FFMA/store | S7 | Arithmetic observer has the same boundary here. |
| `observer_store` | HMMA then global store | S7 | Store visibility needs the protected gap. |
| `observer_bar` | HMMA then CTA barrier/store | S7 | Barrier does not make an incomplete accumulator safe. |
| `observer_branch` | HMMA then predicated branch/store | S7 | Control-flow observer preserves the same boundary. |
| `address_war` | LDSM then immediate address rewrite | S2 | The tested address WAR needs at least one cycle. |

The strongest scheduling signal is the contrast between `independent_3x16`
and the dependent/two-chain probes. It supports a scheduler that distinguishes
dependent accumulator latency from independent tensor-pipe initiation. It does
not prove that all HMMAs can use S1: the same control is unsafe for the
single-chain and two-chain probes, and the observer distance remains separate.

## Timing observations

The tiny kernels measured approximately 4.4--4.8 microseconds for the dense
HMMA cases and about 5.5--5.7 microseconds for the async-copy case. Several
control variants were 1--5% faster than the ptxas control in these samples, but
the absolute duration is close to the launch/timing floor. These timings are
useful for selecting candidates for a larger looped probe, not for claiming an
end-to-end GEMM speedup.

## What this says about OpenAssembler

The current OA policy is too rigid for the opportunity exposed here. A single
HMMA gap and a post-chain drain are safe, but they leave performance on the
table when independent accumulator chains or ready scalar work can cover the
real dependency distance. The correct relaxation is conditional:

1. retain the current HMMA8/drain15 schedule as the correctness fallback;
2. schedule multiple independent HMMA chains with a measured issue-resource
   limit;
3. keep same-accumulator RAW, LDSM address-WAR and async-copy visibility edges;
4. delay stores/scalar/control observers until their measured edge distance;
5. use independent ready work as fillers before considering a drain removal;
6. reject candidates that increase register pressure enough to reduce active
   warps or fail full outputs.

This is the sweet spot between integrity and speed: relax only an edge proven
independent by the graph and hardware probe, rather than globally lowering the
HMMA stall field.

The experiment-only candidate search in `candidate-search.json` shows the same
tradeoff on synthetic traces. With one, two or three accumulator chains, the
baseline policy takes 72 model cycles for eight HMMA steps plus a store. A
targeted observer delay of seven cycles plus an independent HMMA reservation of
one or two cycles takes 64 cycles for the same traces while preserving every
accumulator RAW edge and the store observer edge. This is a model result, not a
production timing claim, but it confirms that targeted drains and independent
chain interleaving can reduce modeled idle time without weakening true RAW
dependencies.

## Counter evidence from native GEMM controls

The counter harness profiled the same A100 with OpenAssembler, matched `ptxas`
and cuBLAS controls. Tensor instruction counts were equal at 134,217,728 in
each case, so the remaining differences are schedule and memory behavior.

| Case | Kernel | Tensor active | Total inst. | Short SB | Long SB | Wait | Shared-load conflicts | DRAM bytes | L2 hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tall m-fast FP16 | OA | 75.32% | 287.13M | 9.56% | 14.25% | 47.15% | 267k | 3.109 GB | 63.03% |
| tall m-fast FP16 | ptxas | 50.77% | 217.38M | 1.00% | 56.38% | 22.33% | 252k | 4.527 GB | 52.10% |
| tall m-fast FP16 | cuBLAS | 87.91% | 317.81M | 1.80% | 5.95% | 35.34% | 1.647M | 1.380 GB | 90.19% |
| tall Morton FP16 | OA | 80.05% | 287.57M | 10.09% | 10.11% | 50.55% | 160k | 1.076 GB | 84.99% |
| tall Morton FP16 | ptxas | 85.48% | 217.96M | 2.11% | 3.93% | 37.70% | 352k | 1.295 GB | 81.82% |
| tall Morton FP16 | cuBLAS | 87.64% | 317.81M | 1.80% | 6.39% | 35.23% | 1.634M | 1.383 GB | 90.07% |

BF16 repeats the pattern: OA has about 287M instructions and 134M tensor
instructions, ptxas about 217M and 134M, and cuBLAS about 242M and 134M. OA's
short-scoreboard and wait fractions remain much higher than ptxas/cuBLAS on the
Morton case. The m-fast OA timing win comes with lower DRAM traffic than ptxas,
while the Morton case still loses despite better locality because OA retains
substantially more non-tensor work and wait time.

These counters answer the original question directly: the current OA schedule
is conservative and expensive in its scalar/control envelope. The opportunity
is to remove artificial gaps and excess work only where independent tensor or
scalar instructions cover the measured dependency edges. A universal S1
rewrite would be unsafe; the exact-SASS probes show why.

## Non-tensor instruction candidate

The separate no-op-address experiment removed eight
`add.u32 %r30,%r30,0` operations from the OA GEMM PTX. All three tested FP16
shapes passed full output and bitwise ptxas checks, and OA static SASS fell from
1,600 to 1,592 instructions. The paired timings were unchanged or slightly
slower: 0.020992→0.020992 ms, 0.020787→0.020890 ms and
0.088576→0.088986 ms. This rejects isolated no-op removal as a speedup and
supports a broader address-CSE/liveness/scheduling pass that changes overlap,
not only instruction count.

## Counter limitation and next probe

The A100 Nsight Compute metric preflight accepted tensor utilization,
instruction counts, short/long scoreboard, wait/barrier, shared-bank, DRAM and
L2 metric names. The tiny direct-driver worker nevertheless produced “No
kernels were profiled,” so no counter values are claimed for this run. The
next counter pass should use the existing runtime-launch path that has already
captured native GEMM profiles, or a CUDA-runtime wrapper around these cubins.

The next exact-SASS extension is a clocked loop probe with enough HMMA work to
move away from the launch floor. It should report `clock64()` deltas for one,
two and three accumulator chains, then pair those results with the existing
large-GEMM Nsight profiles.

That clocked probe has now completed on the A100. The raw record is
`exp/tensor_scheduler/clocked-loop-probes_20261108T050653.662428Z.json`. After
subtracting the empty-loop control, the measured 128-iteration results were:

| Probe | Total cycles | Cycles per HMMA |
|---|---:|---:|
| One dependent accumulator chain | 3,075 | 24.023 |
| Two independent accumulator chains | 3,075 | 12.012 |
| Three independent accumulator chains | 3,075 | 8.008 |

This is the first direct cycle evidence separating dependent accumulator
latency from independent initiation. The approximate 24-cycle dependent result
is consistent with the retained HMMA-to-observer machine-model distances; the
three-chain result reaches about an 8-cycle aggregate initiation interval. It
does not authorize S1: the control-field correctness sweep still found S1
unsafe for the single-chain and two-chain layouts.

## Clocked loop result

That extension now runs in
`exp/tensor_scheduler/clocked_loop_probes.py`. It reads `%clock64` inside a
128-iteration loop, stores one delta per block, and subtracts an empty-loop
control. The final A100 record is
`clocked-loop-probes_20261108T050653.662428Z.json`.

| Loop | Total cycles after empty control | Cycles per HMMA |
|---|---:|---:|
| One dependent accumulator chain | 3,075 | 24.02 |
| Two independent accumulator chains | 3,075 | 12.01 |
| Three independent accumulator chains | 3,075 | 8.01 |

This is the cleanest measurement so far of the distinction the production
scheduler needs. A dependent HMMA result is not readable after the same short
interval as an independent tensor-pipe reservation. Three chains cover the
approximately 24-cycle dependent result latency at an approximately 8-cycle
per-HMMA initiation rate. These values are probe-specific and still need to be
validated across fragment layouts, observers and real GEMM mainloops, but they
justify separate scheduler controls rather than one universal HMMA gap.

The first clocked implementation used an 8-byte block stride while writing a
32-byte record and triggered an A100 misaligned-address XID. That run was
discarded; the retained run uses a 32-byte stride and completed without a GPU
fault. The failed layout remains documented here so it is not reused.

