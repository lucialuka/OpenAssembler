# FP16 pack reuse, layout overhead and bounded tails

The measured implementation is promoted as an explicit `tensor_packing="dual"` policy. `separate` remains the default. All GPU arithmetic, conversion, transpose, crop, loss, gradient and update kernels are OpenAssembler-owned cubins; no vendor computation is an execution fallback.

## Findings and implementation

The original recipes launch two operand packs for every forward, dX and dW product. The new schedule converts each logical operand once per forward/backward iteration, produces normal and transposed FP16 layouts together, and shares fixed packed views. A 32×32 shared-memory tile with stride 33 makes both output layouts coalesced. Every forward clears pack validity, so changed inputs and SGD-updated weights are refreshed even in repeated CUDA Graph execution. The schedule never reuses packs across training iterations.

Odd matrix dimensions use zero-padded operands, the existing OA Tensor Core GEMM, and a guarded FP32 crop with optional bias. The public option is bounded to logical linear dimensions at most 512. Rank-three linear inputs follow existing contiguous flatten/view lowering. FP16 batched attention products and broadcasting remain unsupported; BMM retains the explicit FP32 contract.

## Complete training measurements

Both production campaigns use NVIDIA A100-SXM4-40GB, PyTorch 2.7.1+cu128 and CUDA 12.8. Each sample measures four complete MSE/backward/SGD updates in a CUDA Graph with resident inputs, 15 shuffled samples per round and three rounds. GPU event times exclude compilation, CPU capture, allocation and transfers. Reported speedup is baseline time / dual-pack time. Profiler traces are diagnostic; their durations are not added to manufacture an end-to-end total.

| Workload | Input, hidden | Baseline µs/step (three round medians) | Dual µs/step | Speedup by round | Launches |
|---|---|---|---|---|---|
| mlp128 | [128, 128], hidden 256 | 235.06, 234.96, 234.95 | 234.06, 233.79, 234.04 | 1.0042×, 1.0050×, 1.0039× | 32→26 |
| qkv128 | [128, 128], hidden 128 | 318.65, 318.22, 317.84 | 293.23, 293.29, 292.95 | 1.0867×, 1.0850×, 1.0850× | 45→32 |
| mlp256 | [256, 128], hidden 256 | 3833.49, 3832.28, 3828.22 | 3851.75, 3851.07, 3847.70 | 0.9953×, 0.9951×, 0.9949× | 32→26 |
| batch | [2, 64, 128], hidden 256 | 2034.10, 1982.02, 1984.31 | 1984.59, 1982.61, 1983.23 | 1.0249×, 0.9997×, 1.0005× | 32→26 |

Independent production repeat for the three-projection model: 1.0951×, 1.0944×, 1.0946×. This reproduces the useful complete-training improvement. MLP gains do not reproduce strongly, and the larger-batch MLP is slightly slower, so the policy is not enabled automatically.

For the three-projection model, pack launches fall from 18 to 5 and complete-step launches from 45 to 32. Private guarded allocations grow from 723,012 to 854,052 bytes (about 18.1%), reflecting the second packed layout. This is a launch/layout optimization, not an improvement to the GEMM instruction schedule.

## Packing and matrix trace breakdown

These are sums of raw kernel activities from one separately profiled complete Q/K/V training invocation (µs). They diagnose where time is spent; they are not paired steady-state distributions and are not the end-to-end latency table above.

| Region | Separate packs | Dual-layout reuse |
|---|---:|---:|
| Packing / transpose | 58.49 | 18.40 |
| Forward GEMMs | 41.85 | 41.82 |
| dX GEMMs | 41.85 | 41.89 |
| dW GEMMs | 41.86 | 41.92 |
| Bias / crop | 9.12 | 8.96 |
| Bias gradient reductions | 92.64 | 98.17 |
| MSE | 23.55 | 24.03 |
| SGD | 14.59 | 14.56 |

Pack reuse removes conversion/layout work while GEMM kernel times stay effectively unchanged. Serial bias gradients remain expensive. The 256-row MLP and flattened rank-three case additionally expose slow MSE shapes outside the existing 128×128 parallel-loss contract; that is why fewer packing launches barely move their totals.

## Numerical and ownership gates

Each case passes four changed-input direct training iterations against CPUReference, including prediction, loss, every dX/dW, updated weights, immutable inputs/targets and allocation canaries. Three additional actual graph replays change inputs/targets and poison workspaces before replay. Profiler kernel names/counts match the runtime ledger exactly. The four cases with an existing separate-pack route also match that schedule bitwise; held-out wider shapes retain the declared tolerance without a bitwise guarantee. The fixed numerical gate is rtol=0.002, atol=0.00002.

Exact pack stress passes 12 shape/pattern combinations, covering random FP32 bit patterns and signed zeros, subnormals, half rounding ties, overflow, infinities and signed canonical NaNs. Both padded layouts, zero fill, input preservation and guards are checked byte-for-byte. Training arithmetic is restricted to finite operands that remain finite after FP16 conversion.

The configured 17×65 repository passes worker compilation and verification in both direct execution and the public `TrainingReplay` path. Both report zero vendor compute launches and zero GPU code generation during replay. The same plan and runtime code ship through Modal and SSH; an SSH GPU was not separately benchmarked in this campaign.

## New matrix coverage

| Complete MLP training input | Hidden | Existing separate FP16 route | Dual route |
|---|---:|---|---|
| [17, 65] | 33 | rejects shape | full direct training and captured replay pass |
| [129, 130] | 127 | rejects shape | full direct training and captured replay pass |
| [2, 7, 65] | 33 | rejects shape | full direct training and captured replay pass |

Held-out input [1,129] with hidden 65 also passes. Wider [128,256] with hidden 512 passes direct and captured training; it is near-neutral in performance. These extend shape coverage rather than demonstrating speedups over an existing unsupported route.

## Rejected results and limits

The first implementation scattered transposed writes and failed to improve the MLP despite fewer launches; it remains in `first_candidate.py` and the retained failed/partial runs. A profiler-trace issue required fresh worker subprocesses per case. One early runner envelope incorrectly inherited a child success after a parent failure; that run has returncode=1 and is excluded. The corrected runner requires exit status zero.

The 1024-wide stress model fails a captured-training gradient gate (888 of 1,048,576 elements in the failing tensor). It is not an admitted contract, its speed is not reported, and the production dual-pack option rejects dimensions above 512. The failure does not establish which compiler/runtime component caused it; further isolation is needed before extending this bound.

Packing does not eliminate repeated GEMMs in Q/K/V training: they have distinct weights and outputs. General common-subexpression elimination, GEMM fusion, native FP16 BMM/attention, and faster loss reductions for other output sizes are outside this patch.

## Usage and evidence

```toml
[model]
entrypoint = "model.py"
input_shapes = [[17,65]]
math = "f16_rne_f32_accum"
tensor_packing = "dual"
mode = "training"
execution = "cuda_graph"
reuse_storage = true
```

Python: `compile_model(model, examples, out, math_mode="f16_rne_f32_accum", tensor_packing="dual")`.

- [Production seven-case training campaign](runs/production-20261114T032106Z/result.json)
- [Independent production three-projection repeat](runs/production-repeat-20261114T033422Z/result.json)
- [Exact conversion/padding stress](runs/probe-20261114T031654Z/result.json)
- [Direct/public-replay repository bridge](runs/bridge-20261114T032828Z/result.json)
- [Bounded single-row/512-wide held-out repeat](runs/production-heldout-20261114T033558Z/result.json)
- [Excluded 1024-wide stress and retained failure](runs/production-heldout-20261114T033054Z/artifacts.zip)
- [Reproducer and experiment design](README.md)

Every run directory retains its experiment source, runtime source hashes, full samples and an artifact archive containing OA PTX/cubins and ownership ledgers. Historical sources are frozen; current scripts may include later validation improvements.
