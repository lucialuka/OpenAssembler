# OpenAssembler A100 ML workload comparison

Generated from five frozen end-to-end workloads. This is an early, shape-bounded result; it does not establish parity across whole repositories or general ML training.

## Benchmark contract

All measurements use a Modal NVIDIA A100-SXM4-40GB (SM80), CUDA 12.8, PyTorch 2.7.1 and Triton 3.3.1. The tasks use fixed seeds, identical shapes and FP32 or explicit FP16-RNE operand/FP32-accumulation contracts. Each arm passes four changed-input full-state checks against CPUReference, including loss, all dX/dW, updated weights, canaries and immutable inputs. The OA profiler audit has exact kernel ownership and no unknown compute launches. The training suites capture four consecutive updates and report 15 shuffled event samples in each of three rounds. Compilation, warmup and graph-capture costs are reported separately. Resident graph timings exclude H2D and CPU delivery; the archived JSON also retains host-call, H2D, delivery, per-kernel trace and sample data. Component timings are paired independently and must not be summed to reconstruct totals.

“OA PTX + ptxas” compiles the exact OA-emitted PTX/recipe inputs with NVIDIA ptxas. It isolates the code compiler for a matched operation stream. “PyTorch CUDA ops” is a manual eager framework-operator schedule, with its reverse graph specified explicitly and matrix products dispatched through installed NVIDIA libraries. It is not stock module/autograd execution, a full nvcc CUDA C++ model compile, or a direct cuBLAS algorithm-search comparison. The Triton arm uses direct Triton JIT kernels and does not invoke Inductor. None of these control artifacts enter the OpenAssembler production runtime.

## Whole-task resident CUDA Graph latency

| Workload | Contract | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) | OA speed vs Triton / ptxas / PyTorch ops |
|---|---|---:|---:|---:|---:|---|
| tiny_regression | fp32, training, 17×8 | 69.04 | 44.92 | 39.62 | 77.06 | 0.65× / 0.57× / 1.12× |
| dense_regression_fp32 | fp32, training, 128×128 | 225.95 | 98.73 | 91.43 | 114.74 | 0.44× / 0.40× / 0.51× |
| dense_regression_fp16 | f16_rne_f32_accum, training, 128×128 | 236.26 | 60.29 | 152.68 | 246.14 | 0.26× / 0.65× / 1.04× |
| attention_training | fp32, training, 2×7×8 | 219.37 | 172.64 | 151.85 | 324.46 | 0.79× / 0.69× / 1.48× |
| projection_inference | f16_rne_f32_accum, inference, 128×128 | 67.62 | 19.32 | 60.66 | 104.38 | 0.29× / 0.90× / 1.54× |

Across these five tasks, OpenAssembler is slower than Triton and same-PTX ptxas in every case. The PyTorch CUDA operator path is faster than OA for the two dense training workloads and projection inference, but slower for tiny regression and attention. These figures expose remaining matrix, launch and materialization overhead; they do not support a claim that OA surpasses the NVIDIA toolchain.

## Per-workload component medians

All values below are median of the three round medians in microseconds. Components use the same resident CUDA Graph method but are independently paired.

### tiny_regression (NVIDIA A100-SXM4-40GB; fp32; shape (17, 8))

| Region | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) |
|---|---:|---:|---:|---:|
| forward | 13.20 | 13.70 | 10.21 | 20.01 |
| loss and seed | 16.65 | 7.37 | 7.26 | 8.74 |
| backward | 35.10 | 20.61 | 18.27 | 44.88 |
| update | 8.96 | 8.76 | 8.82 | 8.74 |

### dense_regression_fp32 (NVIDIA A100-SXM4-40GB; fp32; shape (128, 128))

| Region | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) |
|---|---:|---:|---:|---:|
| forward | 41.18 | 34.71 | 20.82 | 27.52 |
| loss and seed | 20.10 | 7.46 | 10.80 | 10.59 |
| backward | 118.53 | 38.04 | 41.10 | 55.44 |
| update | 9.16 | 8.83 | 8.95 | 8.75 |

### dense_regression_fp16 (NVIDIA A100-SXM4-40GB; f16_rne_f32_accum; shape (128, 128))

| Region | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) |
|---|---:|---:|---:|---:|
| forward | 40.99 | 13.54 | 36.40 | 59.30 |
| loss and seed | 18.42 | 6.61 | 9.52 | 9.65 |
| backward | 117.86 | 24.06 | 69.03 | 120.44 |
| update | 8.17 | 7.79 | 7.97 | 8.07 |

### attention_training (NVIDIA A100-SXM4-40GB; fp32; shape (2, 7, 8))

| Region | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) |
|---|---:|---:|---:|---:|
| forward | 41.74 | 42.76 | 34.01 | 59.90 |
| loss and seed | 11.46 | 6.15 | 5.51 | 6.81 |
| backward | 102.62 | 72.06 | 64.32 | 171.86 |
| update | 19.87 | 20.11 | 20.55 | 20.54 |

### projection_inference (NVIDIA A100-SXM4-40GB; f16_rne_f32_accum; shape (128, 128))

| Region | OpenAssembler (µs) | OpenAI Triton (µs) | OA PTX + ptxas (µs) | PyTorch CUDA ops (µs) |
|---|---:|---:|---:|---:|
| forward | 67.76 | 19.27 | 60.64 | 104.44 |
| end-to-end inference | 67.62 | 19.32 | 60.66 | 104.38 |

## Setup and compilation

These are one worker-local sample per task in milliseconds; compile and warmup never enter the steady-state graph measurements. All five exact plans were cache misses, so OA build times show fresh bundle compilation and serialization. See archived result JSON for runtime initialization and one-step graph capture.

| Workload | OA bundle build (ms) | Triton load/compile/warm (ms) | Same-PTX ptxas load/build (ms) | PyTorch CUDA ops warmup (ms) |
|---|---:|---:|---:|---:|
| tiny_regression | 1103.1 | 4650.8 | 94.4 | 352.8 |
| dense_regression_fp32 | 1597.4 | 3422.0 | 147.7 | 262.4 |
| dense_regression_fp16 | 9809.4 | 2263.9 | 1336.9 | 292.1 |
| attention_training | 1891.9 | 6274.6 | 188.3 | 361.6 |
| projection_inference | 3799.8 | 1452.3 | 849.4 | 145.1 |

## Promoted OA experiments and cache evidence

The exact-shape 15-entry FP32 matrix table and deterministic 16,384-element parallel MSE reduction are now in the production OA emitter. The current campaign above tests them through the production build/runtime path. The parallel MSE uses a different reduction association, so its scalar loss is checked under the documented FP32 tolerance; outputs, gradients, updates and guards pass. Other element counts retain their previous reductions.

| Complete FP32 training graph, OA baseline vs promoted candidate | A100-SXM4-80GB speedup by paired round |
|---|---|
| tiny_regression | 1.071×, 1.073×, 1.073× |
| dense_regression_fp32 | 6.471×, 6.877×, 6.872× |
| attention_training | 1.065×, 1.065×, 1.064× |

Matrix and loss candidates pass complete CPUReference state checks and OA profiler ownership audits. The separate operator campaign validates the selected FP32 contracts on A100-SXM4-40GB and A100-SXM4-80GB.

The OA-only content-addressed bundle cache was tested on A100-SXM4-40GB. It keys generated source/toolchain/plan/build options, stores no weights or ptxas donor cubins, refreshes changed weights, validates every hit, and rebuilds a corrupted entry. Cold-vs-hit compilation/materialization results:

| Model | Cold build (ms), three pairs | Validated cache hit (ms), three pairs | Speedup range |
|---|---|---|---|
| native_model | 865.0, 964.2, 871.8 | 56.6, 60.2, 58.0 | 15.02–16.02× |
| native_tensor_model | 11788.3, 11783.6, 11723.6 | 844.8, 834.2, 830.5 | 13.95–14.13× |

## Experimental pack-reuse result (not promoted)

A separate OA-only complete-graph experiment on an NVIDIA A100-SXM4-80GB evaluated reusing FP16 operand packs for the tensor-core MLP and Q/K/V-style projection model. The full four-step update/inference outputs, gradients, updated weights, immutability and guards passed. Full training stayed near-neutral for the MLP; projection training improved about 1–1.5%, while projection inference improved about 7.0%. This remains under `exp/` because the training wins are small and the inference result has not yet been independently repeated.

| Model | Training graph speedup by round | Inference graph speedup by round |
|---|---|---|
| FP16-RNE tensor MLP | 1.001×, 1.001×, 1.001× | 1.000×, 1.002×, 1.001× |
| Three-projection model | 1.010×, 1.015×, 1.015× | 1.071×, 1.071×, 1.070× |

## Interpretation and next gap

The two production changes remove severe OA bottlenecks in the measured workloads: the 128×128 MSE was previously the serial single-thread bottleneck, and exact matrix contracts replace the baseline scalar-dot mapping. The remaining whole-task gap is substantial, especially FP16 tensor-model training and multi-projection inference, where the direct Triton path leads by 3.5–3.9×. The next work should target FP16 pack/GEMM reuse in complete training graphs, launch count and layout/conversion overhead, and broader batch/tail/matrix shapes. The small synthetic attention model is not a serving/KV-cache benchmark.

## Reproduction artifacts

- Cross-engine campaign and raw checks/traces: [`exp/ml_suite/runs/bench_campaign-20261112T195238Z/artifacts.zip`](exp/ml_suite/runs/bench_campaign-20261112T195238Z/artifacts.zip)
- Cross-engine result envelope and source hashes: [`exp/ml_suite/runs/bench_campaign-20261112T195238Z/result.json`](exp/ml_suite/runs/bench_campaign-20261112T195238Z/result.json)
- Complete graph matrix/loss admission run: [`exp/ml_suite/runs/matrix_graph_campaign-20261112T175817Z/artifacts.zip`](exp/ml_suite/runs/matrix_graph_campaign-20261112T175817Z/artifacts.zip)
- Pack-reuse experiment with full-state checks: [`exp/ml_suite/runs/packing_campaign-20261112T143536Z/artifacts.zip`](exp/ml_suite/runs/packing_campaign-20261112T143536Z/artifacts.zip)
- Cache repeat/corruption campaign: [`exp/ml_suite/runs/cache_campaign-20261112T194300Z/artifacts.zip`](exp/ml_suite/runs/cache_campaign-20261112T194300Z/artifacts.zip)
- Frozen workload definitions and benchmark methodology: [README.md](README.md)

Numerical thresholds, full samples, worker/library versions, raw profiler traces and admission ledgers are retained in the campaign archives.
