# OpenAssembler Whole-Repository Test and Performance Review

Test and GPU work was performed on a copied source snapshot in this review directory. The working repository was not modified. Snapshot verification reports the same HEAD, clean/dirty state, and hashes for all included files before and after the runs.

## Review result

This is a broad engineering and benchmark report, not a claim that OA is faster end to end. The constrained FP32 native-model route is correct on the measured contracts, but the two tested models run slower than both eager and Inductor steady-state execution. Per-operation same-PTX timings also show OA behind `ptxas` on most matrix and loss kernels. The explicit FP16-operand Tensor Core contract passes full state checks; complete packing/GEMM regions and direct cuBLAS GEMM controls also pass, with OA slower in these measured contracts. A matching whole-model eager/Inductor half-policy comparator remains not-run.

## Snapshot and environments

- Git revision: `e071448d3120dd952549b7d2fe4708f1496ebf99`; initial worktree dirty state: `clean`.
- Snapshot files hashed: 8598; post-run integrity: `unchanged`.
- CPU frontend: macOS-15.6.1-arm64-arm-64bit-Mach-O, Python 3.13.11, Torch 2.15.0, single CPU thread. This machine has no CUDA device, Modal/Triton Python packages in its local runtime, or running Colima.
- GPU: NVIDIA A100-SXM4-40GB, SM 8.0, 108 SMs; driver 610.57.04; CUDA toolkit 12.8; `nvcc`/`ptxas` V12.8.93; Python 3.12.1, Torch 2.7.1+cu128, Triton 3.3.1. Ratios are from this A100 environment only.

## Coverage and Frontend

The independent CPU matrix records 16 accepted contracts and 12 expected-rejected contracts. It includes linear rank-2/3, residual branches, reshape/transpose, concat, BMM, softmax, tanh GELU, LayerNorm, tails and the three shipped native-model examples. Rejections remain explicit: sigmoid, broadcasting, slicing/indexing, output reductions, exact GELU, dropout, division, data-dependent control flow, convolution, FP16 input, noncontiguous input and NaN input.

Across the CPU matrix, the harness captured FX plans, derived forward/backward nodes, ran four storage-view/reuse planning modes, checked aliases/lifetimes, and prepared unique PTX contracts. Repeated CPU stage samples are in `cpu-stages.json`; GPU worker capture/factory and planning records are in `remote-response.json`. CPU frontend time is not mixed with A100 time.

| Contract | Capture FX/lowering/autodiff (median ms) | Storage planning (median ms) | Unique scalar contracts | Unique tensor recipes |
|---|---:|---:|---:|---:|
| linear_(1, 1) | 0.782 | 0.148 | 18 | 0 |
| linear_(3, 5) | 0.732 | 0.150 | 18 | 0 |
| linear_(17, 8) | 0.751 | 0.153 | 18 | 0 |
| linear_(33, 7) | 0.732 | 0.145 | 18 | 0 |
| linear_(2, 3, 5) | 0.756 | 0.221 | 18 | 0 |
| residual_(3, 5) | 0.598 | 0.124 | 9 | 0 |
| reshape_(2, 3, 5) | 0.544 | 0.089 | 7 | 0 |
| concat_(2, 3, 5) | 0.609 | 0.117 | 11 | 0 |
| bmm_(2, 3, 5) | 0.493 | 0.094 | 10 | 0 |
| softmax_(2, 33) | 0.486 | 0.057 | 6 | 0 |
| softmax_(1, 513) | 0.494 | 0.057 | 6 | 0 |
| gelu_(3, 7) | 0.483 | 0.057 | 6 | 0 |
| layernorm_(2, 33) | 0.556 | 0.056 | 6 | 0 |
| native_model | 0.711 | 0.168 | 19 | 0 |
| native_attention_model | 1.741 | 1.161 | 43 | 0 |
| native_tensor_model | 1.042 | 0.177 | 13 | 6 |

Unsupported cases were not timed as slow cases. The capture matrix is representative coverage, not a proof over all supported PyTorch graphs, shapes or layouts.

CPU Torch 2.15.0 is outside the repository’s declared optional Torch 2.7 range. Those results are separately labelled. GPU capture/reference/comparator work and the remote toolkit regressions used the supported Torch 2.7.1 environment.

## A100 Model Results

All three model cases passed four changing-input training steps before steady-state measurement. Gates compared complete outputs, loss, input and parameter gradients, updated parameters, preserved inputs/targets, workspace canaries, and an independent profiler event multiset against the native launch ledger. Plain SGD has no optimizer state; the comparator uses the same no-momentum/no-weight-decay update. The Tensor Core case uses explicit FP16 RNE operands with FP32 accumulation/output and its separate tolerance contract.

| Model contract | Build first / warm (ms) | OA launches/step | Full-state checks | OA host-step median (ms) | Eager median (ms) | Inductor median (ms) | OA speed ratio vs eager | OA speed ratio vs Inductor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native_model (fp32, [17, 8]) | 1061 / 730 | 18 | 44 | 1.263 | 0.974 | 1.126 | 0.771x | 0.891x |
| native_attention_model (fp32, [2, 7, 8]) | 1587 / 1251 | 74 | 124 | 4.402 | 2.535 | 2.127 | 0.576x | 0.483x |
| native_tensor_model (f16_rne_f32_accum, [128, 128]) | 14126 / 13850 | 32 | 44 | 3.093 | not-run | not-run | not-run | not-run |

`ratio = comparator median / OA median`; >1 means OA was faster. The Tensor Core model’s framework comparators are `not-run`, not omitted results. The host-step path keeps tensors resident and includes host dispatch plus synchronization. Separate CUDA stream event medians are retained in the raw report.

The FP32 MLP reaches 0.77x eager and 0.89x Inductor; attention reaches 0.58x and 0.48x. These ratios mean the OA execution path is slower for these samples. A library/framework or compilation advantage on one operation cannot be generalized to the application.

## Same-PTX and Artifact Results

Each per-operation pair launches OA-produced code and `ptxas` output compiled from the exact same emitted PTX contract. Each kernel uses identical captured inputs, output shape, dtype and launch configuration; both arms are checked against the captured model outputs and the full composed workload passes its independent CPU-reference gate. CUDA graph replay averages 32 launches per sample over 25 shuffled samples. This controls compiler input, but does not replace a distinct per-op mathematical proof for every captured operation.

| Model | Kernel contracts | OA faster than same-PTX ptxas | Median OA speed ratio | Min–max OA ratio |
|---|---:|---:|---:|---:|
| native_model | 17 | 3 / 17 | 0.680x | 0.254–1.054x |
| native_attention_model | 41 | 5 / 41 | 0.714x | 0.274–1.029x |
| native_tensor_model | 25 | 2 / 25 | 0.977x | 0.144–1.017x |

The first two models have 17 and 41 per-operation contracts; the Tensor Core model has 25. OA is faster than same-PTX `ptxas` on few operations; ratios below one indicate OA slower. The report keeps individual raw samples and operation labels. Compiler launch ledgers match independently observed compute counts: 18, 74 and 32 in the profiled steps.

| Model | OA executable cubin | ptxas-reference cubin | Meaning |
|---|---:|---:|---|
| native_model | 99,328 bytes | 58,664 bytes | Full cubin container sizes, not executable text size |
| native_attention_model | 208,640 bytes | 114,856 bytes | Full cubin container sizes, not executable text size |
| native_tensor_model | 72,448 bytes | 45,736 bytes | Full cubin container sizes, not executable text size |

Model native cubins contain only OA-owned kernel entries; Tensor Core stage cubins and ptxas references are recorded separately. The source manifests include OA register counts, generated SASS listings, and PTX/cubin hashes. The final artifact pass independently retrieved 178 per-entry disassembly/resource rows from these saved cubins, exported to `artifact-resources.csv`; instruction counts include encoded padding. `kernel-summary.csv` adds loaded-function register/static shared-memory comparisons and launch dynamic shared-memory bytes for each timed kernel. Cubin size includes ELF metadata and is not executable-text size.

## Complete Tensor Regions and cuBLAS Controls

The saved native tensor artifacts were independently loaded on an A100-SXM4-40GB, Torch 2.7.1/CUDA 12.8, driver 610.57.04. Its GPU UUID differs from the model run; no absolute samples are pooled between sessions. All six regions passed full outputs on zero/integer/normal inputs, immutable input and workspace guards, exact OA/ptxas output-bit comparison, and profiler name/count checks for the OA region before timings. Each timing has 25 randomized samples, averaging 16 region invocations per graph.

| Region | Logical input shapes | OA complete region µs | ptxas complete region µs | OA ratio vs ptxas | OA GEMM-only ratio vs cuBLAS |
|---|---|---|---|---|---|
| linear | [[128, 128], [256, 128], [256]] | 16.41 | 13.99 | 0.853x | 0.482x |
| linear | [[128, 256], [128, 256], [128]] | 19.67 | 15.75 | 0.801x | 0.367x |
| linear_dx | [[128, 128], [128, 256]] | 14.82 | 12.43 | 0.839x | 0.499x |
| linear_dw | [[128, 128], [128, 256]] | 15.28 | 13.01 | 0.851x | 0.480x |
| linear_dx | [[128, 256], [256, 128]] | 18.34 | 14.60 | 0.796x | 0.371x |
| linear_dw | [[128, 256], [128, 128]] | 15.37 | 12.97 | 0.844x | 0.463x |

Complete regions include packing A/B, GEMM and optional bias. The cuBLAS control is the prepacked GEMM only: FP16 inputs, FP32 accumulation/output, alpha=1/beta=0, no workspace, cuBLAS GemmEx default tensor-op algorithm (99), math mode 16. No algorithm search or cuBLASLt tuning was performed. OA is slower in all complete regions (0.796–0.853x ptxas) and all matched GEMMs (0.367–0.499x cuBLAS). The two comparator scopes stay separate. `region-response.json` preserves every sample, order, shape and gate.

### Build and amortization

Build rows use fresh output directories. First-build and warm-build rows share one worker process, so “warm” means a second clean output in the same process, not a persistent cache hit. Timed functions separately record OA `generate`, exact-PTX `ptxas_build`, cubin splicing and `nvdisasm`; stage totals are in `review-summary.json`. Tensor build time is dominated by OA backend generation and disassembly. Source collection/archive packing was about 2.84 MB in one dispatch; queue, remote startup and transfer are available only as one shared dispatch interval, not an attributable per-model measurement.

| Model | Build | Bundle wall ms | OA generate ms | same-PTX ptxas ms | Disassembly ms | Validation ms |
|---|---|---|---|---|---|---|
| native_model | first | 1061.13 | 269.16 | 87.44 | 340.93 | 2.89 |
| native_model | warm | 729.52 | 241.99 | 92.29 | 354.91 | 3.10 |
| native_attention_model | first | 1586.67 | 635.96 | 144.27 | 369.59 | 8.72 |
| native_attention_model | warm | 1250.76 | 608.16 | 146.67 | 377.17 | 8.76 |
| native_tensor_model | first | 14126.20 | 6891.68 | 1148.15 | 5042.72 | 257.58 |
| native_tensor_model | warm | 13850.05 | 6759.46 | 1148.51 | 5129.46 | 257.90 |

Internal `splice_text` and `splice_kernels` calls are nested; their recorded durations must not be added twice. The table lists non-overlapping top-level phases only. First-build rows are process-warm to imports/coverage work and use fresh output directories, so they are not a filesystem-cold or new-container guarantee.

The requested accounting is `total_after_N_runs = build_time + startup_time + N * steady_state_time`. Per-case values are retained for N=1, 10, 100, 1,000 and 10,000. For FP32 contracts, Inductor’s first call includes compile plus execution; eager has no compile phase. OA steady state is slower than both comparators on these two cases, so its build cost does not amortize into an eager-performance win. Against Inductor, OA’s shorter compile can reduce total time at low run counts on attention, but steady-state remains slower and there is no universal crossover claim beyond the exact recorded arithmetic. The Tensor Core row has no framework arm and therefore no comparative amortization claim. **These are partial-path, lower-bound estimates**, using bundle build and Runtime.load; lazy module/workspace initialization, full frontend/import time, queue and deployment startup were not all isolated. They must not be cited as full-deployment amortization results.

Shared transport accounting: archive pack 118.59 ms, dispatch wall 116830.77 ms, remote body 86974.65 ms, and 29856.12 ms residual outside the body. The residual combines queue/startup/serialization/transfers and is not compiler time.

## Test Runs

| Suite | Result | Scope |
|---|---|---|
| Local production | 923 passed, 55 toolkit skips, 1 deselected | CPU, macOS; import-isolation failure kept separately |
| Isolated import guard | 1 passed | Fresh process; fails in full Torch-enabled collection |
| Linux toolkit regressions | 161 passed | CPU toolkit container; 43 local skips resolved, plus overlapping tests |
| Linux toolkit CLI | 57 passed | CPU toolkit container; remaining 12 local skips resolved |
| Independent reviewer tests | 67 passed | Capture/rejection, full four-step FP32 autograd/SGD, storage, report admission |
| Experiments + character Transformer | 268 passed | Current source and copied ignored binary fixtures |
| Feature inventory | 8 passed, 2 failed | Two stale-catalogue rule failures remain |
| New audit regressions | 2 failed | Desired rejection invariant demonstrably fails; production unchanged |
| Documentation | passed | 131 Markdown files, 985 links, 82 catalogue entries, 17 CLI parser levels |

All 55 local toolkit skips were matched by test identity to a remote passing test. The production identity union is 979 passed across environments. This does not erase the original collection-order failure or claim one clean monolithic run. Counts overlap between scopes and are not added. The rowwise Modal matrix tests also pass in the Linux toolkit run.

The first experiment attempt failed collection without Modal, and the initial snapshot omitted ignored binary evidence/Git metadata; these harness setup failures are retained in logs. Supplemental copies resolved them. A temporary directory inside the copied Git tree also changed two discovery tests; the final local run uses a sibling temp directory and passes both. The initial combined GPU run missed profiler activities in later cases, so those failed cases are excluded; fresh-process model runs passed. The expanded follow-up lost its Modal response and has no admitted results. The compact saved-artifact region run completed successfully.

## Findings

1. **[P2] Profiler audit can ignore an extra compute event.** `audit_profiler_events` drops any CUDA event whose name contains `memcpy` or `memset`. Two independent negative tests show an unexpected compute event with either substring passes the audit. This weakens the evidence that no vendor compute launch occurred.

2. **[P2] Feature inventory rejects the current catalogue.** `exp/dispatch_tuning/feature_inventory.py` has no rules for two catalogue entries marked implemented, so both inventory audit tests fail.

3. **[P2] One production test depends on collection order.** `test_gemm_tuner_imports_without_touching_torch` asserts Torch is absent from global `sys.modules`; it fails after other test modules import Torch, but passes in a fresh process.

4. **[P2] Native model report lacks compile-stage and kernel distributions.** The worker records synchronized host step samples and profiler audits, but its shipped report does not retain separated compiler phases or per-kernel GPU distributions. The separate review harness captured those here.
