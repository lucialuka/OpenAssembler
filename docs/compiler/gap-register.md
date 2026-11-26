# Compiler gap register

The supplied 51-gap review remains the scope of this program. Validated additions cover typed helpers/wide integers/buffers/atomics, native memory and parameter contracts, tensor operators, optimizer/checkpoint state, recorded builds and measurement. **The complete 51-gap program is not finished.**

November 26 integration passes 5,136 tests in Python 3.12.10/Torch 2.7.1 with 0 skips, and 4,651 tests with 15 skips in the separate Python 3.14.5 compiler-only environment. A100 evidence and exact source/native-code identities are retained in the linked dossiers and [validation record](../../compiler_exp/gap_closure/python_boolean_abi/storage/validation.json).

The [machine-readable register](../../compiler_exp/gap_closure/gap-register.json) retains every original exit gate. The [supplied review](../../compiler_exp/gap_closure/input-gap-register.txt) is retained unchanged, including its dated evidence and limits. A row marked partial requires the remaining original gate; it is not counted as a completed broad capability.

| Gap | Finding | Current state |
|---|---|---|
| G01 | Unresolved exact-division verification counterexample | [counterexample-resolved](gaps-verification.md) |
| G02 | Full semantic conformance is still a larger task than scalar admission | [partial](gaps-contracts.md) |
| G03 | Custom Python types and numerical semantics remain narrow | [partial](gaps-python-types.md) |
| G04 | Reusable Python functions and library calls are not generally compilable | [partial](gaps-python.md) |
| G05 | Python objects, containers and compile-time programming lack a general route | [partial](gaps-static-values.md) |
| G06 | Python control flow is useful but not unrestricted | [partial](gaps-python.md) |
| G07 | Custom-kernel arrays have no general tensor/view/layout API | [partial](gaps-buffers.md) |
| G08 | Python GPU primitive vocabulary is incomplete | [partial](gaps-python-primitives.md) |
| G09 | Memory safety and shared-memory initialization checks are incomplete | [partial](gaps-buffers.md) |
| G10 | PTX parameter ABI and module grammar are bounded | [partial](gaps-parameter-abi.md) |
| G11 | Scalar arithmetic/types/modifiers do not cover general CUDA-emitted PTX | [partial](gaps-wide.md) |
| G12 | Global/shared memory widths and vector forms remain uneven | [partial](gaps-memory.md) |
| G13 | Address spaces and module storage are not generally lowered | [partial](gaps-storage.md) |
| G14 | Atomics, reductions and memory ordering are missing compiler foundations | [partial](gaps-atomics.md) |
| G15 | Warp/block collective and synchronization support is narrow | [partial](gaps-collectives.md) |
| G16 | Device calls, call stacks and cross-module device linking are absent | [partial](gaps-device-functions.md) |
| G17 | Register/predicate spilling and general pressure management are missing | [partial](gaps-spilling.md) |
| G18 | Tensor instruction and async-pipeline coverage is specialized | open |
| G19 | Native GPU architecture coverage remains SM80/A100 | open |
| G20 | Source-to-binary toolchain independence is incomplete | open |
| G21 | Framework operator coverage is still a small whitelist | [partial](gaps-convolution.md) |
| G22 | Model control flow and shapes are static specializations | open |
| G23 | Model storage/dtype and AMP coverage remain restricted | open |
| G24 | Arbitrary strides, aliases and mutation effects are unsupported | open |
| G25 | Broadcast support is partial and does not extend to batched matmul | [partial](gaps-model.md) |
| G26 | Hard resource bounds exclude common large ML shapes | open |
| G27 | Attention feature coverage remains bounded after masked/causal SDPA | open |
| G28 | Attention is decomposed and materializes quadratic intermediates | open |
| G29 | Embedding does not provide general gather/scatter/index semantics | open |
| G30 | Embedding gradient and broadcast-reduction scaling is a performance gap | open |
| G31 | Loss support is still MSE plus a narrow cross-entropy contract | open |
| G32 | Autograd interoperability is first-order and bounded | open |
| G33 | RNG, dropout and mutable model buffers remain open | open |
| G34 | Optimizer family and option coverage remains limited | [partial](gaps-optimizers.md) |
| G35 | Optimizer snapshots are not complete application checkpoints | [partial](gaps-checkpoints.md) |
| G36 | Adaptive/indexed training cannot use full CUDA-graph replay | open |
| G37 | Indexed runtime validation introduces synchronizing host work | open |
| G38 | Distributed and multi-device training are not integrated native workloads | open |
| G39 | KV decoding requires precompiled positive exact-length variants | [partial](gaps-decode.md) |
| G40 | KV cache memory management copies prefixes and lacks paging | open |
| G41 | Production serving behavior is not established by bounded decode tests | open |
| G42 | Real build and dependency graphs are not resolved | [partial](gaps-build-graph.md) |
| G43 | Custom kernels/operators are not automatically inserted into model execution | open |
| G44 | Arbitrary host Python orchestration and environments are not packaged/executed | open |
| G45 | General scheduling, tiling and cost models remain incomplete | open |
| G46 | Whole-model fusion, packing and launch planning remain limited | open |
| G47 | Cold/warm compilation and cache-hit overhead lack complete attribution | [partial](gaps-measurement.md) |
| G48 | Standard reports lack per-kernel steady-state distributions | [partial](gaps-profiling.md) |
| G49 | Full-workload comparisons and compile amortization remain incomplete | open |
| G50 | Independent application and semantic test coverage is still finite | open |
| G51 | Regression maintenance still contains a stale rejection and nondeterministic sampling | [fixed-for-recorded-regressions](gaps-model.md) |

## Research and engineering cycle

Each bounded dossier defines the gap and numerical/effect/resource contract, compares a candidate before promotion, retains independent validation and failures, and identifies its production path. Experiments live under `compiler_exp/gap_closure/`; production imports none of them. The [policy continuity audit](gaps-continuity.md) separately checks installed policies against frozen code and retained hardware-session artifacts.

The [repository inventory](../../compiler_exp/gap_closure/repository-map.json) reads and hashes all Git-tracked files and extracts every Python module’s imports and top-level definitions. Deep source review covered Python/PTX/CUDA entry routes; selection, allocation, scheduling and encoding; emulator/reference arithmetic; model capture, autodiff, kernels, storage/runtime/replay/state; and repository/build/transport/CLI/packaging paths. This is not a claim of manual semantic review of every historical experiment or proof of arbitrary workloads.

Remaining foundation gates include general cross-stage semantic conformance, buffer/layout and shared-memory safety, wider memory/atomic/order support, pressure management and device ABI. Model dtype/effect/RNG/operator coverage, complete original host/build/application execution, dynamic cache/serving/distributed contracts and held-out performance remain independent work. The complete checkpoint covers only the admitted owned native program. New targets and toolchain independence keep their separate milestones. Do not raise resource bounds, weaken tolerances, add vendor fallback or refresh measured policies to change an unsupported row into a success.

Validation environments and all run outcomes are recorded in the [experiment index](../../compiler_exp/gap_closure/README.md). Historical reports retain their original test counts and dates.
