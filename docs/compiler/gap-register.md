# Compiler gap register

The supplied 51-gap review remains the scope of this program. This implementation closes the fixed division counterexample and repairs the named regression defects; it adds bounded helper control flow, memory widths, tensor operators, matmul and measurement capabilities. **The complete 51-gap program is not finished.**

Final November 21 validation passes 1,686 tests in Python 3.12.10/Torch 2.7.1 with 0 skips, and 1,294 tests with 12 skips in the separate Python 3.14.5 compiler-only environment. A100 evidence and exact source/native-code identities are retained in the linked dossiers and [validation record](../../compiler_exp/gap_closure/integration_20261121/validation.json).

The [machine-readable register](../../compiler_exp/gap_closure/gap-register.json) retains every original exit gate. The [supplied review](../../compiler_exp/gap_closure/input-gap-register.txt) is retained unchanged, including its dated evidence and limits. A row marked partial requires the remaining original gate; it is not counted as a completed broad capability.

| Gap | Finding | Current state |
|---|---|---|
| G01 | Unresolved exact-division verification counterexample | [counterexample-resolved](gaps-verification.md) |
| G02 | Full semantic conformance is still a larger task than scalar admission | [partial](gaps-contracts.md) |
| G03 | Custom Python types and numerical semantics remain narrow | open |
| G04 | Reusable Python functions and library calls are not generally compilable | [partial](gaps-python.md) |
| G05 | Python objects, containers and compile-time programming lack a general route | open |
| G06 | Python control flow is useful but not unrestricted | [partial](gaps-python.md) |
| G07 | Custom-kernel arrays have no general tensor/view/layout API | open |
| G08 | Python GPU primitive vocabulary is incomplete | open |
| G09 | Memory safety and shared-memory initialization checks are incomplete | open |
| G10 | PTX parameter ABI and module grammar are bounded | open |
| G11 | Scalar arithmetic/types/modifiers do not cover general CUDA-emitted PTX | [partial](gaps-memory.md) |
| G12 | Global/shared memory widths and vector forms remain uneven | [partial](gaps-memory.md) |
| G13 | Address spaces and module storage are not generally lowered | open |
| G14 | Atomics, reductions and memory ordering are missing compiler foundations | open |
| G15 | Warp/block collective and synchronization support is narrow | open |
| G16 | Device calls, call stacks and cross-module device linking are absent | open |
| G17 | Register/predicate spilling and general pressure management are missing | open |
| G18 | Tensor instruction and async-pipeline coverage is specialized | open |
| G19 | Native GPU architecture coverage remains SM80/A100 | open |
| G20 | Source-to-binary toolchain independence is incomplete | open |
| G21 | Framework operator coverage is still a small whitelist | [partial](gaps-operators.md) |
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
| G34 | Optimizer family and option coverage remains limited | open |
| G35 | Optimizer snapshots are not complete application checkpoints | open |
| G36 | Adaptive/indexed training cannot use full CUDA-graph replay | open |
| G37 | Indexed runtime validation introduces synchronizing host work | open |
| G38 | Distributed and multi-device training are not integrated native workloads | open |
| G39 | KV decoding requires precompiled positive exact-length variants | open |
| G40 | KV cache memory management copies prefixes and lacks paging | open |
| G41 | Production serving behavior is not established by bounded decode tests | open |
| G42 | Real build and dependency graphs are not resolved | open |
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

The next unresolved foundation gates are general cross-stage semantic conformance, typed buffers and memory safety, wider memory/atomic/order support, pressure management and device ABI. Model dtype/effect/RNG/operator coverage, complete application/build/checkpoint execution, dynamic cache/serving/distributed contracts and held-out performance remain independent work. New targets and toolchain independence keep their separate milestones. Do not raise resource bounds, weaken tolerances, add vendor fallback or refresh measured policies to change an unsupported row into a success.

Validation environments and all run outcomes are recorded in the [experiment index](../../compiler_exp/gap_closure/README.md). Historical reports retain their original test counts and dates.
