# Repository compiler research and implementation

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

The target is `compile PATH`: preserve the CPU application and
compile its GPU computation to OpenAssembler-owned A100/SM80 cubins. CPU host
execution is part of the design. OA-only is the default GPU policy; unsupported
GPU computation fails explicitly. A dependency's vendor kernel does not count
as an implemented OA operation.

## Current implementation and evidence

- [Recorded build commands and dependencies](gaps-build-graph.md): per-source
  NVCC flags, generated inputs, dependency hashes and remote relocation.
- [Pointer parameter metadata](gaps-parameter-abi.md): global/generic pointer
  annotations, scalar slot layout and launch-alignment/ownership checks.
- [Native atomics and bounded ordering](gaps-atomics.md): 32-bit global RMW,
  GPU acquire/release and CTA shared operations with independent hardware gates.
- [Typed Python atomic primitives](gaps-python-primitives.md): old-value returns,
  nested effects, checked buffer indices and convergence diagnostics.
- [Full-warp shuffle modes](gaps-collectives.md): indexed/up/down exchanges,
  reconvergence, aliased operands and explicit participation limits.
- [Static local storage](gaps-storage.md): per-thread frames, scalar32 access,
  donor-independent metadata and exact A100 resource/ownership gates.
- [Register and predicate spilling](gaps-spilling.md): private slots, aligned
  pairs/quads, guarded values, virtual pressure scheduling and bounded
  explicit/automatic pressure handling with separate kernel/compiler cost gates.
- [Checked shaped buffers](gaps-buffers.md): explicit layouts, bounds behavior
  and validated storage bindings.
- [Wide Python integer types](gaps-python-types.md) and
  [wide scalar lowerings](gaps-wide.md): wrapping integer contracts,
  shifts/comparisons, typed verification and source/hardware gates.
- [Floating literal contexts](gaps-wide.md#decimal-and-raw-floating-literal-contexts):
  binary64 decimal representation, use conversion, raw payload/width behavior
  and independent context/rounding/A100 gates.
- [Typed constant expressions](gaps-wide.md#typed-scalar-operand-constant-expressions):
  integer precedence/casts and binary64 intermediate rounding, with explicit
  unresolved type/exception diagnostics and retained failed hardware controls.
- [Scalar 64-bit selection](gaps-wide.md#scalar-64-bit-conditional-selection):
  integer/FP64 bit copies, aliases/guards/negated predicates, parameter/immediate
  snapshots and original PTX/CUDA A100 controls.
- [Low64 integer products](gaps-wide.md#low-64-bit-integer-multiplication):
  PTX/CUDA low products and Python i64/u64 multiplication, exact wrapping,
  aliases/guards, helpers/loops/casts and original-source A100 controls.
- [Optimizer policy and state](gaps-optimizers.md): SGD momentum/Nesterov,
  AMSGrad/maximize, exact options and resumed trajectory validation.

- [Complete supplied gap register](gap-register.md): all 51 original exit gates,
  bounded repairs and explicitly remaining work.
- [Exact-division verification](gaps-verification.md): hardware-isolated FCHK
  repair and deterministic emulator regressions.
- [Typed device helpers](gaps-python.md): static same-file/local-package scalar
  specialization, exact source/diagnostic/dependency custody and explicit effect limits.
- [Direct scalar PTX device calls and source libraries](gaps-device-functions.md):
  static specialization, lexical slots, deterministic module resolution, per-unit
  dependencies, local resources and executable custody.
- [Static local-package Python helper graphs](gaps-python.md#static-local-package-device-helpers):
  aliases/relative imports/re-exports, defining-module scalar specialization,
  original diagnostics and exact local/remote source ownership.
- [Explicit static numeric and tuple values](gaps-static-values.md): original Final
  declarations, bounded arithmetic/index/slice/len, defining-module specialization
  and exact source/value/dependency/native ownership.
- [Finite static tuple loops](gaps-static-loops.md): empty/target/break/continue/
  nested/exhaustion semantics, numeric type/convergence limits, exact original
  source and loop-value ownership.
- [Pure scalar helper returns](gaps-helper-returns.md): original return-path
  completeness, live-path initialization, caller isolation and native source ownership.
- [Compilation attribution](gaps-measurement.md): source/cache/worker phases,
  cold/hit evidence and remaining measurement boundaries.
- [Policy continuity](gaps-continuity.md): all installed policy binaries checked
  before rebinding compiler identities after reporting changes.
- [Global/shared memory and predicate assignments](gaps-memory.md): byte/halfword/64-bit
  global stores, scalar shared loads/stores, vector widths, guarded predicates,
  immutable snapshots and a hardware-validated shared64 barrier gap.
- [Repeated kernel profiling](gaps-profiling.md): optional repeated audits,
  per-contract distributions/resources and separately labeled whole-step times.

- [Masked attention and persistent model state](model-semantics.md): broadcasting,
  integer embeddings/CE, Adam/AdamW, cache ownership, CLI and A100 gates.
- [Batch-broadcast matrix products](gaps-model.md): rank-two through rank-four
  normalization, replicated-axis gradients, A100 ownership and regression repair.
- [Sigmoid and static tensor reductions](gaps-operators.md): bounded FP32
  framework operations, derivatives, A100 ownership and complete CLI updates.
- [Extended native optimizer policies](gaps-optimizers.md): SGD momentum/Nesterov,
  Adam/AdamW AMSGrad/maximize, complete direct state and resume gates.
- [Complete owned-state checkpoints](gaps-checkpoints.md): parameters/buffers,
  optimizer/RNG/application state, atomic loading and interrupted resume.
- [Explicit bounded decode specialization](gaps-decode.md): guarded exact-length
  misses, builder budgets, native cache ownership and full A100 transitions.

- [Semantic acceptance and diagnostics](semantic-acceptance.md): November 19
  saturation/FTZ correctness repairs, explicit scalar contracts and source errors.
- [Cross-stage scalar contract ledger](gaps-contracts.md): versioned forms,
  operand variants, stage rejections and a separate memory-width matrix; the
  [latest local-storage ledger](../../compiler_exp/gap_closure/local_storage/ledger-local-20261123.json)
  records 553 declarations with explicit unexecuted stages.
- [Coverage priorities and independent matrix](coverage-priorities.md): frozen
  CPU evidence, current bounds, open custom-kernel/model/training work and gates.

- [Source completeness and launch contracts](source-completeness.md): repaired
  PTX definition discovery, explicit parameter/header rejection, retained launch
  directives, launch checks and bounded local CUDA-header discovery.
- [Memory and latency research](memory-latency.md): existing positive/negative
  studies, new experiment contracts, resource tradeoffs and promotion criteria.
- [Research protocol and roadmap](research-plan.md): remaining application
  compilation work and measurable completion criteria.
- [Final validation and retained failures](validation.md): suite, A100,
  continuity and exact-policy evidence with explicit limits.

The [research catalogue](../research/README.md) remains the cross-project feature
index. Detailed compiler contracts live here, linked from their existing feature
dossiers. [STATUS](../STATUS.md), [repository workflow](../REPOSITORY_WORKFLOW.md)
and [commands](../COMMANDS.md) describe deployed behavior. Experiments for this
program live under `compiler_exp/`; production modules never import them.

Coverage is a property of an explicit workload and input domain. Source discovery,
successful compilation, functional emulation, hardware correctness and performance
are separate evidence states. This program does not claim arbitrary repositories,
all future inputs, or universal speed superiority are already supported.

The subsequent [runtime loop-control cycle](gaps-range-control.md#original-source-production-extension)
fixes finite i32 range-counter wrap and preserves runtime range/while else,
nearest-loop exits, source/initialization/convergence and native ownership.
The retained initial toolkit/native divergence has a separate minimized study.

[Typed condition chains](gaps-comparison-chains.md) now preserve single-evaluation/
short-circuit effects, pairwise literal types, numeric special-value truth and
original expression/source/native ownership. Their final original-source gate
passes 2,400 A100 launches; first-class Boolean values remain separate work.

[Boolean values](gaps-boolean-values.md) now preserve operand-returning lazy
and/or, contextual literals, explicit truth/conversion and original source/value
ownership. Actual A100 failures led to a control-flow predicate-readiness repair;
Boolean ABI and mixed-type unions remain separate work.

The subsequent [Boolean ABI study](gaps-boolean-abi.md) establishes the original
scalar/array/helper and byte-parameter admission boundary. Complete physical ABI
and hardware integration remain required.

[Logical Boolean helper signatures](gaps-boolean-helpers.md) now carry predicate
arguments/returns through pure static expansion, with exact caller/return/source
ownership and a 2,400-launch independent A100 gate. Boolean physical launch and
storage contracts remain in the separate [ABI study](gaps-boolean-abi.md).

[Narrow scalar entry ABI](gaps-narrow-parameters.md) now coordinates byte/halfword
slots and extension across parser, both selectors, verifier and public launch
bindings. Its original-source A100 gate passes 14,400 launches plus 240 explicit
run-spec launches. Logical Python Boolean physical launch/storage remains a
subsequent frontend/storage contract.

[Logical Boolean kernel parameters](gaps-boolean-parameters.md) now preserve
predicate bodies, canonical byte launch values and explicit source/manifest/native
roles. Original-source A100 and actual public Boolean run-spec gates pass;
Boolean Array/Buffer/shared storage remains separate work.

The next [Boolean storage study](gaps-boolean-storage.md) retains three original
Array/Buffer/shared rejections and defines independent canonical byte-storage,
host-buffer/shape/safety and shared-convergence gates before implementation.

- [Logical Boolean comparison research](gaps-boolean-comparisons.md): original scalar/storage/helper/chain boundaries and required evaluation/effect contract.

- [FP32 Conv2d and CNN training](gaps-convolution.md): static NCHW convolution, input/weight/bias gradients and a runnable two-layer CNN.
