# Repository compiler research and implementation

The target is `openassembler compile PATH`: preserve the CPU application and
compile its GPU computation to OpenAssembler-owned A100/SM80 cubins. CPU host
execution is part of the design. OA-only is the default GPU policy; unsupported
GPU computation fails explicitly. A dependency's vendor kernel does not count
as an implemented OA operation.

## Current implementation and evidence

- [Recorded build commands and dependencies](gaps-build-graph.md): per-source
  NVCC flags, generated inputs, dependency hashes and remote relocation.
- [Pointer parameter metadata](gaps-parameter-abi.md): global/generic pointer
  annotations, scalar slot layout and launch-alignment/ownership checks.
- [Relaxed GPU atomics](gaps-atomics.md): 32-bit global RMW/reduction semantics
  with multi-block contention and old-value return validation.
- [Checked shaped buffers](gaps-buffers.md): explicit layouts, bounds behavior
  and validated storage bindings.
- [Wide Python integer types](gaps-python-types.md) and
  [wide scalar lowerings](gaps-wide.md): wrapping integer contracts,
  shifts/comparisons, typed verification and source/hardware gates.
- [Optimizer policy and state](gaps-optimizers.md): SGD momentum/Nesterov,
  AMSGrad/maximize, exact options and resumed trajectory validation.

- [Complete supplied gap register](gap-register.md): all 51 original exit gates,
  bounded repairs and explicitly remaining work.
- [Exact-division verification](gaps-verification.md): hardware-isolated FCHK
  repair and deterministic emulator regressions.
- [Typed device helpers](gaps-python.md): statically inlined same-file scalar
  helpers with explicit type/effect limits.
- [Compilation attribution](gaps-measurement.md): source/cache/worker phases,
  cold/hit evidence and remaining measurement boundaries.
- [Policy continuity](gaps-continuity.md): all installed policy binaries checked
  before rebinding compiler identities after reporting changes.
- [Global memory and predicate assignments](gaps-memory.md): byte/halfword/64-bit
  stores, vector widths, predicate moves and immutable value snapshots.
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

- [Semantic acceptance and diagnostics](semantic-acceptance.md): November 19
  saturation/FTZ correctness repairs, explicit scalar contracts and source errors.
- [Cross-stage scalar contract ledger](gaps-contracts.md): 426 exact forms,
  operand variants, stage rejections and a separate memory-width matrix.
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
