# Repository compiler research and implementation

The target is `openassembler compile PATH`: preserve the CPU application and
compile its GPU computation to OpenAssembler-owned A100/SM80 cubins. CPU host
execution is part of the design. OA-only is the default GPU policy; unsupported
GPU computation fails explicitly. A dependency's vendor kernel does not count
as an implemented OA operation.

## Current implementation and evidence

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
