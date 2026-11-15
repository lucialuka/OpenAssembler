# Application compilation research contract

The user confirmed CPU host execution plus OA GPU computation and OA-only as
the default on November 15, 2026. Preserve sources, datasets, checkpoints,
precision, program effects and training policy throughout the research.

## Workstreams and completion criteria

| Workstream | Current state | Required evidence before calling it solved |
|---|---|---|
| Source/entry completeness | Bounded repair implemented; see the [contract](source-completeness.md) | Positive/negative parser tests, every entry accounted for, complete native module inventory, ABI/launch checks and A100 regression |
| Dependency/build resolution | Proposed; existing inspection records imports/includes/build markers | Resolve real include/macro/generated-source configuration, package/native dependency identities and translation-unit/link graph; compare original build semantics |
| Host program execution and launch integration | Proposed; current CLI executes standalone plans or a bounded captured model | Preserve entrypoint/arguments, CPU code and dependencies; connect extension registrations and launches to OA modules; package a runnable application |
| Framework semantic coverage | Existing bounded FX/native model route | Compare FX extension, exported ATen normalization and dispatch integration; preserve alias/mutation/RNG/dtypes; test forward, derivatives, losses and optimizer state |
| Device compilation | Bounded scalar/tensor backend | Add blocker-driven device calls/linking, atomics/memory ordering, spilling and missing types with independent semantics and A100 correctness |
| Dynamic execution | Proposed | Compare guarded ahead-of-time variants, host control flow plus OA dispatch and explicit lazy specialization; test guard misses and unseen inputs |
| Memory/latency optimization | Measured existing exact policies plus new [research](memory-latency.md) | Complete numerical/resource/dependency gates, paired timing against incumbent and held-out confirmation at an explicit workload boundary |

A host dependency can stay on the CPU without pretending its algorithms were
translated into SASS. Moving a scikit-learn estimator to the GPU is a separate
implementation project. An optional mixed GPU backend would require a separate
explicit policy and ownership report; it has not been enabled here.

## Candidate process

1. Record a concrete failure and reduce it without changing its semantics.
2. Read production source, its tests, prior optimization dossiers and raw records.
   Fix the precision/layout, inputs, effects, baseline, resources and timing boundary.
3. Implement credible alternative candidates under `compiler_exp/`. Compare
   correctness, rejection behavior, compile cost, generated code and resources.
4. Refine failed candidates using their actual failure evidence. Keep failed
   records; distinguish an unsuccessful finite search, unavailable infrastructure
   and a fundamental impossibility.
5. Freeze a candidate before independent confirmation. Check complete outputs,
   guards, input immutability, synchronization, ownership, and application state
   where relevant. Do not alter tolerance or optimizer policy to pass a gate.
6. Promote a complete bounded feature into ordinary compiler/runtime/CLI paths
   with regression tests. Record implementation, correctness, performance and
   limitations separately. Recheck installed dispatch continuity when hashed
   compiler inputs change; source-hash updates alone are insufficient.

The baseline and new experiments are retained in
[repository coverage](../../compiler_exp/repository_coverage/README.md).
The initial structural survey inventoried 646 Python files and all 104 compiler
modules. It was not a manual line-by-line audit of all historical experiments.

## Application corpus and comparisons

Use pinned source revisions and dependency environments for distinct workload
families: CPU preprocessing plus GPU model; CNN inference/training; transformer
masking and changing sequence lengths; and a custom extension with cross-file
device dependencies. Generated sources, package/src layouts, CPU-only work and
checkpoint resume are explicit cases. A small synthetic coverage set is not an
estimate of arbitrary-repository success rate.

Measure both the original application and the OA incumbent, on the same A100
variant with fixed numerical contracts. Separate compilation/cold start from
steady execution. Application timing includes required CPU work, transfers,
packing, launches and optimizer updates. Preserve outputs, parameter/buffer
state, gradients, optimizer state, RNG progression and checkpoint behavior.
Bitwise identity is a separate stronger requirement from a fixed numerical
tolerance. No kernel-only measurement establishes a whole-application speedup.

Run the relevant focused tests, full suite and documentation checks. Keep
missing dependencies and skipped tests explicit. If a compiler identity changes
but emitted code does not, a complete retained-binary audit can establish
continuity; it does not generate new hardware timing evidence.
