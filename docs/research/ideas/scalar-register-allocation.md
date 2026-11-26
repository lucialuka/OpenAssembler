# Register allocation, aligned groups and guarded definitions

Status: implemented

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Keep live values in a bounded set of GPU registers, reuse registers only after their old values are dead, and reserve consecutive aligned groups for wide loads and instructions. A guarded instruction must preserve the old value when its predicate is false.

## Findings and implementation decision

The scalar compiler uses virtual values, liveness, stable homes for repeated/guarded definitions and aligned register-group allocation. It handles the tested pointer-increment, nested-loop, vector and partial-definition cases. Treating `@p dst=...` as a fresh unconditional definition previously lost the old value and was repaired.

Allocation is an implemented correctness/resource feature; its speed contribution was not independently isolated from instruction selection and scheduling. Later November 6 alias-binding and scoreboard repairs are separate from the original allocator design.

The [November 23 atomic-ordering investigation](../../compiler/gaps-atomics.md)
also repairs branch-local lazy pointer materialization and definition-time
snapshots for wide products with mutable operands. Independent counterexamples
fail the original compiler and pass normal/poisoned execution and A100 controls.
These are value-lifetime repairs, with no isolated speed claim.

The [bounded spill extension](../../compiler/gaps-spilling.md) adds private
slot/reload/store semantics when ordinary allocation exhausts or an explicit
register limit requires them. Separate A100 controls validate guarded loop
values, static-local coexistence, actual registers and frame reservations.
The subsequent extension validates entire aligned pairs/quads and predicates,
including self-guarded/negated loop updates, with another 320 A100 launches.
Slot reuse and general occupancy policy selection remain open.

The subsequent [virtual pressure pass](../../compiler/gaps-spilling.md#pressure-aware-virtual-scheduling)
compares a reordered allocation only after ordinary pressure requires spills.
It preserves guards, dependency edges, fixed boundaries and source load order,
and admits only fewer slots with non-increased per-block local transfers/frame.
Independent A100 controls retain full bytes and paired resident intervals;
host code-generation cost is measured separately and can increase even when
the old executable is retained. General occupancy policy remains open.

## Method and validity checks

Compare the PTX interpreter and poisoned-register SASS execution on branching/loop fixtures, validate consecutive pairs/quads, inspect declared counts and run selected kernels on A100. A lower count alone does not select the fastest kernel: spills, lifetime, occupancy and operand collection interact.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
.venv/bin/python -m pytest -q tests/test_backend.py tests/test_backend_loops.py tests/test_backend_ops.py
.venv/bin/python -m openassembler compile tests/data/backend_loops/dot_rows.ptx --verify --out-dir output/register-check
```
The final command is a compiler/interpreter check, not a claim about an arbitrary launch shape. Use an actual launch specification for GPU execution.

## Code and retained evidence

- [Allocator and selector](../../../openassembler/backend/ra.py)
- [Loop fixtures/tests](../../../tests/test_backend_loops.py)
- [Correctness regression dossier](scalar-dependency-correctness.md)

## What remains open

Slot reuse, cross-region pressure scheduling, occupancy policies and larger model
kernels need further coverage. Retain all implicit register readers when
optimizing vector/tensor groups.
