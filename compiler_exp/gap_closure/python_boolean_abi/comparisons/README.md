# Logical Boolean comparisons

Status: original boundaries recorded; isolated candidate passes initial offline gate; production unchanged.

The scalar and storage cycles make Boolean operands first-class predicates while
the comparison route still rejects `a == b`, `a < b` and Boolean chains with
`comparing comparison results`. Four original same-file scalar/array/helper/chain
sources retain exact [baseline diagnostics](baseline.json). A top-level exception
forbids user-code execution during discovery. Scalar/storage support must not
silently count these source operations as admitted.

The intended contract follows false/true as zero/one for equality, inequality and
ordering. Evaluate each operand once in original order. Chains compare adjacent
values and short-circuit before later loads/atomics/helper calls. Original literal
contexts, numeric-versus-Boolean type rules, exact helper annotations, caller copies,
control/convergence and source/expression/native ownership remain explicit.

A candidate should compare canonical scalar Boolean representations using existing
validated integer/predicate primitives. That does not by itself prove the source
contract. Independent truth-table/expression/effect references, every operand pair,
comparison operator, literal/shadow/union negatives, pure helper return paths,
loops/branches/shared storage, chained effects, poisoned/native/PTX/ptxas/A100 and
actual public scalar/buffer source bindings are required. Production promotion
must retain previous native/policy/installed/full-suite/docs gates without changing
numerical reference bits, tolerances or resource bounds. Identity/membership tests,
Boolean arithmetic/atomics, mixed scalar unions and general Python remain distinct.


The isolated [candidate](candidate-pyfront.py) materializes each predicate as a
canonical zero/one u32 register and uses existing integer comparison primitives.
The original operand evaluation and chain snapshots remain in the existing source
route. Its [initial independent offline gate](candidate-offline.json) passes 2,784
complete executions / 116 rows: all six relations with pair/literal/branch forms,
helper results, nested numeric comparison results, chains and per-lane atomic
once-only/short-circuit effects at four PTX/normal/poisoned levels. This is initial
CPU evidence. Original-source packages, negative boundaries, property/host/native
custody, actual ptxas/A100 and full integration remain required before promotion.
