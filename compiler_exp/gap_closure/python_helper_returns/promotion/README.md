# Original-source pure scalar helper returns

Status: implemented and integration validated for the bounded original-source contract.

## Gap and solution

The earlier [flag-normalization experiment](../README.md) changed source identity
and excluded ordinary break/continue/exhaustion and branch-complete return sets.
This candidate analyzes original scalar helper paths and emits a private return
home and exit label at each expansion. Each return copies its typed scalar and
branches to that label, leaving the caller's copied arguments, locals and loops
isolated. Nested calls have distinct return homes/labels. No helper return emits
a kernel exit or executes user module/package top-level code.

The analyzer joins only live fall-through paths for definite initialization;
return, break and continue paths retain separate states. General loops may run
zero times. Nonempty finite tuples can prove an unconditional first-iteration
return; empty tuples do not initialize targets. Tuple exhaustion else and nearest
loop breaks retain their distinct paths. Runtime range/while else remains
unsupported. Scalar annotations, pure calls, no-memory/no-atomic/no-barrier effects,
recursion/expansion limits and original dependency/source ownership remain active.
One existing terminal return keeps its exact prior emission path.

## Experiments

- [Frozen baseline](baseline.json): sixty original source/entry rejections across
  same-file, imported and remote package trees.
- [Initial direct experiment](initial-direct-offline.json): three original helper
  routes through four PTX/normal/poisoned levels, 36 complete executions.
- [Path controls](paths-probe.json): five accepted all-path/initialization/loop
  controls and seven missing/type/zero-trip/empty/break rejection boundaries.
- [Negative controls](negative-controls.json): 24 original completeness/type/
  initialization/effect/recursion rejection controls with original file/line.
- The expanded fixture has twenty kernels: piecewise/branch-only/live-path
  initialization, range/while/nested returns, break/continue/exhaustion, empty and
  unconditional first-iteration returns, caller snapshots, argument mutation,
  nested/multiple calls and i64/u64/FP32 results. Independent references explicitly
  wrap signed intermediate values where later comparisons observe them.

`first-path-harness-failed.log` retains a missing interpreter-memory import.
`first-diagnostic-path-alias.log` retains a macOS `/var` versus `/private/var`
diagnostic assertion mismatch; comparisons now resolve both original paths.
`first-offline-source-race.json` and its log are **excluded from validation**:
candidate edits overlapped that first matrix run. The final harness freezes all
candidate inputs at entry and rejects any source change before publication.
This is a harness repair, not numerical or source-identity evidence from that run.

The final [A100 gate](candidate-a100-20261124T055307Z.json) passes **2,400 launches**
on three original source trees/twenty kernels with emitted PTX controls, four OA
levels, two seeds and lengths 0/1/259/384. The [production audit](promotion-audit.json)
checks **16,465,920 complete independent bytes**, **240 exact native word/register/
frame outputs**, original source/helper ownership and corrupted-output rejection.
The [offline matrix](candidate-offline.json) passes **4,320 executions / 240 rows**;
[generated controls](control-properties.json) add **64 original combinations /
768 executions**. Actual [ptxas](oracle-probe.json) accepts sixty original kernels,
with 22 retained raw artifact hashes. All **156 new tests** and **425 focused
regressions** pass. The [public CLI/Modal check](normal-remote-audit.json) retains
original graph and ownership through normal worker/local publication.

The first 2,400-launch passed archive remains retained; a focused single-entry
test subsequently exposed missing same-file source ownership without static
constants/tuples. The final candidate preserves that graph and reruns the complete
source-exact hardware gate. `first-float-cast-fixture-failed.log` retains a fixture
using an unsupported f32 marker call; typed FP32 literal returns now test the
existing language contract. `first-continuity-relative-path-failed.log` retains a
report-publication path assertion, repaired by resolving the report at entry.
The first existing-frontend run passes 610 tests and finds four old early-return
rejection assertions. Their [updates](test-contract-updates.json) use genuine
missing-path/bare-value failures with full positive capability controls.

The [frontend audit](frontend-continuity.json) preserves eighteen earlier files /
62 exact PTX kernels and rejections. The [application audit](application-continuity.json)
newly regenerates **240 prior tuple originals** to identical complete PTX/native
words/registers/frames. It separately inherits **2,294 earlier direct replay
occurrences** only after checking every retained proof/archive hash and unchanged
executable backend identity. These are **2,534 prior occurrences in total**, with
the inherited/direct counts explicit; no claim of a new direct replay of all
2,534 is made. All 49 measured policies and 98 sessions retain their exact inputs.
The full Python 3.12.10/Torch 2.7.1 suite passes **3,962 tests**. The separate
Python 3.14.5/no-Torch suite passes **3,477 tests with fifteen skips**. The
[installed-wheel audit](wheel-audit.json) checks 132 exact source/data identities
and 240 original source/native regenerations. The [integration record](validation.json)
binds final source/test/experiment identities, all independent evidence/logs and
documentation. The full 51-gap program and broader Python/ABI/effect semantics
remain incomplete. The [canonical dossier](../../../../docs/compiler/gaps-helper-returns.md)
retains the current contract and limits.
