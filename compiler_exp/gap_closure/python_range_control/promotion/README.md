# Original finite ranges and runtime loop else

Status: implemented and integration validated for the bounded original-source contract.
The canonical [loop-control dossier](../../../../docs/compiler/gaps-range-control.md)
retains the contract, historical counterexamples and remaining G06 scope.

## Gap and solution

The [bounded baseline study](../README.md) reproduces finite i32 ranges that wrap
before exhaustion. A separate minimized O0/O3/runtime-bound/signed/unsigned A100
study isolates an external loop-optimization disagreement. The final lowering
checks whether a nonunit next counter would leave i32 and branches to exhaustion
before applying that increment. Since stop is i32, such a value necessarily
crosses stop. Unit-step loops retain their prior emitted path. Runtime range and
while else have separate exhaustion and break labels; else itself uses its
outer loop context. Continue reaches the increment/test, break skips else, and
kernel/helper returns retain their own exit scopes.

The live-path helper analyzer treats range/while as possibly empty and keeps
else/break/return paths distinct. No body-only assignment becomes definitely
initialized merely because the body has a constant-looking trip count. The
barrier checker propagates thread-dependent loop exits into exhaustion else.
Bounds remain snapshotted i32 values; steps remain constant, nonzero i32 numeric
integers. Dynamic/Boolean steps remain outside the existing admitted contract.
No backend/encoding/resource limit, vendor fallback or user module execution is
introduced. Production imports no experiment module.

## Independent gates

- [Frozen classifications](baseline.json): sixty original source/entry records,
  with 36 runtime loop-else rejections and 24 accepted potentially overflowing
  kernels. Those accepted originals are not launched without the separate
  watchdog; they are not mislabeled as rejection evidence.
- [Offline matrix](candidate-offline.json): **4,320 complete executions / 240 rows**,
  three original source trees, twenty kernels, four levels, two seeds, three
  lengths and PTX/normal/poisoned SASS. Independent whole-bit references compare
  all output/input/guard/tail bytes.
- [Generated controls](control-properties.json): **128 finite boundary/step/bounds
  snapshot/target mutation/skip/break/exhaustion cases / 1,536 executions**.
  Original sources, inputs, exact Python range values and references remain
  retained under seed 79123.
- [Negative controls](negative-controls.json): sixteen range-domain, missing-path,
  initialization, nearest-loop and convergence rejections with original diagnostics.
- [Toolkit](oracle-probe.json): actual ptxas accepts three original trees / sixty
  kernels, with 22 raw source/cubin/listing/log identities.
- [A100](candidate-a100-20261124T111006Z.json): **2,400 launches**, three original
  source trees/twenty kernels, emitted PTX controls and four OA levels, two seeds,
  lengths 0/1/259/384 and three 128-thread CTAs. Original graphs/manifests/sources,
  cubins/disassembly and all before/after input/output bytes remain archived.
- [Candidate](candidate-audit.json) and [production](promotion-audit.json) audits:
  **17,203,200 independent bytes / 240 exact native word/register/frame outputs**,
  zero frames, original graph/native ownership and corrupted-output rejection.
- [Production regressions](targeted-suite.log): **920 frontend tests**, including
  **150 new tests** for original source/control/type/ownership and nearest loop
  else/return binding. The first old-test run passes 765 cases and finds five
  stale runtime-else rejection assertions. [Exact updates](test-contract-updates.json)
  replace those with genuine missing-value/path/initialization or unsupported
  runtime-step cases, alongside positive numerical controls. Numerical references,
  tolerances, rejection classes and resource bounds are not weakened.
- [Application continuity](application-continuity.json): **240 preceding original
  helper-return PTX/native outputs newly regenerate exactly**; **2,534 earlier
  direct/inherited occurrences** keep their proof/archive hashes and unchanged
  backend input binding. The **2,774 total** is explicitly split into these counts.
- [Frontend comparison](frontend-continuity.json): eighteen original files / 62
  kernels retain admission/rejection identities. `test_pyfront_semantics.py` has
  changed PTX from its intended nonunit guards; independent Python/PTX/SASS
  regressions validate those source semantics. This is not blanket PTX equality.

`first-fixture-target-initialization-failed.log` retains a fixture reading a new
runtime range target after a possibly empty loop. The fixture now initializes
its target explicitly; production flow analysis remains conservative.
`first-boolean-step-fixture-failed.log` retains an unsupported Boolean step used
by a new positive fixture; the unit-step capability control uses numeric one.
No source admission was changed to hide either rejection.

The full Python 3.12.10/Torch 2.7.1 suite passes **4,112 tests**. The separate
Python 3.14.5/no-Torch suite passes **3,627 tests with fifteen skips**. The
[wheel audit](wheel-audit.json) checks 132 exact installed identities and 240
original source/native regenerations. The [public Modal build](normal-remote-audit.json)
retains original graph/native ownership. The [integration record](validation.json)
binds final source/test/experiment identities, all proof/log/research evidence and
live/exported documentation checks. Broad G06 and the complete 51-gap program
remain incomplete. No speed claim is made.
