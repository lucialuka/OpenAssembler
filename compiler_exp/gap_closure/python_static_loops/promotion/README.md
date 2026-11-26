# Original-source finite tuple loops

Status: implemented and integration validated for the bounded original-source contract.
The canonical [tuple-loop dossier](../../../../docs/compiler/gaps-static-loops.md)
contains the language/type/control/resource/source contract and open G05/G06 scope.

The [initial normalized experiment](../README.md) omitted observable loop control.
The original eight-case matrix remains retained. This cycle lowers finite tuples
in original source to an internal counter/value-selection chain and one emitted
body. Real break/continue/exhaustion labels preserve targets, empty/nested/else
semantics, thread-dependent exits and kernel returns. The frontend preserves the
existing helper terminal-return, numeric type and convergence contracts. Exact
iteration values/original locations and source hashes join native ownership.
No backend/encoding template, vendor fallback or source execution is introduced.

## Independent gates

- [Frozen baseline](baseline.json): sixty original source/entry rejections, replayed
  with exact pre-promotion frontend modules instead of current production.
- [Offline gate](candidate-offline.json): **4,320 whole executions / 240 rows**,
  original same-file/imported/package trees, twenty kernels, four levels, two
  seeds, three lengths and PTX/normal/poisoned SASS. Independent modulo64/FP32
  references compare every output/input/guard/tail byte.
- [Generated control combinations](control-properties.json): **128 original
  tuple/skip/break/target-mutation/exhaustion combinations**, independently computed
  whole-bit references and **1,536 additional PTX/normal/poisoned executions**.
  Complete sources/inputs/expected bits and seed 53711 remain retained. This is
  finite extra semantic evidence, not additional GPU execution or universal proof.
- [Negative controls](negative-controls.json): twenty unsupported iterable/type/
  target/resource/helper-return/convergence rejection controls.
- [Toolkit gate](oracle-probe.json): three original trees / sixty kernels accepted
  by actual ptxas; **22 raw source/cubin/listing/log identities**.
- [A100 gate](candidate-a100-20261124T051600Z.json): **2,400 launches**, original
  emitted PTX controls and four OA levels, two seeds, lengths 0/1/259/384, three
  128-thread CTAs. Exact original files/PTX/graphs/manifests/cubins/disassembly and
  full inputs/outputs remain in the hashed archive.
- [Candidate audit](candidate-audit.json) and [production audit](promotion-audit.json):
  **16,465,920 complete independent bytes**, **240 native word/register/resource
  outputs**, source/loop/helper/ownership contracts, zero frames and corrupted
  output rejection. No speed claim or pooled timings.
- [Production focused regressions](targeted-suite.log): **150 pass**, including
  kernel early return, empty/exhausted else, target mutation/retention, nearest
  loop else/break, metadata/native custody and changed original source.
- All 464 earlier frontend tests pass before promotion. The [frontend continuity audit](frontend-continuity.json)
  retains eighteen earlier source files and 62 identical PTX kernels/rejections.
- The [wheel audit](wheel-audit.json) checks 131 installed source/data identities
  and all 240 original-source PTX/native regenerations in an isolated process.
- The [public Modal check](normal-remote-audit.json) passes original imported
  tuple graphs through ordinary CLI/worker/local publication and native ownership.
- The [policy audit](../../foundations_20261121/audit-20261124T052003Z.json)
  preserves all 49 measured policies and 98 original sessions. The
  [application replay](application-continuity.json) passes 2,294 prior native
  occurrences with exact words/registers and retained original proof chains.

## Retained failures

`first-keyword-fixture-failed.log` keeps source fixtures that accidentally used
Python keywords as function names; the corrected fixtures use valid names.
`first-target-type-failed.log` found a real candidate admission bug: a float
literal could enter an existing integer target through a permissive literal
helper. `first-tuple-emitter.py` retains that lowering; the final tuple path checks
that every integer-target element is an integer before materialization.
`first-policy-update-race.log` retains a replay inventory check started before
policy metadata rebinding completed; the harness now allows either metadata state,
while separately requiring the complete policy/native source and binary audit.

Production sources are exact copies of independently validated candidate modules.
Original dependency identities and iteration records flow through the existing
local/remote source graph. Runtime containers, arbitrary iterables, heterogeneous
nonrepresentable values, early helper returns and general Python host/control
semantics remain open. Production imports no experiment modules.

```sh
python compiler_exp/gap_closure/python_static_loops/promotion/baseline.py
python compiler_exp/gap_closure/python_static_loops/promotion/offline.py
python compiler_exp/gap_closure/python_static_loops/promotion/negative_controls.py
python compiler_exp/gap_closure/python_static_loops/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_static_loops/promotion/run_modal.py
python compiler_exp/gap_closure/python_static_loops/promotion/audit.py compiler_exp/gap_closure/python_static_loops/promotion/candidate-a100-20261124T051600Z.json /tmp/tuple-loop-audit.json
python -m pytest -q tests/test_pyfront_static_loops.py
```

The full supported Python 3.12.10/Torch 2.7.1 suite passes **3,806 tests**.
The separate Python 3.14.5/no-Torch suite passes **3,321 tests with fifteen skips**.
The [integration record](validation.json) binds exact production/test/experiment
sources, independent numerical/native/ownership evidence, full logs, documentation
and retained failures. Broad G05/G06 and the full 51-gap program remain incomplete.
