# Original-source static numeric and tuple values

Status: production integrated; complete source-exact evidence retained.
The canonical [static-value dossier](../../../../docs/compiler/gaps-static-values.md)
contains the exact language, source/effect/resource contracts and open G05 scope.

## Gap to validated production

The [retained first experiment](../README.md) replaced constants in a separate
normalized source tree. It exposed an annotated-rebinding counterexample and
could not establish original dependency/diagnostic identity. This cycle resolves
explicit Final declarations within the existing original-source module graph,
folds them before typed helper/entry checks and records exact values/use identity.
Each helper retains its defining namespace. A per-kernel specialization cache
preserves values for every kernel using the same helper.

`baseline-*.py`, candidate generators and exact generated modules retain the
pre-change and reviewed sources. No arbitrary source execution/eval, implicit
mutable capture, runtime container or vendor fallback is introduced. Plain Final,
qualified/import aliases, direct/module/re-exported local constants, numeric
operators, finite tuples/indices/slices/len and original paths/hashes are covered.
Resource/type/effect/import/mutation/cycle boundaries remain explicit.

## Evidence

- [Baseline](baseline.json): 36 original source/entry rejections before promotion.
- [Original-source offline gate](candidate-offline.json): 144 rows / **2,592 whole
  executions** with independent modulo64/FP32/negative-zero bit formulas, complete
  guards, unchanged inputs/tails, and PTX/normal/poisoned SASS at all four levels.
- [Expression property gate](property-probe.json): **2,048 deterministic integer
  expression trees**, complete source expressions and independent reference values.
- [Negative controls](negative-controls.json): **29** original diagnostic/type/
  binding/effect/operation/size/cycle/tuple-index restrictions.
- [Toolkit controls](oracle-probe.json): four original trees / 36 kernels accepted
  by actual ptxas; **30 raw artifact hashes**.
- [First A100 report](candidate-a100-20261124T021108Z.json) and
  [byte/source audit](first-hardware-audit.json): 1,440 launches and 9,338,880 bytes
  before bounded-power and local-len-shadow hardening. Exact old sources stay
  retained in `first-static_values.py` and `first-fixtures.py`.
- [Final A100 report](candidate-a100-20261124T021349Z.json): **1,440 launches**,
  original emitted PTX controls and four OA levels, four source trees/nine kernels,
  two seeds, lengths 0/1/259/384, three 128-thread CTAs. Original files, full PTX/
  graphs/manifests/cubins/disassembly/inputs/outputs remain in its hashed archive.
- [Candidate audit](candidate-audit.json) and [production audit](promotion-audit.json):
  **9,338,880 complete bytes**, **144 exact current native outputs**, registers,
  zero frames, exact original source/static/helper/ownership and corrupted-output
  rejection. No tolerance or timing is pooled between successful gates.
- [Production regressions](targeted-suite.log): **118 pass**, including source-only
  constants, local-shadow rejection, helper usage for multiple kernels, exact
  FP32 rounding, original dependency changes and ownership tampering.

The first focused test used -52 for `(-17//3)*9+(-17%3)`; its correct independent
value is -53. `first-floor-reference-failed.log` retains that harness failure;
no compiler arithmetic or tolerance changed. `first-continuity-inventory-failed.log`
keeps a replay inventory check that omitted the two policy-audit metadata files;
the corrected harness includes their actual source identities, with all old
native/code/numerical gates still required.

Both full suites pass: **3,656 supported tests** (Python 3.12.10/Torch 2.7.1,
zero skips) and **3,171 compiler-only tests with fifteen skips** (Python 3.14.5,
no Torch). The [wheel audit](wheel-audit.json) checks 131 source/data identities
and all 144 original-source PTX/native outputs. The [public Modal audit](normal-remote-audit.json)
preserves original declarations/values and dependency ownership through the normal
CLI/worker/publication. The [frontend audit](frontend-continuity.json) retains
seventeen earlier files / 62 unchanged kernels and rejection records. The
[policy audit](../../foundations_20261121/audit-20261124T022017Z.json) preserves all
49 policies and 98 original sessions. The [complete native replay](application-continuity.json)
passes **2,150 earlier occurrences**, with exact current words/registers and the
full preceding numerical/ownership/source/archive chain. The
[final integration record](validation.json) binds final sources, suites, failures,
original byte/native/ownership evidence and live/exported documentation. Production imports no experiment module.
General Python objects, metaprogramming and arbitrary host state remain open.

```sh
python compiler_exp/gap_closure/python_static_values/promotion/offline.py
python compiler_exp/gap_closure/python_static_values/promotion/property_probe.py
python compiler_exp/gap_closure/python_static_values/promotion/negative_controls.py
python compiler_exp/gap_closure/python_static_values/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_static_values/promotion/run_modal.py
python compiler_exp/gap_closure/python_static_values/promotion/audit.py compiler_exp/gap_closure/python_static_values/promotion/candidate-a100-20261124T021349Z.json /tmp/static-value-audit.json
python compiler_exp/gap_closure/python_static_values/promotion/normal_remote_check.py
python -m pytest -q tests/test_pyfront_static_values.py
```
