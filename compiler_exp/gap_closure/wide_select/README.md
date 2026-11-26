# Scalar 64-bit select research and promotion

Status: production integrated; complete source-exact evidence retained.
The canonical contract and remaining G11 limits are in the
[wide scalar dossier](../../../docs/compiler/gaps-wide.md#scalar-64-bit-conditional-selection).

## Gap and solution

The actual [CUDA noinline fixture](noinline.cu) emits `selp.b64`, a production
admission rejection after successful direct-call expansion. `baseline-ra.py`
and `baseline-contracts.py` retain the preceding production sources;
[baseline.json](baseline.json) retains 36 rejected typed sources.

The [candidate](candidate.py) adds only `selp.b64/u64/s64/f64` using two existing
recovered scalar SEL instructions under one predicate. Operand materialization
and fresh result binding preserve low/high snapshots, parameters, immediates,
negation, outer guards and aliases. No encoding table or guessed field changes.

## Experiments and independent validation

- [First offline record](first-candidate-offline.json): sixteen cases and 1,152
  functional executions. Their FP64 declarations were not legal toolkit inputs.
- [First toolkit record](first-oracle-probe.json): twelve accepted integer/bit
  sources, four rejected FP64 sources; original [raw artifacts](first-oracle/)
  remain unchanged. `oracle-api-failed.log` keeps an earlier listing-API error.
- [Typed investigation](typed-investigation/) and [reference observation](ptx-reference.json):
  all four corrected FP64 sources pass ptxas. Matching FP64 registers are required.
- [Expanded toolkit audit](oracle-audit.json): 36 accepted sources and 144 raw
  artifact hashes. Register, both immediate positions, constant-bank parameter,
  both destination aliases, false guard and negated predicate forms are covered.
- [Expanded offline gate](candidate-offline.json): 144 rows / 2,592 PTX/normal/
  poisoned executions, complete input/output guards and inactive tails.
- [A100 report](candidate-a100-20261123T205126Z.json): all 1,480 launches pass.
  Thirty-six legal PTX configurations plus original CUDA noinline source;
  original ptxas control and four OA levels, lengths 0/1/259/384, two seeds,
  384 lanes. Complete raw archive is named and hashed by the report.
- [Candidate audit](candidate-audit.json) and [production audit](promotion-audit.json):
  15,155,200 independent bytes and 148 native word/register/resource comparisons,
  ownership, source identities and corrupted-output rejection.
- [Focused regressions](targeted-suite.log): 87 pass. The
  [first diagnostic mismatch](first-targeted-diagnostic-failed.log) preserves a
  parser rejection that the initial test incorrectly expected from the backend.

The exact validated candidate was copied to production. Tests use the standalone
[fixture corpus](wide_select_fixtures.json); production imports no experiment.
Both full suites pass: **3,361 supported tests** (Python 3.12.10/Torch 2.7.1)
and **2,876 compiler-only tests with fifteen skips** (Python 3.14.5/no Torch).
The [isolated wheel audit](wheel-audit.json) matches all 129 installed source/data
identities and 148 native outputs. The [policy audit](../foundations_20261121/audit-20261123T205620Z.json)
keeps all 49 measured policies and 98 original sessions unchanged. The prior
[native replay](application-continuity.json) passes all **1,786 prior kernel
occurrences**. The [integration record](validation.json) binds exact current
sources, independent reports, raw artifacts, full suites and live/exported docs. This is a correctness result, not performance
or general FP64 arithmetic support. The original 51-gap program is incomplete.

```sh
python compiler_exp/gap_closure/wide_select/probe.py --candidate
python compiler_exp/gap_closure/wide_select/oracle_probe.py
python compiler_exp/gap_closure/wide_select/audit_oracle.py
modal run compiler_exp/gap_closure/wide_select/run_modal.py
python compiler_exp/gap_closure/wide_select/audit.py compiler_exp/gap_closure/wide_select/candidate-a100-20261123T205126Z.json /tmp/wide-selection-audit.json
python -m pytest -q tests/test_backend_wide_select.py
```

The baseline replay explicitly installs frozen baseline modules and the candidate
replay installs staged modules. `pre-replay-*` records preserve the first expanded
runs before this reproducibility improvement; results and production sources did
not change. `replay-sample.log` retains a failed optional process-inspection
command (a shadowed executable lacked its Python module); the complete replay
subsequently passed without using that diagnostic.
