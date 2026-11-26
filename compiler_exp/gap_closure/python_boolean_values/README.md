# Boolean values and operand selection: next G06 cycle

Status: retained first-stage hypothesis. The subsequent
[original-source cycle](promotion/README.md) promotes contextual Boolean values
and a separately hardware-validated native predicate-path repair.

General Boolean-valued expressions remain outside the production source contract.
Python `and` and `or` return one operand; forcing every result to zero/one changes
ordinary numeric programs. The candidate keeps homogeneous scalar operand values
and evaluates later operands only on the path that needs them. Predicate locals,
literal True/False, not, bool(x) and chain results use explicit truth semantics;
typed integer conversion produces zero/one without treating arbitrary numeric
and/or results as Boolean. FP32 NaN must be truthy and signed-zero operand results
must retain their selected bits.

The probe independently computes results and per-thread atomic counts for both
short-circuit directions, Boolean composition and chain/integer conversions. It
compares complete input/output/count/guard bits through four PTX/normal/poisoned
levels. This is first-stage CPU evidence only. Numeric/FP32 special-value contexts,
literal/type rules, mixed-kind rejections, source/helper effects and ownership,
Boolean parameter/array ABI, original diagnostics, toolkit, independent A100,
remote/wheel/full integration and resource policy remain separate required work.
The candidate is not imported by production.

```sh
python compiler_exp/gap_closure/python_boolean_values/candidate.py
python compiler_exp/gap_closure/python_boolean_values/probe.py
```

The [frozen baseline](baseline.json) retains the original rejection. The
[effect probe](candidate-offline.json) passes twelve complete executions and exact
atomic counts. The [domain probe](domains.json) adds sixty executions with raw
selected operand bits across five numeric types, including FP32 NaN payloads,
infinities, signed zeros and subnormals; five actual ptxas controls pass.
[Ten rejection controls](negative-controls.json) keep mixed kinds, leading-literal
contexts, Boolean ABI/array, helper effects and convergence limits explicit.
No GPU execution or production promotion is claimed. The
[canonical dossier](../../../docs/compiler/gaps-boolean-values.md) records what
must still be resolved before a general Boolean-value production contract.
