# Explicit static scalar and tuple values: next G05 cycle

Status: retained normalization hypothesis. The [original-source promotion](promotion/README.md)
now integrates explicit Final values with source/diagnostic/dependency/ownership
identity. Earlier results below remain the original narrower experiment.

The [baseline](baseline.json) records three current failures: explicit
`typing.Final` scalar multiplication/bias values, indexed tuple coefficients,
and an FP32 scale. The [original helper source](corpus/helpers.py) and entry
raise at top level; the compiler reads source without executing either module.
The new local-package helper graph resolves the calls correctly, then rejects
captured globals or helper memory access. This is a G05 boundary beyond call
resolution.

The [literal evaluator](constexpr.py) is an isolated finite-AST hypothesis.
It recognizes explicit Final declarations, numeric literals/operators, previous
static values, finite tuples and compile-time tuple indices, with depth/size/
integer-bit bounds. It never imports, evals or executes user source. A separate
normalized source tree replaces statically used values under the existing typed
frontend, including contextual i64/u64 wrapping and FP32 literal conversion.
Original sources and normalized dependency identities remain distinct.

The [candidate probe](candidate-offline.json) passes **36 complete executions**:
three source kernels × four levels × PTX/normal/poisoned SASS, comparing all
input/output/guard/tail bytes with independent signed/unsigned modulo64 and exact
FP32 formulas. Seven controls reject calls, rebinding, tuple mutation, excessive
shift sizes, unresolved names and list declarations. These finite controls are
not a complete constant/type/effect grammar. Comprehensive arithmetic/exception,
import-alias, annotation/host-effect, source-location and original dependency
identity integration, toolkit and original-source A100 gates remain open. This
candidate is not production support or arbitrary Python metaprogramming.

The [first annotated-rebinding counterexample](first-annotated-rebind-failure.json)
retains a wrong specialization when a later non-Final annotation overwrote a
Final binding. `first-constexpr.py` keeps that candidate; the corrected finite
normalizer rejects the rebinding and adds its explicit regression. The
[toolkit gate](oracle-probe.json) accepts normalized emitted PTX and retains raw
cubin/listing/log hashes; it does not establish original-source graph integration.

The next cycle must bind static values to the original source graph and declare
which values are compile-time constants versus host-only effects. It must retain
host application behavior rather than treating every module binding as immutable.
The canonical [helper dossier](../../../docs/compiler/gaps-python.md) links this
research boundary; broad G05 remains open.

```sh
python compiler_exp/gap_closure/python_static_values/baseline.py
python compiler_exp/gap_closure/python_static_values/probe.py
python compiler_exp/gap_closure/python_static_values/oracle_probe.py
```
