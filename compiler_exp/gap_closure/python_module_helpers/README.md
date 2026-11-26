# Static local-package device helpers: next G04 cycle

Status: retained normalization hypothesis. The subsequent [promotion cycle](promotion/README.md)
uses defining-module namespaces and integrates source/dependency/diagnostic,
ownership and remote contracts before production. Earlier records below remain
unchanged and do not establish that broader result.

The [baseline](baseline.json) records three independent call-target rejections:
local-package signed repeated products through an import alias, unsigned helper
calls through a module alias and floating affine helpers. The
[entry source](corpus/entry.py), package initializer and helper modules deliberately
raise at top level. No module was imported or executed by the compiler. All
source hashes and exact production frontend/helper identities are retained.
This is not an arbitrary-package compatibility or host-execution test.

The intended research contract extends same-file pure scalar specialization to
an explicit local source graph: deterministic absolute/relative module binding,
per-module annotation/intrinsic resolution, hygienic helper names, transitive
calls, preserved source diagnostics and cycle/expansion limits. Top-level host
statements must retain the current static discovery boundary. Runtime imports,
mutable captures, host effects and untyped calls cannot silently become constants
or disappear. Source/dependency identities, remote packaging and ordinary build
ownership must include the exact reachable helper graph before promotion.

The [resolver candidate](resolver.py) now normalizes the reachable local source
graph to hygienic same-module helpers with canonical per-module annotations and
intrinsics. `candidate-contract.json` retains original source paths/hashes and
helper origins. No entry/helper/package top-level statement is executed. Existing
helper type/effect/flow validation still applies before normalization.

The [candidate probe](candidate-offline.json) passes **108 complete executions**:
three source routes × four levels × three lengths × PTX/normal/poisoned SASS.
Independent modulo64 and exact representable FP32 formulas compare every output,
input and guard byte and inactive tails. Six controls reject recursive calls,
captured globals, host calls, missing modules, local target shadowing and helper
rebinding. This is a hypothesis test. Preserved source diagnostics, import graph
edge cases, package initializer/dependency identities, build/ownership/remote
integration and original-source GPU/toolkit validation remain required. It is
not yet a validated production feature. The [toolkit control](oracle-probe.json)
accepts the exact normalized three-entry PTX and retains its cubin/listing hashes;
this does not establish original-source graph integration or GPU execution. The canonical scope is the
[helper dossier](../../../docs/compiler/gaps-python.md).

```sh
python compiler_exp/gap_closure/python_module_helpers/baseline.py
python compiler_exp/gap_closure/python_module_helpers/probe.py
```
