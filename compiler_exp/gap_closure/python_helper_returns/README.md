# Pure helper early returns: next G04/G06 research cycle

Status: retained normalized-source hypothesis. The subsequent
[original-source cycle](promotion/README.md) implements typed return paths with
independent original-source/hardware validation.

The [original source](source.py) contains three useful pure scalar helpers:
piecewise classification, return from a range loop and return from nested static
tuple loops. The [probe record](candidate-offline.json) retains all three current
source rejections. User module top-level code is never executed during discovery.

The [normalizer](normalize.py) is an isolated flag/returned-value hypothesis.
It guards statements after possible returns, terminates active loops when a
return flag is set, and ends with one scalar return. A terminal fallback is
required; the candidate does not hide fall-through/missing returns with a default
zero. Arbitrary branch-only complete return sets, original break/continue/loop
else, imported source identity and source diagnostics remain unvalidated.

The [numerical gate](candidate-offline.json) passes **36 complete executions**:
three helper routes × four levels × PTX/normal/poisoned SASS. Independent piecewise,
range-search and nested-search formulas compare complete modulo64 output bits,
guards/tails and unchanged inputs. The normalized source is explicitly a different
source/dependency identity. This is not a production feature or hardware result.

`first-ast-api-failed.log` retains a nonexistent AST comparison class used by the
first prototype. `first-definite-initialization-failed.log` retains an excessive
flag-guard transformation that made an ordinary local appear uninitialized. The
correction guards only statements after a possible return; it does not weaken the
production definite-initialization checker. The next implementation must preserve
original source/type/effect/return-path guarantees and call-argument isolation,
then pass independent original-source GPU/build/ownership/remote gates.

The [toolkit check](oracle-probe.json) accepts the three normalized kernels
and retains exact normalized source/PTX/cubin/disassembly/tool-log identities.
It does not validate the original return paths or original source graph and does
not provide GPU execution evidence.

The canonical current contract is the
[helper dossier](../../../docs/compiler/gaps-python.md). General G04/G06 remain
open beyond the already admitted static helper/tuple control subset.

```sh
python compiler_exp/gap_closure/python_helper_returns/probe.py
```
