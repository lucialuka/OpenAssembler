# Static tuple iteration: next G05/G06 research cycle

Status: retained normalized hypothesis. The [original-source promotion](promotion/README.md)
now implements finite tuple control with runtime counter/selection and real
break/continue/exhaustion labels. Earlier records below remain the narrower
experiment, with their source/control limitations unchanged.

The [source](source.py) declares an explicit Final coefficient tuple and uses it
in a pure device polynomial helper. The original frontend still rejects the
`for coefficient in COEFFICIENTS` form. Explicit source values now specialize
correctly; static tuple iteration has its own control-flow and binding semantics.

The [probe](candidate-offline.json) retains the original rejection and a separate
finite unrolling hypothesis. It replaces a homogeneous tuple of i32 literals
with copied assignments and scalar helper statements in a normalized source
file, without executing the user module. An independent Horner polynomial modulo
2^64 passes **twelve complete executions** across four levels and PTX/normal/
poisoned SASS. Every output/input/guard/tail byte is compared. The normalized
source has a distinct dependency identity and is not a production feature.

The candidate explicitly rejects break/continue/return, nested loops, loop else,
heterogeneous/wider iteration values and tuples longer than 64; it cannot stand
in for their intended Python semantics. The next cycle must implement original
source identity, helper/annotation/loop-carried binding, empty tuples, nested
control, early exits and expansion limits, then independent original-source
hardware/build/remote validation before promotion. General iteration and G05/G06
remain open. The canonical current language is the
[static-value dossier](../../../docs/compiler/gaps-static-values.md).

```sh
python compiler_exp/gap_closure/python_static_loops/probe.py
```

The [control matrix](control-matrix.json) now retains eight original-source
rejections and independent expected values for plain/empty iteration, break,
continue, nesting, post-loop target values and loop else. These are intended
semantic contracts for the next implementation, not passing execution results.
`control_matrix.py` records every original source and deterministic reference
value so a future implementation cannot silently specialize away observable loop
control or stale target behavior.
