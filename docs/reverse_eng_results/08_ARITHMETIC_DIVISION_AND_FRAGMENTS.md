# Arithmetic fragments and division lowering

## ptxas arithmetic fragments

For operations that need exceptional handling or precise rounding, the compiler
retains measured ptxas instruction sequences as fragments. Examples include
division, reciprocal/square-root helpers, conversions and selected floating
special cases. A fragment is not copied blindly: register roles, predicates,
rounding and exceptional paths must remain explicit to the allocator and
scheduler.

## Constant division and remainder

The implemented 32-bit constant divider replaces a reciprocal-style sequence
with proven multiply-high, shift and correction operations. The survey covered
17 divisors and division/remainder forms. Formal and edge-value checks cover
the supported domain. A historical division-by-seven loop improved from
0.629 ms to 0.464 ms.

The result is limited:

* dynamic divisors use a different path;
* 64-bit division is not implied;
* signed and unsigned formulas are distinct;
* a CUDA frontend may already strength-reduce a constant before OpenAssembler
  sees it.

## Numerical contract

FMA contraction, rounding modes, FTZ behavior, NaNs, infinities and subnormals
must be part of the operation contract. A mathematically equivalent formula can
still differ by ULPs. The early NN campaign kept exact/tolerance failures visible
instead of widening gates after the fact.

See [Recovered arithmetic instruction sequences](../research/ideas/arithmetic-instruction-selection.md),
[Constant division](../research/ideas/constant-division.md) and the retained
fragment library in [`openassembler/backend`](../../openassembler/backend).

