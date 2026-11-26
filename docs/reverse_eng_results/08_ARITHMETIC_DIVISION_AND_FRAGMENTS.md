# Arithmetic fragments and division lowering

The [floating-literal contract](../compiler/gaps-wide.md#decimal-and-raw-floating-literal-contexts)
now normalizes decimal binary64 representation and raw bits before arithmetic
selection. Independent rational controls expose a real direct-to-f32 rounding
counterexample. Native literal moves/stores and exact arithmetic positions pass
320 A100 launches; broader IEEE arithmetic remains a separate fragment gate.
The later [typed-expression route](../compiler/gaps-wide.md#typed-scalar-operand-constant-expressions)
adds integer precedence/casts and binary64 intermediate rounding, with a final
384-launch gate. The first failed report exposes differences across local and
worker ptxas builds; unresolved remainder/conditional/generated-NaN semantics
remain explicit diagnostics, not inferred arithmetic support.

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


## Scalar 64-bit selection from recovered SEL words

The [wide-selection dossier](../compiler/gaps-wide.md#scalar-64-bit-conditional-selection)
records a validated composition for `selp.b64/u64/s64/f64`. Two existing scalar
SEL words select both halves under one predicate; fresh result binding and the
existing guard path preserve snapshots and old values. No new encoding field
was inferred. The original CUDA noinline source and 36 legal PTX controls pass
1,480 A100 launches; independent full-bit audit checks 15,155,200 bytes and 148
current native outputs. Wrongly typed initial FP64 sources remain rejected in
the original evidence. This establishes bit selection, with no FP64 arithmetic
or speed claim.

## Low64 products from recovered IMAD words

The [low64 multiplication dossier](../compiler/gaps-wide.md#low-64-bit-integer-multiplication)
records exact composition from a low32 product, its unsigned high32 carry and
two low32 cross terms. Four recovered IMAD/IMAD.HI words implement both signed
and unsigned low64 products without new fields or templates. Actual original
CUDA, typed PTX and Python source/helper/loop/guard/square cases pass 1,560 A100
launches, 15,974,400 audited bytes and 156 exact native outputs. Unsigned high-word
carry in the decomposition is required even for the signed low64 product.
This establishes correctness under the admitted product contract, with no new
latency/performance or general high/wide-product claim.
