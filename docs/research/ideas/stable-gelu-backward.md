# Stable exponential form for tanh-GELU backward

Status: validated-not-integrated

The candidate is isolated under `exp/nn_round2/`. It improves derivative accuracy
under the original gate and is not a production compiler or CLI feature. The
[implemented forward GELU example](native-gelu.md) has its own separate contract.

## Why change the derivative


The broad arithmetic campaign added tanh-GELU backward. Directly differentiating
with the hardware tanh approximation is **not admitted**: ten initial
kernel/shape/launch cases fail the predeclared analytic FP64 gate on normal
inputs, for both OA and same-PTX ptxas. Zero and structured inputs alone would
have missed these failures. The unchanged gate is `rtol=8e-5, atol=2e-5`.
This is separate from the implemented forward example and does not revoke or
retroactively repair any earlier exact comparison.

An isolated candidate computes the same tanh-GELU derivative more stably.
Writing `u = 0.7978845608028654*x*(1 + 0.044715*x*x)`, it evaluates:

```text
e = exp(-2*abs(u))
r = 1 / (1 + e)
sigmoid(2*u) = r if u >= 0 else e*r
sech(u)^2 = 4*e*r*r
GELU'(x) = sigmoid(2*u) + 0.5*x*sech(u)^2*u'(x)
```

This avoids deriving a small `1-tanh(u)^2` from a tanh approximation near
saturation. Exponentials and reciprocals still use the declared FP32 approximate
instructions. It is the derivative of **tanh-approximation GELU**, not erf-GELU,
and covers the tested bounded finite domain rather than general exceptional
values.

In a fresh A100-SXM4-40GB process, the stable version passes **26/26** complete
kernel/shape/launch/order-round cases for both compilers. Three patterns, complete
outputs, unchanged inputs, both guard regions and poisoned graph replay pass.
The old tanh form, tested with the exact same inputs across all layouts, passes
14 cases and fails 12; those failures remain excluded from timing. The case-count
difference from the initial ten failures reflects the new common input seed and
two rounds, not a changed tolerance.

A separate diagnostic retains exact failing values. On 1,048,576 normal inputs,
the old OA maximum absolute derivative error is about `4.25e-5`; the stable form
reduces it to `6.75e-7`, with zero mismatches for either compiler. For example,
`x=-3.4172425, dy=1.7128356` produces about `-0.00554698` in the old native form,
versus the FP64 reference `-0.00552586`, outside its unchanged mixed gate.
A bounded stress grid with `x` from -12 to 12 and `dy` in
`{-100,-1,0,1,100}` has six failed values per compiler in the old form and zero
in the stable form. All guards pass.

This is a correctness improvement, **not an OA-versus-CUDA speedup**. On the
million-element vectorized candidate, the two order rounds give OA about
6.304–6.336 µs and same-source ptxas 5.792–5.888 µs. Four scalar elements per
thread are substantially slower than the vector-load/store version. The timing
contains 19 shuffled compiler-pair samples and 32 operations per graph; compile,
module loading and host submission are excluded. Failed old-form rows have no
admitted timing comparison.

## Current-source revalidation: November 9

The current Modal runner rebuilt the stable and direct-tanh templates against
this checkout on an A100-SXM4-40GB. It recorded 52 method/shape/launch rows:
40 stable or control rows pass all numerical, canary, input-immutability and
graph checks; the 12 direct-tanh rows at the million-element cases reproduce the
known accuracy failures. Passing stable rows have ptxas/OA ratios from about
0.48 to 0.99, so the stable formula remains slower or roughly tied with the
same-source ptxas control. This validates a correctness repair under its stated
contract, not a production performance win. The source-hashed record is
[`operator-gelu-stable-20261109T063435.427437Z.json`](../../../exp/nn_round2/runs/operator-gelu-stable-20261109T063435.427437Z.json).

The candidate remains in `exp/`; no production GELU-backward operator or automatic
application rewrite is added. Its experimental cubin uses the fixed campaign ABI
of four input pointers, three output pointers and an unsigned element count;
only `X0=x`, `X1=dy` and `Y0=dx` are used. Vectorized cases require an element count
divisible by four and 16-byte-aligned pointers; scalar cases separately check an
odd tail.

Reproduce from the repository root with the same compiler/toolkit prerequisites
as above and a Modal-capable Python client:

```sh
.venv/bin/python -m pytest -q exp/nn_round2/test_operator_gelu_stable.py
.venv/bin/python exp/nn_round2/operator_gelu_stable.py
PYTHONPATH=. python3 -m modal run exp/nn_round2/operator_gelu_stable_modal.py \
  --samples 19 --rounds 2
```

- [Stable formula/source and exact failing-value diagnostic](../../../exp/nn_round2/operator_gelu_stable.py),
  [CPU bounded-domain/guard tests](../../../exp/nn_round2/test_operator_gelu_stable.py),
  [full raw comparison and diagnostic values](../../../exp/nn_round2/runs/operator-gelu-stable-20261107T143841.822588Z.json).
- [Initial 274-case arithmetic campaign, retaining all ten original failures](../../../exp/nn_round2/runs/operators-20261107T143128.366417Z.json),
  [initial interpretation and layout-selection caveats](../../../exp/nn_round2/operator_analysis_initial.json).
