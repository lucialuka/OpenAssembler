# Native tanh-approximation GELU example

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

The standalone example passes its predeclared FP64-reference tolerance on the
measured cases. It is not bitwise equal to ptxas in general and does not replace
every PyTorch GELU automatically.

## Idea and precision contract

The source evaluates the tanh approximation
`0.5*x*(1+tanh(0.7978845608*(x+0.044715*x³)))`, one element per thread.
This is the chosen approximation, not the exact Gaussian-integral definition
of GELU. The native frontend/backend choose and schedule the corresponding
FP32 arithmetic and recovered special-math sequences.

The November 2 byte-exact OA/ptxas comparison failed, including a mismatch at
element 5444. That failure remains a failure of that exact contract. The later
November 6 study declares a separate mathematical accuracy gate **before timing**:
`rtol=2e-5, atol=5e-6` against the same FP64 formula on exact FP32 inputs.
All normal/zero outputs and native guard regions pass. This does not retroactively
make the earlier bitwise result successful.

## Measured result

A100-SXM4-40GB single-launch event medians,20 shuffled samples per method:

| Elements | OA ms | Same-PTX ptxas ms | PyTorch tanh-GELU ms |
|---|---:|---:|---:|
| 1,048,576 | 0.02253 | 0.02355 | 0.02560 |
| 4,194,304 | 0.04659 | 0.04710 | 0.03738 |

The PyTorch control calls its public tanh-approximate out variant; no artificial
copy is added. OA's small-size win disappears at the larger size, and numerical
errors differ. Single-launch event spans can include enqueue effects. Neither
these small differences nor old best-of-round numbers establish a general
activation/compiler advantage.
The recorded maximum OA absolute error is about `5.56e-6`; passing a mixed
tolerance does not make every method's error identical. Two earlier scalar-suite
records are superseded: one added artificial PyTorch output copies, and another
timed a different nonzero distribution from the checked inputs. The primary
November 6 result removes those copies and restores the identical checked input
before timing. Neither earlier timing is pooled into this table.

Updated scalar dependency handling separately passes a paired old/current OA
normal/zero regression with unchanged resources and no observed slowdown.
Approximate-math, finite-domain and tolerance qualifications remain.

## Reproduce

Run from the repository root with Python 3.12+, CUDA 12.8 tools and an optional
authenticated Modal client:

```sh
compile examples/nn_ops.py --kernel gelu --out-dir output_gelu
modal run exp/whole_stack/native_ops.py --rebuild
.venv/bin/python -m pytest -q tests/test_pyfront.py tests/test_pyfront_semantics.py \
  tests/test_backend_ops.py
```

The hardware driver checks every mathematical output, guards and two input
patterns, then times checked normal inputs. It runs five families at two sizes;
read its GELU rows. A generic byte-exact `run` mismatch is not automatically
an independent accuracy failure or a valid speedup; determine which declared
contract is being tested. Do not relax one after seeing a faster candidate.

## Source and evidence

- [Source formula](../../../examples/nn_ops.py),
  [FP64-reference driver/gates](../../../exp/whole_stack/native_ops.py).
- [November 6 admitted tolerance result](../../../exp/whole_stack/runs/native-ops-20261106T110300.124164Z.json),
  [November 2 exact mismatch](../../evidence/nn-study-a100-20261102.json),
  [paired compiler regression](../../../exp/weaknesses/reductions_runs/scalar-regression-20261106T134737.702713Z.json).

Other GELU definitions, backward kernels, dtypes and model fusion remain separate
contracts; this example is not a complete activation-training package.


A separate [stable tanh-GELU backward experiment](stable-gelu-backward.md)
retains the failed direct-tanh derivative gates and validates a more stable
exponential formulation. It remains experimental; it is not part of this
implemented forward operator.
