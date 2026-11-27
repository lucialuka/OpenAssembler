# Fused Adam-like scalar update example

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

The explicit one-kernel formula passes its declared accuracy gate. It is not
complete bias-corrected AdamW and is not automatically used by captured models.
Its fusion win must not be presented as beating a tuned fused optimizer.

## Idea and formula

For each element, compute updated moment estimates and a weight output together:

```text
mn = 0.9*m + 0.1*g
vn = 0.999*v + 0.001*g*g
out = w - 0.001*mn / (sqrt(vn) + 1e-8)
```

The kernel returns `out`, `mn` and `vn` in separate buffers. It does not mutate
the input arrays, so repeated timing sees the same workload. This combines
several elementwise operations into one launch and avoids the intermediates of
the chosen eager expression. Bias correction, step policy and decoupled weight
decay are absent; the whole-transformer study uses shared real AdamW instead.

## Methodology and results

The November 6 study uses positive second-moment inputs, normal/zero patterns,
full FP64-reference outputs, three output buffers and native canaries.
The predeclared gate is `rtol=2e-5, atol=2e-6`; all measured outputs pass.
The old November 2 byte-exact gate failed for weight and moment entries. Exact
equality is still not promised: the later tolerance result answers a different,
explicitly declared question.

Historical A100-SXM4-40GB medians over 20 shuffled samples per method:

| Elements | OA fused ms | Same-source ptxas fused ms | Eager PyTorch expression ms |
|---|---:|---:|---:|
| 1,048,576 | 0.03994 | 0.03584 | 0.12288 |
| 4,194,304 | 0.11110 | 0.10138 | 0.36864 |

The eager control launches multiple operations; ptxas's same fused algorithm
is faster than OA. Thus the large gain is fusion versus that expression, not
superior native instruction scheduling or an optimizer-library result.
Single-launch event timing includes enqueue/allocation effects where relevant.
An earlier suite artificially copied the PyTorch outputs, and a subsequent
attempt timed different nonzero inputs from those validated. Both are superseded;
the primary result has no added copies and times the same checked distribution.
The recorded maximum absolute OA weight error is about `2.38e-7`; normal
OA/ptxas outputs are tolerance-equal, not bitwise-equal.

The later paired pre-fix/current OA regression passes bitwise on its valid
inputs, with unchanged resources and no observed slowdown. This does not
change the formula into AdamW or repair the sensitive BF16 training trajectory.

## Reproduce

From the repository root, use Python 3.12+, CUDA 12.8 tools and an authenticated
Modal client for the hardware study:

```sh
compile examples/nn_ops.py --kernel adam_step --out-dir output_adam_like
modal run exp/whole_stack/native_ops.py --rebuild
.venv/bin/python -m pytest -q tests/test_pyfront.py tests/test_backend_ops.py
```

Use the explicit benchmark/spec rather than the generic pointer auto-harness:
it guarantees positive `v`, correct buffers/launch and checks all three outputs.
The native-ops driver runs all five operator families; select `adam_step` rows.
Do not call this a benchmark against `torch.optim.AdamW(fused=True)`; that
comparison was not performed.

## Source and evidence

- [Formula and ABI](../../../examples/nn_ops.py),
  [matched driver](../../../exp/whole_stack/native_ops.py),
  [primary result](../../../exp/whole_stack/runs/native-ops-20261106T110300.124164Z.json).
- [Earlier exact mismatches](../../evidence/nn-study-a100-20261102.json),
  [paired scalar regression](../../../exp/weaknesses/reductions_runs/scalar-regression-20261106T134737.702713Z.json).
- [Full-training epsilon sensitivity](adam-epsilon-sensitivity.md).

A complete fused training optimizer, mixed-precision state policy and whole-step
integration need their own matched accuracy and performance tests.
