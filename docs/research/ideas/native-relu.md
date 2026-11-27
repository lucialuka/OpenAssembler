# Native scalar ReLU example

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

The explicit example compiles and passes exact finite-input output checks.
It is a correctness/performance baseline, not an automatically selected fused
model operation or a universal advantage over CUDA libraries.

## Idea, implementation and methodology

Each thread handles one element and writes `max(x,0)`. The Python source uses
the typed frontend, then the normal native scalar compiler. Every timed call
overwrites its output without mutating input, avoiding a changing workload.

The November 6 scalar study checks every output for normal and zero inputs,
an independent FP64 formula, and native output guards. ReLU's gate is exact.
OA and same-PTX ptxas are also bitwise equal on normal inputs. Timing uses 20
shuffled samples per method on the checked nonzero distribution. The public
PyTorch control uses `clamp_min(...,out=...)`, with no artificial output copy.

## Results and limits

Historical A100-SXM4-40GB single-launch event medians:

| Elements | OA ms | Same-PTX ptxas ms | PyTorch CUDA ms |
|---|---:|---:|---:|
| 1,048,576 | 0.02304 | 0.02355 | 0.02150 |
| 4,194,304 | 0.04608 | 0.04608 | 0.03226 |

The small OA/ptxas difference is not a universal improvement. PyTorch is faster
on these sizes, especially the larger memory-traffic-heavy case. The old
November 2 smaller exact-comparison probe also passed, but its best-of-round
timing is not pooled with this later workload.
The primary scalar-suite result also supersedes an earlier timed-input versus
checked-input mismatch. The new data distribution is checked in full before
timing; a valid result from one distribution does not license timing another.

After scalar dependency fixes, a paired pre-fix/current-native regression
retains full bitwise equality and canaries, identical resources and no observed
slowdown. The example is still a standalone kernel; the application frontend
does not automatically insert it into captured graphs. NaN/Inf, signed-zero
policy and other dtypes were not established by the finite benchmark.

## Reproduce

Run from the repository root with Python 3.12+, CUDA 12.8 tools, and a GPU or
authenticated Modal client for hardware:

```sh
compile examples/nn_ops.py --kernel relu --out-dir output_relu
run output_relu --kernel relu --backend modal -n 1048576 --budget 300
modal run exp/whole_stack/native_ops.py --rebuild
.venv/bin/python -m pytest -q tests/test_pyfront.py tests/test_backend_ops.py
```

The normal CLI run is a same-PTX smoke comparison. The retained native-ops
driver supplies independent formulas, guards, matched inputs and all five
operator families at two sizes; select the ReLU rows from its result. A new
run uses the current backend and writes new provenance rather than updating
historical timing estimates.

## Source and evidence

- [Explicit source](../../../examples/nn_ops.py),
  [full scalar study](../../../exp/whole_stack/native_ops.py),
  [primary result](../../../exp/whole_stack/runs/native-ops-20261106T110300.124164Z.json).
- [Old exact-comparison result](../../evidence/nn-study-a100-20261102.json),
  [paired scalar regression](../../../exp/weaknesses/reductions_runs/scalar-regression-20261106T134737.702713Z.json).

Future integration should test fusion with the preceding producer instead of
assuming a separate activation launch is optimal. Such fusion needs its own
numerical and whole-step result.
