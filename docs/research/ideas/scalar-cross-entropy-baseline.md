# Serial scalar cross-entropy baseline

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: tested-not-selected

The example source exists and compiles, but the retained November 2 hardware
run fails its declared byte-exact ptxas comparison. No newer admitted independent
accuracy contract is recorded for this kernel. It is not an approved optimized
cross-entropy path merely because it is present in `examples`.

## Idea and implementation

One thread handles an entire row of logits. It finds the maximum, sums
exponentials of shifted logits, and returns
`log(sum(exp(x-max))) + max - x[label]`. This avoids direct exponential overflow
for ordinary finite inputs. It is a serial work mapping, useful as a small
compiler baseline rather than a cooperative training loss kernel.

The source has explicit label and dimension arguments; the experiment uses
128 rows of width 128 and valid deterministic labels. It is not fused with
softmax backward, gradient scaling or a complete training loss interface.

## Test and rejected result

The retained A100-SXM4-40GB study runs eight NN kernel families, three shuffled
rounds and 20 repetitions with explicit buffer/launch specs. Cross-entropy
fails the byte gate in every round, including `loss[4]`:
OA bits `11278d40` versus ptxas `10278d40` in the runner's byte display.
Best-round timing is approximately 0.05837 ms OA versus 0.04096 ms ptxas, but
**the failed gate makes that an inadmissible acceleration result**.

One-bit floating differences may reflect expression contraction or approximate
math, but this record alone does not establish the cause or an acceptable
mathematical error bound. A new test must declare its reference/tolerance
before timing; it cannot retroactively relabel this exact failure as a pass.

## Reproduce

From the repository root, use Python 3.12+, CUDA 12.8 tools and an authenticated
Modal client or compatible GPU backend:

```sh
compile examples/nn_ops.py --kernel cross_entropy --out-dir output_cross_entropy
.venv/bin/python tools/nn_study.py --help
.venv/bin/python tools/nn_study.py --dry-run --out-dir output_nn_study
.venv/bin/python tools/nn_study.py --backend modal --budget 180 --rounds 3 --reps 20 \
  --out-dir output_nn_study
```

The dry run still prepares/compiles the source unless `--skip-build` is used
with an existing valid output tree. The hardware command runs the full eight
family study; inspect the cross-entropy row and retained gate. A current run
uses the changed compiler and records new evidence; its results cannot simply
be substituted for the old snapshot.

## Source, evidence and remaining work

- [Baseline function](../../../examples/nn_ops.py),
  [explicit specs and timing driver](../../../tools/nn_study.py),
  [raw failed result](../../evidence/nn-study-a100-20261102.json).

Independent full-output FP64-reference checks, label/boundary behavior,
cooperative row reduction, backward and whole-loss integration remain open.
The [cooperative softmax](cooperative-softmax.md) provides building blocks but
does not implement this loss automatically.
