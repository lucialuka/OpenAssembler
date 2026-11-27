# Naive scalar attention-score baseline

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: tested-not-selected

The retained exact hardware comparison fails. This is an illustrative source
baseline, not a validated fused attention or training implementation.

## Idea and implementation

One scalar thread computes each entry of `Q@K.T/sqrt(head_dimension)`, looping
over its dot product. Query and key arrays have contiguous `[tokens,dim]`
storage. The test uses 32 tokens and 64 dimensions. The source contains no
softmax, causal mask, multiplication by V, saved activations or backward.
Calling this complete attention would misstate the implemented function.

The baseline is useful for exercising scalar loops, indexing and approximate
reciprocal-square-root scaling. It does not exploit tensor cores, tiled
operand reuse or a fused attention dataflow.

## Test and outcome

The November 2 A100-SXM4-40GB eight-operation study uses explicit buffers/launches,
three shuffled rounds and 20 repetitions. All three bytewise OA/ptxas comparisons
fail; the first reported entry has byte display `cb87023d` versus `c987023d`.
Best-round medians are about 0.02458 ms OA and 0.01946 ms ptxas, but failed
correctness makes these diagnostic timings, not an acceleration result.

The retained experiment does not independently quantify mathematical error or
isolate scaling versus dot-product accumulation as the cause. Approximate
rounding is a hypothesis, not a demonstrated reason to ignore the failed gate.
There is no later admitted full-output accuracy contract for this function in
the retained study.

## Reproduce

From the repository root with Python 3.12+, CUDA 12.8 tools and an authenticated
Modal client or another supported GPU backend:

```sh
compile examples/nn_ops.py --kernel attention_scores \
  --out-dir output_attention_scores
.venv/bin/python tools/nn_study.py --help
.venv/bin/python tools/nn_study.py --dry-run --out-dir output_nn_study
.venv/bin/python tools/nn_study.py --backend modal --budget 180 --rounds 3 --reps 20 \
  --out-dir output_nn_study
```

The study runs all eight operation families; read the attention-score row.
Its dry run builds unless `--skip-build` refers to a prepared output. New
compiler revisions must produce new evidence with their own numerical admission.
Do not compare these tiny scalar scores to a complete fused-attention kernel
without matching all mathematical stages, precision and memory traffic.

## Source and evidence

- [Function](../../../examples/nn_ops.py),
  [explicit study/specs](../../../tools/nn_study.py),
  [raw exact failure](../../evidence/nn-study-a100-20261102.json).
- [Whole-transformer evaluation](whole-transformer-evaluation.md) measures
  attention supplied by the application runtime/Inductor libraries; it does
  not use this scalar example as its attention implementation.

Independent numerical checks, tiled MMA scores, masking, softmax/value fusion,
backward and real model integration remain separate work. The
[cooperative softmax](cooperative-softmax.md) alone does not supply them.
