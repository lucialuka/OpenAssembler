# Whole-stack and native-stage evidence

The complete original cross-cutting study is
[WHOLE_STACK_RESULTS.md](../../docs/research/WHOLE_STACK_RESULTS.md). Per-feature
methods, commands and limitations now have canonical dossiers:

| Subject | Canonical dossier | Principal drivers |
|---|---|---|
| Synthetic transformer suite | [Whole-transformer evaluation](../../docs/research/ideas/whole-transformer-evaluation.md) | `run.py`, `model.py`, `analyze.py` |
| Configured CLI capture/replay | [Application capture](../../docs/research/ideas/pytorch-application-capture.md) | `cli_proof/prove.py` |
| Native forward/dX/dW MLP | [Hybrid MLP](../../docs/research/ideas/native-hybrid-mlp.md) | `native_mlp.py` |
| Native tensor counters | [Counter profiling](../../docs/research/ideas/tensor-counter-profiling.md) | `native_counters.py` |
| Serial scalar stages | [ReLU](../../docs/research/ideas/native-relu.md), [GELU](../../docs/research/ideas/native-gelu.md), [Adam-like update](../../docs/research/ideas/fused-adam-like-update.md) | `native_ops.py` |
| BF16 sensitivity | [Adam epsilon diagnostic](../../docs/research/ideas/adam-epsilon-sensitivity.md) | `bf16_diagnostic.py` |
| Per-update state trajectory | [Complete-state diagnostics](../../docs/research/ideas/state-verification-diagnostics.md) | `first_divergence.py`, `trajectory_gate.py` |

`runs/` retains successful, failed and superseded JSON, and `native_static.json`
retains the native instruction inventory. The dossiers label which records are
admitted; a JSON timing beside a failed gate is not a speedup. Historical
snapshot manifests and superseded prose remain traceable through
[documentation provenance](../../docs/evidence/documentation-history/README.md).
Raw source/measurement fingerprints are retained. This directory no longer
contains duplicate native/attempt/operator narrative reports.

## Untimed first-observed-divergence diagnostic

The original BF16 diagnostic checks parameter/gradient details at updates 1 and
29. It cannot identify the first failing update. The new diagnostic observes
an initial snapshot and five boundaries at every update: full pre-update
logits, forward loss, raw backward, clipped backward, and post-update state.
It clones all named parameters/buffers, explicit missing gradients, Adam
moments/step counters and optimizer group settings. The first 29 batches match
the old diagnostic; update 30 repeats its last batch. Epsilon remains `1e-8`
and the original `rtol=.05, atol=.005` gates remain fixed.

```sh
modal run exp/whole_stack/first_divergence.py --steps 30
python3.13 -m pytest -q exp/whole_stack/test_trajectory_gate.py \
  tests/test_torchfront_numerics.py
```

This compares eager PyTorch with AOTAutograd/Inductor. It does not execute
native OA-generated kernels or time a workload. Extra full-logit forwards and
state copies occur only in this diagnostic. A first failed boundary identifies
the earliest observed gate failure, not the exact operation that caused it.
The streaming ledger can also be used by a native experiment with that
experiment's own comparator and fixed numerical policy. Incomplete, duplicate
or reordered checkpoint sequences cannot pass. It does not claim unobserved
Python attributes, stochastic RNG histories or every intermediate activation
are equivalent.
