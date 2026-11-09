# Adam epsilon sensitivity of near-invariant BF16 key bias

Status: tested-not-selected

This is a numerical training-policy diagnosis, not a compiler optimization.
It remains useful to interpret the historical BF16 state failure, but it does
not apply a current OpenAssembler kernel or justify changing production AdamW
settings.

The intervention is a useful numerical diagnostic under separately fixed
training policies. It is not an original-policy compiler repair and is not
promoted as an optimization or production default.

## Why test this idea?

The whole-model BF16 workload has close initial outputs but drifts in key-bias
parameters after repeated AdamW updates. In ideal arithmetic, adding the same
key bias at every token adds the same offset to a query's attention scores;
softmax cancels that offset. This makes the direction nearly invariant, so
tiny finite-precision gradient differences can matter more than its apparent
effect on the model's initial loss.

Adam scales an update using an accumulated squared-gradient denominator plus
epsilon. When gradients are tiny, a small epsilon can allow rounding/sign
differences to produce visible parameter drift. The diagnostic compares
identically initialized eager/compiled models first with epsilon 1e-8, then
with epsilon 1e-4 fixed independently for both methods. It does not change the
gate midway or compare different epsilon values within a matched pair.

## Method and findings

The retained A100-SXM4-40GB run uses FP32 master parameters, BF16 autocast,
TF32 disabled, seed/data held fixed and 29 updates. It examines raw/clipped
gradients, first updates, final parameters and all 524,288 initial logits.
The initial logit relative-L2 difference is 0.002706, maximum absolute 0.015625,
within the declared mixed gate; first losses are 6.3812256/6.3815293.

At the first step, key-bias gradient L2 is about 1.8e-6, versus query 5.8e-4 and
value 8.4e-3. Key-bias gradient signs disagree on 146/512 entries. With epsilon
1e-8 the final maximum key-bias difference reaches 0.006191 and fails the
`rtol=0.05, atol=0.005` parameter gate. In the separately matched epsilon 1e-4
pair, key-bias drift falls to 4.67e-6/7.85e-6 and no parameter tensor fails that
gate.

This strongly supports denominator amplification as an explanation. It does
not prove every implementation difference harmless or preserve the original
training trajectory. Epsilon changes the optimizer policy; reporting the
larger-epsilon pass as a fix for the failed original workload would be invalid.
The production configuration therefore remains unchanged.

## Reproduce

Run from the repository root with an authenticated Modal client; the retained
driver supplies the pinned PyTorch 2.7.1/CUDA 12.8 model/runtime:

```sh
modal run exp/whole_stack/bf16_diagnostic.py
modal run exp/whole_stack/bf16_diagnostic.py \
  --out exp/whole_stack/runs/reproduced-bf16-diagnostic.json
```

The driver saves a numerical sensitivity report, not a valid performance
benchmark. Check each epsilon pair's own parameter gate and retain the declared
policy in any derivative result. A driver that completes is not evidence that
the original BF16 training passes.

## Source and evidence

- [Diagnostic implementation](../../../exp/whole_stack/bf16_diagnostic.py),
  [model](../../../exp/whole_stack/model.py).
- [Raw initial/logit/gradient/update result](../../../exp/whole_stack/runs/bf16-diagnostic-20261106T110747.459824Z.json).
- [Full model numerical failure](../WHOLE_STACK_RESULTS.md),
  [complete-state diagnostics](state-verification-diagnostics.md),
  [failed cast-preservation attempt](bf16-cast-preservation.md).

Longer trajectory/convergence measurements, additional seeds and isolated
gradient-ordering experiments remain open. This intervention supplies evidence
for where to investigate, not permission to discard a sensitive state direction.
