# Whole-transformer training and inference evaluation

Status: validated-not-integrated

The evaluation harness and its admitted
measurements are valid research evidence. They do not implement a complete
native OpenAssembler transformer compiler. The measured application frontend
itself is implemented separately in [this dossier](pytorch-application-capture.md).

## Question and comparison

Does the existing application compilation workflow improve an entire model
run, and which stages account for the difference? The experiment compares
eager PyTorch CUDA against OpenAssembler's AOTAutograd/Inductor/Triton worker.
Eager PyTorch already uses optimized CUDA libraries; it is not a deliberately
slow handwritten CUDA baseline. A stock tuned `torch.compile` comparison is
still missing, so the results do not establish an advantage over Inductor itself.

Ten synthetic decoder-transformer configurations share one architecture,
approximately 85 thousand–7 million parameters. The model includes embeddings,
RMSNorm, causal attention, GELU feed-forward blocks and residuals. Training adds
cross-entropy, backward, clipping and AdamW. These are ten configurations,
not ten independently pretrained model families or a large-model convergence test.

## Methodology and findings

The November 6 run uses A100-SXM4-40GB, PyTorch 2.7.1+cu 128 and CUDA 12.8.
Each pair has fresh processes and identical seed/data/source. Two rounds reverse
eager/compiled process order. Training uses eight warmup updates, twenty measured
updates and a separate diagnostic update: 29 updates per method.

FP32 disables TF32. Its output/gradient gate is `rtol=1e-3, atol=1e-4` and
parameter/optimizer gate `rtol=2e-2, atol=2e-3`. BF16 uses `rtol=0.05, atol=0.005`
for both. Initial observed outputs and final state are compared; every
intermediate or complete optimizer trajectory is not checked.

| Configuration | Precision | Eager / compiled ms | Paired eager/compiled ratio | Gate |
|---|---|---:|---:|---|
| Tiny training, 2×32×64, one layer | FP32 | 3.513 / 2.841 | 1.20–1.27× | pass both |
| Base training, 8×128×128, two layers | FP32 | 5.796 / 4.032 | 1.35–1.53× | pass both |
| Long training, 2×512×256, two layers | FP32 | 5.461 / 3.880 | 1.39–1.42× | pass both |
| Wide training, 8×128×512, two layers | FP32 | 5.800 / 4.380 | 1.24–1.42× | pass both |
| Deep training, 4×128×256, eight layers | FP32 | 17.202 / 11.007 | 1.53–1.59× | pass both |
| Single prefill, 1×128×256, four layers | FP32 | 2.004 / 1.171 | 1.68–1.75× | pass both |
| Long prefill, 8×512×256, four layers | FP32 | 4.296 / 3.765 | 1.13–1.15× | pass both |
| Batched prefill, 16×256×512, two layers | FP32 | 5.245 / 4.953 | 1.04–1.08× | pass both |
| Wide training, 8×128×512, two layers | BF16 | diagnostic only | **not admitted** | fail both |
| Batched prefill, 16×256×512, two layers | BF16 | 1.306 / 0.947 | 1.37–1.39× | pass both |

Nine configurations pass in both rounds; the BF16 training speed is excluded.
All four inference configurations also pass exact generated-token comparison.
Decode recomputes a fixed window at batch one; it is not KV-cache serving.

The measured step includes resident-data batch selection, forward/loss,
backward, clipping, optimizer and host enqueue gaps. It excludes imports,
construction, compilation, storage/data loading, `zero_grad` at step start and
final verification copies. Compiled FP32 training program invocations take
roughly 14–28 seconds including compilation/profiling/snapshots versus 1–4
seconds eagerly; these short workloads do not amortize startup.

Separate profiles show base launches falling **207→123**, deep training
**615→367**, and BF16 prefill **74→30**. Standalone pointwise/reduction work
moves partly into fused kernels. These benefits belong to graph fusion and
launch/intermediate reduction. Fused attention includes its matrix products
and softmax; the study does not invent separate costs inside a fused kernel.
Profiles overlap and are non-additive; they are not replacements for step timing.

## Reproduce and inspect

Use an authenticated Modal client. The driver builds the pinned image and
requests one A100 container at a time. Run from the repository root:

```sh
modal run exp/whole_stack/run.py --dry-run
modal run exp/whole_stack/run.py --rounds 2
modal run exp/whole_stack/run.py --case train_base --rounds 2
python3 exp/whole_stack/analyze.py \
  exp/whole_stack/runs/whole-stack-20261106T105757.119747Z.json
python3 exp/whole_stack/analyze.py --selfcheck
```

The analysis commands inspect retained evidence without a GPU. A fresh run
uses today's code, records its own hashes and may not reproduce historical
timings exactly. Two rounds test order sensitivity, not confidence intervals
over machines, datasets or random seeds. An execution marked `completed` may
still have failed numerical gates; use the analyzer's eligibility fields.

## Source and evidence

- [Model](../../../exp/whole_stack/model.py),
  [paired process driver](../../../exp/whole_stack/run.py),
  [profile-aware analyzer](../../../exp/whole_stack/analyze.py).
- [Primary raw run](../../../exp/whole_stack/runs/whole-stack-20261106T105757.119747Z.json),
  [derived summary](../../../exp/whole_stack/runs/summary-whole-stack-20261106T105757.119747Z.json).
- [BF16 cast-repair attempt](bf16-cast-preservation.md),
  [epsilon sensitivity](adam-epsilon-sensitivity.md),
  [native hybrid MLP](native-hybrid-mlp.md),
  [native direct-pointer option](direct-pointer-homes.md).

## Subsequent native-kernel progress

The completed November 11 [42-contract production replay](../../../exp/dispatch_tuning/RESULTS.md)
promotes 34 exact GEMM policies and records eight reference retentions. Five
separately held-out [rowwise contracts](shape-aware-rowwise-dispatch.md) now
select measured CTA warp counts. Neither selector is automatically substituted
into this AOTAutograd/Inductor transformer study. These isolated results do not
change its nine admitted configurations, failed BF16 training gate, timing
scope or missing stock compiled-PyTorch control.

## Remaining work

Add tuned compiled-PyTorch controls, more model families, real KV-cache decode,
larger vocabularies and sustained numerical/convergence evaluation. Integrating
native kernels must include saved activations, packing and backward/optimizer
costs before claiming transformer tokens-per-second acceleration.

The later [native NN arithmetic campaign](nn-arithmetic-campaign.md) addresses
part of that native integration question using explicit forward/backward
adapters across six networks. It is separate from this Inductor study. Matched
GEMM-only OA passes all six full-state training cases, while several native
operator combinations fail and remain excluded. Standard compiled-PyTorch and
production fused-attention comparisons are still outside that new contract.
