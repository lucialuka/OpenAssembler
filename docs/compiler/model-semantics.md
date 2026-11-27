# Masked attention, indexed models and persistent training state

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented within the contracts below, November 20, 2026. Research
candidates preceded production changes. The [protocol](../../compiler_exp/model_semantics/PROTOCOL.md)
fixes numerical policy and gates; [source research](../../compiler_exp/model_semantics/sources.json)
records upstream references, retrieval failures and inspected official PyTorch
source identities. No vendor GPU compute fallback is introduced.

## Supported semantics

| Feature | Admitted contract | Explicit boundary |
|---|---|---|
| Broadcasting | FP32 add/multiply with trailing-axis broadcasting, rank 0..4; backward reduces each replicated dimension to its original operand shape | No implicit dtype conversion |
| Batch-broadcast products | Rank 2..4 matmul normalizes compatible batch axes through copies/BMM and reduces replicated gradients; [contract and A100 evidence](gaps-model.md) | Exact positive shapes and existing resource/half-recipe bounds; `bmm` still requires equal batches; extra copy cost is not a speed claim |
| Attention | `F.scaled_dot_product_attention`, FP32 rank 3/4, matching batch/head axes, rectangular Q/K/V; Boolean broadcast mask (`True` participates), or upper-left causal mask; explicit finite scale or default inverse square root | Width 1..1024; dropout zero, no GQA or additive floating mask; existing matrix resource limits remain |
| Fully masked rows | Zero probability, output contribution and score gradient | General softmax NaN/Inf semantics are unchanged |
| Embedding | `nn.Embedding` / `F.embedding`, external contiguous int64 indices, FP32 table, dense deterministic duplicate-index gradient | No padding index, max-norm mutation, sparse gradients or frequency scaling |
| Cross-entropy | Rank-two FP32 `[rows,classes]` logits, int64 `[rows]` labels, stable log-sum-exp, unweighted mean; classes 1..1024 | No ignored labels, class weights, label smoothing or probability targets |
| SGD extensions | Momentum/dampening/Nesterov, coupled decay and maximize; [contract and evidence](gaps-optimizers.md) | Direct execution for nondefault policies; no parameter groups |
| Adam | FP32 master weights/moments, bias correction, coupled weight decay, configurable betas, positive epsilon outside square root; AMSGrad/maximize via [extended policy](gaps-optimizers.md) | No differentiation through updates or parameter groups; AMSGrad uses split updates |
| AdamW | Same state with decoupled weight decay | No silent epsilon/precision changes; fusion is an explicit option |
| KV cache | `DecodeSession`: OA prefill and exact positive past-length variants, with optional explicit bounded specialization; one-token append; two immutable contiguous FP32 cache inputs and outputs | Copies the prefix; no empty-prefix bucket, paging, in-place append or continuous batching |
| Repository route | Ordinary model factory/config, integer/Boolean input declarations, CE/adaptive policy, source snapshot and saved native replay | Does not resolve arbitrary custom extension launch graphs, arbitrary Python effects or all third-party packages |

Storage/compute/gradients remain FP32. Only index/mask inputs add int64/Boolean
storage. Integer bounds are checked on CPU before any native launch, using a
synchronizing device-to-host transfer when necessary. This is visible host
latency, not a GPU arithmetic fallback. Invalid labels/indices reject before
parameter updates. The plan validator re-derives embedding domains from the
executable graph; deleting or widening metadata does not bypass bounds checks.
Inputs are still exact shape, contiguous and device matched.

Attention decomposes into incumbent OA BMM, scale, masked warp softmax and BMM.
Backward differentiates this decomposition and never differentiates indices or
masks. Fused mask handling avoids a separate mask materialization pass; row
reductions retain the full-warp/tail discipline. No async-copy, LDSM or HMMA
schedule is changed by this work. The new hardware evidence is FP32; it does
not establish new half-precision attention training trajectories.

## Training, checkpoints and caches

`Runtime.train_step` and `CompiledModel.native_train_step` select the saved
`loss` and `optimizer` policies. Existing MSE/plain-SGD defaults remain. The
runtime offers `optimizer_state_dict()` and `load_optimizer_state_dict()` for
CPU-owned moment/step snapshots. Save and restore parameter tensors separately
at the same completed-update boundary. Loading requires initialized storage,
no outstanding tape/gradients, identical policy and complete finite moments
with nonnegative second moments. All validation precedes replacement; resumed
moment allocations have canaries. Eight-update/resume tests compare every
weight, gradient, first moment, second moment and step against PyTorch CPU.

The later [complete checkpoint API](gaps-checkpoints.md) saves all captured
parameters/buffers, full optimizer state, CPU RNG and explicit application JSON
in one versioned artifact. Runtime `save_checkpoint`/`load_checkpoint` and module
`save_native_checkpoint`/`load_native_checkpoint` stage and validate complete
replacement bindings at an update boundary, with failure rollback checks.

`DecodeSession(prefill_runtime, buckets, cache_inputs=('past_k','past_v'))`
owns its runtimes/cache bindings. Each program returns prediction, keys and
values in that order. All variants share the prefill parameter owner. Use
`prefill(inputs)`, then `step(non_cache_inputs)`; `reset()` starts a fresh
sequence. By default an unknown past length, caller cache mutation, externally rerun cache
producer or changed model parameters rejects before a cache transition. Returned
tensors borrow runtime storage. Variants must already be compiled; no hidden
code generation occurs during decoding by default. The separate
[explicit bounded specialization policy](gaps-decode.md) admits exact-length
misses through a declared host factory, maximum context and build budget,
with source/state/hardware validation. It adds no automatic padding or serving
policy.

Indexed inputs, cross-entropy and nondefault optimizers currently require direct
execution. CUDA graph training replay rejects them explicitly: CPU bounds
validation and host step advancement cannot safely be captured as static GPU
work. The optimizer microbenchmark captures only a fixed-coefficient update,
which does not establish support for full adaptive-training replay.

## Production use

The checked example contains embedding, causal attention, residual branches,
LayerNorm, tanh GELU and a vocabulary projection:

```sh
compile examples/native_language_model --backend modal --verify --no-sass
run examples/native_language_model --backend modal --verify
```

Its [configuration](../../examples/native_language_model/openassembler.toml)
declares `input_dtypes=["i64"]`, `input_ranges=[[0,11]]`,
`loss="cross_entropy"`, `optimizer="adamw"`, explicit betas/epsilon/decay,
and `optimizer_fusion=true`. Input ranges generate verification examples; they
never replace runtime semantic bounds. `compile_model` accepts `loss`, an
`optimizer` options mapping and `optimizer_fusion` for the same library path.
Repository construction, file I/O and reference checking stay on CPU.

## Evidence and promotion

The [split-update candidate](../../compiler_exp/model_semantics/candidate-20261120T140713Z.json)
passes 653 complete tensor comparisons. The
[fused candidate](../../compiler_exp/model_semantics/candidate-20261120T141013Z.json)
passes 671. Both execute on A100-SXM4-40GB with Torch 2.7.1/CUDA 12.8. The
[production hardware gate](../../compiler_exp/model_semantics/production-hardware-20261120T142112Z.json)
passes **779 comparisons / 1,771,766 tensor elements**, including all four
view/reuse combinations, poisoned workspaces, guarded allocations, two changed
forward/backward inputs, repeated indices, widths 1/33/513, rectangular causal
attention, large-offset CE and tiny/zero gradients at unchanged epsilon values
1e-8 and 1e-4. Eight updates per optimizer and resume after update four pass.
All outputs/caches and emitted PTX/cubins/plans are retained in the linked ZIP
archives; reports record compiler/harness/source/binary identities and launches.
No output sampling or tolerance relaxation is used (FP32 rtol=1e-4, atol=1e-5).

The production CLI build and saved replay each pass 304 full-field checks over
eight updates, with workspace guards, profiler/ledger agreement, zero vendor
compute and zero replay code generation. See the [final CLI record](../../compiler_exp/model_semantics/production-cli-20261120T142424Z.json).
The [pinned Torch 2.7.1 regression](../../compiler_exp/model_semantics/production-tests-20261120T143351Z.json)
passes 215 tests, including cache-ownership and invalid-checkpoint checks.
Its four warnings concern missing pytest mark registration in the transported
fixture; marked tests execute normally. The no-Torch full suite
passes 1,075 cases with eight optional scopes skipped. The earlier exact division
emulator counterexample remains a known limitation even when the random suite
passes; this work does not repair or relax it.

The first CPU run accidentally imported the incumbent package and failed the
new positive cases. A second regression run exposed absent overlay encoding
data, a fake-tensor fixture incompatibility, and an obsolete broadcasting
rejection assertion; these were corrected without weakening mathematical
gates. The first expanded production hardware run failed on a harness-only
`bias` versus `parameter:bias` lookup; its JSON/archive remain retained. It is
not counted as a successful gate. A first pinned-regression transport omitted
`tests/backend_harness.py`; a subsequent connection attempt timed out before
execution. Both logs remain visible, and the corrected pinned run above passes.
The [artifact/source audit](../../compiler_exp/model_semantics/audit.json) checks
every retained archive hash and proves post-hardware docstring updates preserve
the frozen executable Python AST for all twelve modified production modules.

## Memory and latency decisions

Arithmetic broadcast indexing avoids expanded operand buffers. The separate
[batch-broadcast product normalization](gaps-model.md) materializes expanded
matrix operands and retains that cost explicitly. Backward sums only
replicated coordinates, in deterministic order; large reductions still need
parallel/split reduction studies. Embedding dW deliberately scans token positions
per table element, avoiding atomic ordering variability. This is a correctness
baseline with potentially poor large-vocabulary performance, not an optimized
embedding library claim.

Three-launch and fused adaptive updates use the same formulas and policies.
The production-worker microbenchmark records 15 alternating paired CUDA-event
samples, 20 fixed-coefficient graph replays per sample. All three outputs
(weight/m/v) pass before timing:

| Elements | Split median µs | Fused median µs | Split / fused |
|---:|---:|---:|---:|
| 33 | 7.2192 | 5.1200 | 1.410 |
| 4,096 | 7.7312 | 5.0176 | 1.541 |
| 262,144 | 14.8128 | 7.3360 | 2.019 |

Fusion removes two launches and intermediate moment reads. These are optimizer
component measurements on resident inputs; compilation, initialization,
transfers, host validation, changing step coefficients and full training are
excluded. They establish neither end-to-end speedup nor superiority over tuned
vendor/Triton optimizers. The earlier candidate timing is retained separately;
worker variation is not hidden by selecting the best run.

[FlashAttention](https://arxiv.org/abs/2205.14135) motivates online-softmax tiling
to avoid materializing quadratic scores/probabilities.
[PagedAttention](https://arxiv.org/abs/2309.06180) motivates explicit cache-page
ownership and reduced fragmentation. Applying either here still requires native
layout, register/shared-memory budgets, synchronization and full backward/cache
correctness evidence. Paged cache allocation, fused attention/backward,
segmented embedding gradients and adaptive graph replay remain open research;
no paper result substitutes for OA hardware testing. See the
[memory/latency dossier](memory-latency.md) for incumbent positive and negative
pipeline results.
