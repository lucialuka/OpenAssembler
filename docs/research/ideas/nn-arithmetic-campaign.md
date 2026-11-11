# Native NN arithmetic and composed-network campaign

Status: validated-not-integrated

This is an experimental measurement framework and a set of tested candidates
under `exp/nn_round2/`. Its operator templates, autograd adapters and inference
caches are not a production native training route. Subsequent independent
GEMM work did promote exact shape-specific pointer/tile/address/traversal plans;
those later integrations are documented in the
[dispatch dossier](shape-aware-native-gemm-dispatch.md), rather than inferred
from this November 7 network campaign.

## Question and boundary

Can native OpenAssembler generate faster GPU work than NVIDIA's tools, and do
isolated improvements survive packing, backward computation and optimizer work?
The November 7, 2026 campaign tests both questions. The target is a goal, not an
assumed outcome: **the measurements do not establish general native OA
superiority over ptxas or cuBLAS**.

“All transformer arithmetic” needs an explicit boundary. The inventory contains
23 broad families, while the generated scalar/row suite contains 26 named
operators. These are different ways of counting coverage. The campaign tests
ordinary dense transformer arithmetic, including selected derivatives; it does
not implement every neural-network operator, dtype, distribution, layout or
training policy. Unsupported operations remain visible shared CUDA work.

## Controls and numerical policy

Three different comparisons are kept separate:

1. **Compiler control:** native OA and CUDA 12.8 ptxas `-O3` compile the same PTX.
   Same shapes, inputs, outputs and launch geometry isolate code generation.
2. **Library control:** cuBLAS GemmEx computes matched FP16/BF16-input,
   FP32-accumulation/output products. This is a selected library dispatch, not
   exhaustive cuBLASLt tuning. Correctness failure excludes its timing.
3. **Composed networks:** explicit autograd adapters route matrix products and
   selected other operators to native OA or ptxas. Other CUDA operations and
   packing costs stay in the measured workload. These are experimental networks,
   not automatic native compilation of arbitrary PyTorch programs.

Every isolated method must pass complete mathematical outputs, input
nonmutation and both guard regions before timing. Floating references use FP64
formulas on the exact FP32/low-precision inputs; exact integer/one-operation
contracts retain exact gates. Captured outputs are poisoned after capture and
checked after replay. Native tensor words are disassembled, and matching tensor
products also require bitwise equality with their exact-source ptxas reference.
No proven copy, address-WAR, HMMA or synchronization edge is removed.

The scalar studies use bounded finite FP32 inputs. Norm epsilon is `1e-5`;
norm derivatives return input gradients without learned affine-gradient outputs.
Cross-entropy uses fixed label zero and returns per-row loss plus logit gradient;
dropout receives an external mask; rotary arithmetic receives sine/cosine values.
The Adam-like isolated update lacks bias correction and weight decay and is not
the whole network's optimizer. Those limitations are part of the contract.

## Executed individual arithmetic coverage

The first broad run uses A100-SXM4-40GB, PyTorch 2.7.1+cu128 and CUDA 12.8.
It builds 94 native kernels and tests **274 launch/shape configurations**:
**264 pass, ten fail**. All ten failures concern the original tanh-based GELU
derivative in both compilers. Their timing is not admitted.

| Family | Tested work and variants |
|---|---|
| Elementwise arithmetic | Residual add, gated multiply, division, reciprocal-square-root scaling |
| Activations and derivatives | ReLU, tanh-approximate GELU, SiLU, sigmoid, ReLU/GELU/SiLU/tanh backward |
| Training support | Externally masked dropout scaling, rotary pairs, SGD momentum, Adam-like update |
| Integer addressing arithmetic | Exact unsigned division/remainder by seven, including high-bit edge cases |
| Cooperative row work | Softmax and backward; population LayerNorm and backward; RMSNorm and backward; row sums; fixed-label cross-entropy plus gradient |
| Source/launch choices | One element/thread, four scalar loads/thread, four-wide vector loads/stores; 128/256 threads; four/eight row warps; width/row tails |

Some individual OA source/launch settings beat their matching ptxas setting.
For example, the first scalar-four ReLU result is about 1.50× faster at one
million elements. After choosing the best tested vectorized setting for both
compilers, both are 5.184 microseconds in that run. That is not evidence of a
general compiler advantage. Several reductions remain substantially slower.
Cross-source layout variants in this initial sweep use different random seeds;
OA/ptxas within each case share identical inputs. Follow-up candidate studies
repair that limitation with a stable seed per operation/shape across variants.

## Improvements tested after the first measurements

Each idea has its own canonical dossier with exact numbers, methods and commands:

| Idea | Evidence and decision |
|---|---|
| [Fused backward local reductions](fused-backward-reductions.md) | 240/240 initial cases, 16/16 separate-process selected confirmations, and 64/64 standard-width cases pass. At 4096 rows, standard widths 512/1024 gain about 1.16–1.31× over earlier OA forms. Smaller row counts can lose; same-source ptxas remains faster. |
| [Stable GELU backward](stable-gelu-backward.md) | New equivalent analytic form passes 26/26 cases under the original gate; direct-tanh failures remain recorded. This repairs accuracy rather than establishing a ptxas speed win. |
| [Earlier GEMM pointer updates](early-gemm-pointer-advance.md) | The November 7 wide study gains about 7.0–7.6% over legacy OA, or 1.5–2.3% over direct mode; tall controls regress. All 84 complete and 28 graph checks pass. Later exact dispatch promotes independently validated complete early-pointer plans; this historical experiment alone is not the promotion evidence. |
| [Joint existing tensor options](joint-tensor-options.md) | Direct+Morton is mostly tied/slower; direct+fusion has limited conditional gains. The main 18-case tensor study passes 402 complete and 134 graph checks. |
| [Immutable inference weight packing](inference-weight-packing-cache.md) | 26 method/network pairs pass accuracy, input-change graph checks and versioned eager invalidation. OA graph inference gains about 2.3–7.6% relative to its uncached form; equally cached ptxas remains faster. |

These are separate workloads and some share evidence. Do not add the ratios to
predict a combined network speedup, count each checked element as independent
evidence, or treat within-process order rounds as independent hardware trials.

## Whole forward/backward/optimizer measurements

The six configurations span a residual ReLU MLP at two sizes, a SiLU-gated MLP,
and three small decoder-transformer configurations. The transformer includes
embedding/position lookup, QKV and output projections, explicit causal attention,
softmax, RMSNorm, GELU feed-forward work, residuals and a vocabulary projection.
Training includes cross-entropy or MSE, raw/clipped gradients, clipping and AdamW.
They are three architecture families/configuration groups, not six pretrained
production models. There is no KV-cache decode or convergence claim.

| Configuration | Resident tokens / width / hidden | Layers / attention |
|---|---|---|
| ReLU MLP small | 128 / 128 / 512 | One residual layer |
| ReLU MLP wide | 512 / 512 / 2048 | One residual layer |
| Gated MLP | 256 / 256 / 1024 | One SiLU-gated residual layer |
| Transformer single | 128 / 128 / 512 | One layer, one head, sequence 128 |
| Transformer multihead | 256 / 256 / 1024 | Two layers, two heads, batch two, sequence 128 |
| Transformer long | 256 / 256 / 1024 | One layer, two heads, sequence 256 |

Parameters and non-GEMM arithmetic use FP32. Every product rounds both operands
to FP16, accumulates/outputs FP32, and uses an explicitly prescribed
straight-through derivative for that rounding. This is not unmodified autocast.
The full AdamW policy is `lr=1e-4, eps=1e-8, weight_decay=.01`, with clipping at
one. Packing, transposes, casts, saved activations and launches are included.

Two reference groups are necessary to attribute changes:

- `oa_legacy`/`oa_direct` compare with `ptxas`: selected pointwise/row operators
  also use their respective native compiler. The failed isolated GELU-backward
  template is excluded; its derivative remains shared CUDA.
- `oa_tensor_only` and `cuda_library` compare with `ptxas_tensor_only`: every
  surrounding operation is the same Torch CUDA work, and only GEMM code changes.

The complete original 40GB round has 36 method/configuration pairs. Twenty-four
pass the initial full-state gate, and **21 remain admitted after the timed
trajectory**. Native GEMM-only OA passes all six networks. The fully native-routed
operator combinations fail some initial or later state gates; a passing
standalone formula does not make the whole optimizer trajectory equivalent.
No tolerance, optimizer epsilon or policy was changed to admit those failures.

Host-dispatched training ratios vary around parity and depend on the run.
Inference graphs expose an OA tensor-only disadvantage of roughly 11–18% in
the first complete cohort. A separate 80GB cache cohort uses its own within-run
comparisons and cannot be pooled in absolute time with 40GB measurements.
Both hosts and GPU kernels matter, but apparent host-span wins do not override
the numerical gates or prove better SASS scheduling.

Independent inference checks allow an inference result to pass even when that
method's training state fails. Changed inputs plus output poisoning verify that
intermediate native launches participate in a captured graph. Weight-cache
graphs require immutable weights and recapture after weight changes.

## Counters and unresolved training capture

[Counter profiling](tensor-counter-profiling.md) records matched scalar/tensor
profiles and a four-profile FMA follow-up. A wide-CSV parser initially rejected
real measurements; all twelve original child output proofs pass, and recovery
parses the preserved raw data without rerunning hardware. Unavailable metrics
are marked unavailable, never zero.

The tall ptxas m-fast/Morton contrast has a large DRAM/L2 difference, consistent
with traversal locality. It is not proof of an operand-collector bottleneck.
NCU replay duration, unlocked clocks and cache conditions are different from the
paired graph-timing protocol and are not substituted for performance samples.

Experimental full training-graph capture has retained stream-dependency failures.
Some complete small-network graph comparisons pass; later captures can fail
with a legacy-stream/capturing-stream dependency. Fresh input poisoning and
multiple consecutive update checks are required before admission. There is no
complete six-network training-graph result yet. This is a separate integration
limitation from kernel numerical correctness and from successful inference
graphs. Failed and partial attempts remain in `runs/`.

## Reproduce

Run from the repository root. The compiler environment needs Python 3.12+ and
the CUDA 12.8 oracle tools; the Modal client can be a separate compatible Python
installation. An authenticated Modal account and paid A100 access are required
for GPU commands. Each worker is bounded and requests one GPU; invoking several
drivers concurrently requests several GPUs.

```sh
# Local checks and build; no GPU requested by these commands.
PYTHONPATH=. .venv/bin/python -m pytest -q exp/nn_round2
PYTHONPATH=. .venv/bin/python exp/nn_round2/operator_suite.py --build
PYTHONPATH=. .venv/bin/python exp/nn_round2/tensor_suite.py build --profile full
PYTHONPATH=. .venv/bin/python exp/nn_round2/whole_suite.py build

# Individual arithmetic and all six custom networks.
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase operators --samples 11
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase whole --samples 9
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase whole --samples 7 --cache-experiment

# Replace BUILD_MANIFEST with the path printed by the tensor build.
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase tensor \
  --payload BUILD_MANIFEST --rounds 3 --samples 11

# Summaries are read-only and request no GPU. Supply a newly completed run file.
PYTHONPATH=. python3 exp/nn_round2/whole_analysis.py \
  exp/nn_round2/runs/whole-20261107T143549.109468Z.json
PYTHONPATH=. python3 exp/nn_round2/operator_analysis.py \
  exp/nn_round2/runs/operators-20261107T143128.366417Z.json
```

The individual idea dossiers own candidate-specific build and execution commands.
`--training-graph-experiment` is an explicit diagnostic option, not a required
passing reproduction or an advertised general-purpose execution path. A replay
uses its actual current source identities; historical raw records retain the
recorded versions and failures rather than being silently replaced.

## Evidence and limits

- [Source and driver index](../../../exp/nn_round2/README.md),
  [prospective inventory](../../../exp/nn_round2/coverage.json),
  [observed coverage](../../../exp/nn_round2/coverage_execution_20261107.json).
- [Exact GPU-worker source history](../../../exp/nn_round2/source_history/README.md)
  retains every recorded operator, tensor, whole-network and counter worker
  revision. Some superseded local orchestration/helper versions are explicitly
  unavailable; the archive does not claim those have been reconstructed.
- [Initial operator run](../../../exp/nn_round2/runs/operators-20261107T143128.366417Z.json),
  [operator analysis](../../../exp/nn_round2/operator_analysis_initial.json).
- [Tensor run](../../../exp/nn_round2/runs/tensor-20261107T143201.758879Z.json),
  [fixed-holdout summary](../../../exp/nn_round2/tensor_summary_20261107T143201.json).
- [Complete native-routed cohort](../../../exp/nn_round2/runs/whole-20261107T143549.109468Z.json),
  [admission-aware summary](../../../exp/nn_round2/whole_summary_20261107T143549.109468Z.json),
  [independent inference/cache cohort](../../../exp/nn_round2/runs/whole-20261107T144210.754688Z.json).

Atomic scatter/embedding backward, random-number generation, full affine norm
training interfaces, arbitrary tensor tails/transposes, production fused
attention, distributed collectives, BF16 whole-training repair, TF32 and
long-horizon convergence remain open. The matrices are small enough that host
and launch work can dominate; isolated large square/tall GEMMs provide a
different compute/cache stress test, not a replacement for whole training.
