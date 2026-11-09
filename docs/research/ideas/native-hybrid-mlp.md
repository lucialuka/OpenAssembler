# Native GEMMs inside a complete hybrid residual MLP

Status: tested-not-selected

This was a bounded hybrid application study, not a native OpenAssembler
transformer compiler. Its whole-step comparisons are mixed or slower, so the
candidate wiring was not selected for the current compiler/runtime.

The experimental autograd integration and measurements are validated under a
declared mixed-precision contract. They are not an automatic compiler feature
that substitutes native GEMMs throughout arbitrary models.

## Idea and contract

Measure useful work around matrix multiplication, not only isolated GEMMs.
The network is `x + W2(ReLU(W1(LayerNorm(x))))`. Native OA, same-PTX ptxas or
cuBLAS supply both forward products, both input-gradient products (`dX`) and
both weight-gradient products (`dW`). All methods share FP32 normalization,
activation, loss, master parameters, residual and AdamW CUDA operations.

Operands are rounded to FP16, GEMMs accumulate/output FP32, and backward uses
explicit prescribed packing/transposes. This is a custom mixed-precision
derivative contract, not a claim of equivalence to unmodified PyTorch autocast.
Allocation, conversion, transposed packing and optimizer work are included in
whole-step timing. Merely adding fast GEMMs without those costs would answer
a different question.
The shared AdamW policy is learning rate `3e-4`, epsilon `1e-4`, with its
foreach implementation. This is the MLP's declared policy, separate from the
BF16 transformer's original epsilon-sensitivity experiment.

## Initial full-step result

On A100-SXM4-40GB, the OA/ptxas paths are bitwise equal for complete outputs,
gradients, parameters and AdamW state after three updates. All 108 verification
GEMMs pass full FP32-reference checks and output canaries. Fifteen shuffled
paired samples follow five warmups:

| Batch / width / hidden | Method | Training ms | Inference ms |
|---|---|---:|---:|
| 512 / 512 / 2048 | Native OA | 1.6896 | 0.3635 |
| 512 / 512 / 2048 | Same-PTX ptxas | 1.6916 | 0.3451 |
| 512 / 512 / 2048 | Default cuBLAS GemmEx | 1.6056 | 0.3318 |
| 1024 / 1024 / 4096 | Native OA | 2.1340 | 0.4567 |
| 1024 / 1024 / 4096 | Same-PTX ptxas | 2.0746 | 0.4065 |
| 1024 / 1024 / 4096 | cuBLAS | excluded | excluded |

The tiny small-training OA/ptxas ordering reverses in earlier complete runs;
no reliable OA training win follows. OA inference is slower. Larger cuBLAS
fails the unchanged gradient-relative-L2 gate at `gradient/w1`:
0.006364 > 0.005, despite small absolute error. It remains excluded from that
whole-state comparison; isolated cuBLAS GEMM checks are separate.

The first setup attempt failed before execution with `CUDA_ERROR_INVALID_CONTEXT`:
`torch.cuda.init()` had not materialized a current primary context. The driver
now creates a small CUDA allocation before module loads. A subsequent attempt
completed the small case but aborted on the larger cuBLAS gate. Result handling
was changed to save passing cases and retain failed numerical diagnostics while
omitting their timing admission; neither arithmetic nor the threshold changed.

Separate stage events identify substantial integration work. On the larger
OA MLP, backward packing is about 0.171/0.179 ms per linear versus about 0.09–0.12 ms
per GEMM; instrumented AdamW is about 0.72 ms. Event instrumentation inflates
short stages and stage medians are non-additive. These numbers locate work;
they are not an exact summed application budget.

## Direct-pointer follow-up

Two new device runs compare accurately labelled legacy OA/direct OA/ptxas/
cuBLAS methods, keeping every surrounding operation and derivative fixed.
All 33 state tensors are bitwise equal between direct, legacy and ptxas.
The first run is 80 GB, the second 40 GB; comparisons are within device.

- Larger model: direct mode reduces training time about 1.15–1.31% and inference
  about 2.97–6.01% in the two runs.
- Smaller model: training is neutral/slower by 0.18–0.35%; inference is slower
  by 2.48–5.22%, despite isolated GEMM improvements.
- The larger cuBLAS state failure remains excluded.

This is evidence for modest, workload-dependent application impact, not a
native transformer or tokens-per-second result. The promoted
[direct-pointer option](direct-pointer-homes.md) remains explicit.

## Reproduce

Run from the repository root with Python 3.12+, the local CUDA 12.8 tools and an
authenticated Modal client. The pinned remote image supplies PyTorch/CUDA.

```sh
modal run exp/whole_stack/native_mlp.py --mode build
modal run exp/whole_stack/native_mlp.py
modal run exp/whole_stack/native_counters.py
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_homes.py --mode production --rounds 2
PYTHONPATH=. modal run exp/weaknesses/gemm_pointer_homes_mlp.py
```

The follow-up first needs validated legacy/direct build caches; its MLP driver
checks complete native word hashes and refuses an unvalidated stream. Rebuild
historical controls at the recorded baseline commit if needed. Fresh current
results record their own hashes and must not be silently pooled with old data.

## Source and evidence

- [Original autograd/whole-step driver](../../../exp/whole_stack/native_mlp.py),
  [original full result](../../../exp/whole_stack/runs/native-mlp-20261106T105944.492233Z.json),
  [isolated native counters](../../../exp/whole_stack/runs/native-counters-20261106T110405.921716Z.json).
- [Earlier setup/gate attempt record](../../../exp/whole_stack/runs/native-mlp-20261106T105801.257126Z.json).
- [Four-method follow-up](../../../exp/weaknesses/gemm_pointer_homes_mlp.py),
  [80 GB run](../../../exp/weaknesses/gemm_pointer_homes_mlp-20261106T131316.808136Z.json),
  [40 GB repeat](../../../exp/weaknesses/gemm_pointer_homes_mlp-20261106T131500.693468Z.json).

Packing fusion, real transposed/tail kernels, saved activations, affine norm,
attention and automatic forward/backward integration remain open. They need
the same full-state gate before their timing can count as acceleration.

## Later native arithmetic integration cohort

The [November 7 campaign](nn-arithmetic-campaign.md) adds six explicit networks,
including gated MLP and per-head causal attention, plus independent inference
and cache checks. Its reference groups distinguish native GEMMs with identical
Torch surroundings from native pointwise/norm combinations. GEMM-only OA passes
all six training networks; several all-native operator combinations fail their
initial or post-trajectory state gate. That is a new numerical contract and
cohort, not a replacement for the earlier MLP tables or an admitted whole-native
transformer. Its whole-training graph captures also have retained stream errors.
