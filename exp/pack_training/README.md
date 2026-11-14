# FP16 packs, reusable training schedules and matrix tails

Experiments start here. Production selection requires two independent A100
runs of complete changed-input training and actual CUDA Graph replay with full
prediction/loss/dX/dW/updated-weight checks, immutable inputs and guard bytes.
FP16 operands use round-to-nearest-even and FP32 accumulation/master state.
No Triton, Inductor, cuBLAS or vendor cubin is an execution fallback.

The candidates are:

- One bounded dual-layout FP32-to-half pack per logical operand per iteration,
  with normal and transposed packed views and fixed zero-padded dimensions.
- A 32×32 shared-memory transpose with stride 33 and coalesced paired global
  stores. The first scattered-store candidate is retained for ablation.
- Static storage bindings and pack reuse across forward and backward. All packs
  refresh at forward; SGD never reuses a previous iteration's parameter pack.
- Padded OA Tensor Core GEMMs with a guarded crop/bias epilogue for odd dimensions,
  small batch sizes and flattened rank-three linear inputs through logical matrix
  dimensions of 512. No broadcasting or
  arbitrary masked batched-attention claim follows.

The launch count, allocation bytes, complete training GPU event distributions,
raw profiler audits and per-case build artifacts are retained. Each workload
runs in its own process to avoid profiler state leaking across workloads.

```sh
modal run exp/pack_training/modal_run.py --phase probe
modal run exp/pack_training/modal_run.py --phase graphs
```

Production measurements, negative cases and usage: [RESULTS.md](RESULTS.md).
