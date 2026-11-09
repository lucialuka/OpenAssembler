# Reuse packed inference weights instead of converting them on every call

Status: tested-not-selected

This application-level cache was not integrated into the current native
compiler path. Both OA and ptxas benefit from cached conversions, and the
matched cached ptxas path remains faster; this is not an OA compiler speedup.

This is a validated experiment under `exp/`, not an enabled compiler feature.
It improves part of inference without changing the GEMM arithmetic. The same
technique helps the ptxas and cuBLAS controls, so its benefit is not evidence
that OpenAssembler's SASS scheduler has overtaken them.

## Idea and exact contract

The round-two networks keep master parameters in FP32. Before every matrix
product, they convert both operands to FP16 and make the required layout
contiguous. GEMM accumulates and returns FP32. During repeated inference the
linear-layer weights are unchanged, so converting those same weights for every
request repeats memory traffic and launches. Convert each linear weight once,
retain that FP16 tensor, and reuse it. Activation conversion and all matrix
multiplication still execute on every request.

The experimental cache identifies each registered linear weight by its Python
identity, PyTorch version counter, shape, dtype and device. A version change
discards that parameter's old packed value. It does not cache arbitrary input
activations, attention probabilities, embedding tables or optimizer state.
The implementation uses PyTorch's internal `_version` property and remains
experimental; a production integration needs a stable mutation/ownership
contract and broader invalidation tests.

For CUDA graphs, packing happens before capture and the graph uses those fixed
packed buffers. **The weights must remain unchanged while that graph is
replayed.** An eager cache lookup can detect a later version change, but a
previously captured graph does not rerun Python cache logic. Updating weights
requires rebuilding the packed buffers and recapturing or another explicit
safe update policy. This result does not establish a training-cache speedup.

## Experiment and correctness checks

The complete November 7, 2026 run used an **A100-SXM4-80GB**, PyTorch
2.7.1+cu128 and CUDA 12.8. The six custom workloads are small/wide residual ReLU
MLPs, one SiLU-gated MLP and three small decoder-transformer configurations.
They are explicit native-routed research networks, not arbitrary models
automatically compiled by the CLI. Shared CUDA operations still supply
embedding, masks, loss, affine/reduction work as selected, clipping and AdamW.

Twenty-six method/workload combinations independently pass inference and cache
admission. Each passes all five checks:

1. Cached output agrees with its full-output ptxas reference under the
   predeclared inference gate, `rtol=1e-3`, `atol=1e-5`.
2. A real versioned mutation to one linear-weight element invalidates its pack;
   cached and uncached outputs agree after that mutation.
3. Restoring the weight restores agreement with the original reference.
4. The cached graph produces the independently calculated output after the
   input changes, demonstrating that native intermediate launches are captured.
5. Poisoning the final output after capture and replaying the original input
   produces the complete expected output.

The networks' native GEMMs and native output buffers also retain the experiment's
full-output and guard checks. These inference gates are separate from training
state checks: passing cached inference does not admit a method whose multi-step
gradient/parameter/optimizer comparison failed.

## Results

The table uses `oa_tensor_only`: native direct-pointer OA GEMMs, with the
surrounding pointwise/reduction operations shared with the matched
`ptxas_tensor_only` control. Within each workload, cached/uncached host calls
and cached/uncached graph replay are shuffled together. Seven paired samples
each measure five forwards and retain all raw observations. Graph times below
are **microseconds per complete inference**, including required activation
casts, layout work and surrounding operations captured in that graph.

| Workload | Graph uncached µs | Graph cached µs | Uncached/cached | Packed weights |
|---|---:|---:|---:|---:|
| Small ReLU MLP | 60.826 | 56.525 | 1.076× | 256 KiB |
| Wide ReLU MLP | 129.024 | 122.470 | 1.054× | 4 MiB |
| SiLU-gated MLP | 118.784 | 112.435 | 1.056× | 1.5 MiB |
| Single transformer | 186.368 | 176.742 | 1.054× | 448 KiB |
| Multihead/two-layer transformer | 697.139 | 681.165 | 1.023× | 3.125 MiB |
| Longer-sequence transformer | 276.070 | 265.830 | 1.039× | 1.625 MiB |

Host-dispatched inference improves approximately 0.9–3.0% for five of these six
rows; the multihead case is about 0.12% slower. Host spans include enqueue gaps
and allocation behavior, so this small difference is not an isolated GPU
instruction effect. The cached graph still loses to the **equally cached**
same-source ptxas tensor-only graph on every row: ptxas/OA time ratios are about
0.833–0.899. Comparing cached OA to an uncached vendor path would attribute
the shared packing optimization unfairly.

Cold pack plus forward is separately recorded. For `oa_tensor_only`, its six
single observations are 0.622, 0.515, 6.515, 1.595, 5.267 and 2.081 ms in table
order. These include a full first cached forward and have large outliers;
they are neither pure packing costs nor a reliable amortization estimate.
No cold-start speedup is claimed. These are one process run's results; a fresh
independent confirmation and wider shapes are required before promotion.

## Reproduction

Run from the repository root with Python 3.12+, the repository's CUDA toolkit
setup for local builds, and an authenticated Modal client. The remote image
provides the recorded PyTorch/CUDA environment. All generated files remain in
the experiment workspace.

```sh
# Build the scalar/reduction and full-tile GEMM controls without fallback.
.venv/bin/python exp/nn_round2/operator_suite.py --build
.venv/bin/python exp/nn_round2/whole_suite.py build

# Launch the six-workload inference-cache comparison, keeping training-graph
# capture disabled because that is a separately tested experiment.
PYTHONPATH=. modal run exp/nn_round2/campaign.py --phase whole \
  --samples 7 --cache-experiment

# Summarize this retained completed run; a new run gets its own timestamp.
.venv/bin/python exp/nn_round2/whole_analysis.py \
  exp/nn_round2/runs/whole-20261107T144210.754688Z.json
.venv/bin/python -m pytest -q exp/nn_round2/test_whole_suite.py \
  exp/nn_round2/test_whole_analysis.py
```

The exact measured worker SHA-256 is
`f73e9a91938ea1b11454a93b4cbe75d9595ff04e83e7dba695c3bb095c479861`.
Its [content-addressed source archive](../../../exp/nn_round2/source_history/f73e9a91938ea1b11454a93b4cbe75d9595ff04e83e7dba695c3bb095c479861.py.gz)
retains the measured version independently of the current file.
Current experimental source can evolve; a fresh run must retain its own source
and binary identities instead of replacing the old measurements. The cache is
not exposed as a production `openassembler` command-line flag.

## Source, evidence and remaining work

- [Experimental network/cache worker](../../../exp/nn_round2/whole_worker.py),
  [bounded transport](../../../exp/nn_round2/campaign.py),
  [build/precision contract](../../../exp/nn_round2/whole_suite.py).
- [Raw 80GB run](../../../exp/nn_round2/runs/whole-20261107T144210.754688Z.json),
  [correctness-aware derived summary](../../../exp/nn_round2/whole_summary_20261107T144210.754688Z.json),
  [summary implementation](../../../exp/nn_round2/whole_analysis.py).
- [Earlier hybrid MLP contract](native-hybrid-mlp.md) and
  [direct-pointer GEMM](direct-pointer-homes.md) describe separate optimizations.

Next validate independent runs, representative weight sizes, model/device
movement, multiple mutations, parameter replacement and explicit graph
invalidation. Measure amortization with repeated cold starts and actual request
counts. Packing reuse can become an inference frontend/runtime optimization;
it does not repair the remaining HMMA schedule gap, general training numerical
failures or missing native operator coverage.
