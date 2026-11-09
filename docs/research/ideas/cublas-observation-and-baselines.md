# cuBLAS dispatch observation and cuBLASLt controls

Status: validated-not-integrated

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Observe which vendor kernels are selected and compare equivalent public library APIs. Use the findings to understand tile size, resource use and possible optimization opportunities without copying a vendor kernel into OpenAssembler.

## Findings and implementation decision

Research inspected public dispatch/SASS and queried cuBLASLt heuristics. The queried setup returned 10 unique FP16 and 13 BF16 algorithms; selected zero-workspace candidates measured about 0.576/0.564 ms versus then-current OA 0.672/0.655 ms. This was not exhaustive enumeration. Default GemmEx and tuned Lt are separate baselines.

No cuBLAS kernel module was extracted, redistributed or used as an OA implementation. The source observations informed independent kernels. Vendor APIs already expose bias/activation and applicable residual semantics, so the claim that cuBLAS always writes every intermediate to global memory is false.

## Method and validity checks

Match layouts, transpose modes, alpha/beta, input/accumulation/output precision, workspace, epilogue semantics and numerical admission. Record selected algorithm/resources. Validate full outputs before timing and do not combine timings taken at different clocks or from unrelated library dispatches.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
PYTHONPATH=. modal run exp/a100_counters/run.py --mode build-current
PYTHONPATH=. modal run exp/a100_counters/run.py --mode validate-cublas \
  --out-root output/a100-counters
python3 -m json.tool docs/evidence/a100-mma/phase3/cublaslt-results.json
python3 -m json.tool docs/evidence/a100-mma/phase3/cublas-dispatch.json
```
The first two commands rebuild the current OA FP16/BF16 m-fast/Morton cubins and
compare them with same-source ptxas and cuBLAS on A100. The November 9 run passes
all full-output, canary and ptxas checks for the four tall cases; cuBLAS's output
also passes its FP32-reference gate. Timing is order/clock-sensitive in this
counter harness and is not used as a dispatch result. The raw current comparison
is [`counter-validation_20261109T070650.778858Z.json`](../../../exp/a100_counters/counter-validation_20261109T070650.778858Z.json).
The historical full Lt heuristic sweep/observation generator was retired; this
current run does not recreate its old algorithm selection.

## Code and retained evidence

- [cuBLAS observation](../../evidence/a100-mma/phase3/cublas-observation.json)
- [Dispatch data](../../evidence/a100-mma/phase3/cublas-dispatch.json)
- [Lt heuristic/benchmark results](../../evidence/a100-mma/phase3/cublaslt-results.json)
- [Current paired counter runner](../../../exp/a100_counters/run.py)

## What remains open

Small-grid tile selection, split-K costs and whole-step numerical differences need matched tests. A fast library GEMM that fails the prescribed complete training-state gate remains ineligible for that whole-step comparison.
