# Structured-sparse MMA capability probes

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: validated-not-integrated

Status applies to the stated experiment and contract, not to every possible version of the idea.

## Idea in plain language

Use A100's structured-sparsity matrix instruction on correctly compressed operands and metadata. Skipping work is only useful when the model actually satisfies the required sparsity structure and compression costs are included.

## Findings and implementation decision

Three uniform compressed-input capability probes were compiled by NVIDIA ptxas and executed on A100-SXM4-40GB: ones, negative B and zero B. All recorded checksums and out-of-bounds guard checks passed. This is stronger than assembly-only evidence, but much narrower than a general sparse kernel test.

There is no OpenAssembler sparse selector or generated training-GEMM contract, and no sparse speedup claim. The production native dense selector explicitly refuses unsupported MMA forms. These probes establish that the particular ptxas-built hardware path works, not that dense weights can be pruned losslessly or that OA emits sparse SASS.

## Method and validity checks

Match compression/lane mapping and metadata to the sparse instruction, check known complete results/guards for the tiny uniform probe, then extend to nonuniform inputs and end-to-end sparsity semantics before benchmarking. Report pruning, packing and metadata preparation separately.

## Reproduction and command set

Run from the repository root. Use Python 3.12+ and an installed test environment. Commands that compile require the CUDA 12.8 toolkit; GPU commands additionally require an NVIDIA driver or an authenticated Modal client. Explicit skips mean a dependency/corpus is unavailable, not a passing hardware result.

```sh
python3 -m json.tool docs/evidence/a100-mma/phase2/sparse-results.json
.venv/bin/python -m pytest -q tests/test_backend_tensor.py
```
The original sparse probe generator/cubins were retired; their hashes and outcomes remain in JSON. No `gemm --sparse` command exists. Current backend tests verify supported dense forms and explicit refusal, not general sparse hardware correctness.

## Code and retained evidence

- [Three executed ptxas capability probes](../../evidence/a100-mma/phase2/sparse-results.json)
- [Native selector scope/refusal](../../../openassembler/backend/tensor.py)
- [Hardware/literature record](../../evidence/a100-mma/phase2/literature-discovery.json)

## What remains open

Implement independent sparse encoding/selection, arbitrary correct compressed data and a model-level sparsity contract. Compare total pruning/packing/inference or training cost; do not apply a sparse peak-throughput multiplier to a dense task.
