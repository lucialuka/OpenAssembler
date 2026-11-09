# Validated-not-integrated revalidation

This directory records the November 9, 2026 pass over every idea that was
previously labelled `validated-not-integrated`. The pass uses the current
OpenAssembler source identity, current experiment tests, and current A100
records where a hardware runner still exists. It separates compiler features
from diagnostics and application studies.

## Revalidation commands

```sh
.venv/bin/python -m pytest -q exp/tensor_scheduler exp/gemm_dispatch \
  exp/nn_round2 exp/weaknesses/test_reduction_overlap_codegen.py
python3 tools/check_docs.py
modal run exp/tensor_scheduler/load_next_compute_current_modal.py \
  --heldout --rounds 2 --samples 8
modal run exp/gemm_dispatch/grouped_n_modal.py \
  --heldout --rounds 2 --samples 8
```

The two Modal runners perform complete outputs, canaries, poisoned CUDA-graph
replays, matched ptxas bitwise checks, disassembly and resource checks before
timing. The runs launched after integration also save current source hashes,
Python package/runtime identity, CUDA version, `ptxas` version, `nvdisasm`
version and device model. The current load-next production-path confirmations are
`exp/tensor_scheduler/load-next-compute-current_20261109T072411.844342Z.json`
and the transposed-shape confirmation is
`...060813.116274Z.json`; the held-out set is
`load-next-compute-current_20261109T045855.363914Z.json`. The grouped-N
production-path confirmation is
`exp/gemm_dispatch/grouped-n_20261109T053553.114518Z.json` and the held-out
set is `grouped-n_20261109T051142.571274Z.json`.

Additional current-code A100 replays during the status review are retained in:

- `exp/nn_round2/runs/operator-standard-fma-20261109T063211.248802Z.json`: 64/64 standard-width local-FMA cases pass full output/canary/graph checks; ptxas remains faster.
- `exp/nn_round2/runs/operator-gelu-stable-20261109T063435.427437Z.json`: stable GELU passes all admitted rows, while the direct-tanh million-element controls retain 12 accuracy failures.
- `exp/a100_counters/counter-validation_20261109T064116.119142Z.json`: current-source NCU profiles for OA and ptxas; equal HMMA count, more OA scalar instructions, traversal-dependent stalls and bank conflicts.
- `exp/a100_counters/counter-validation_20261109T070650.778858Z.json`: current-source full-output comparison against cuBLAS; event timing is noisy and not an admission result.
- `exp/tensor_scheduler/clocked-loop-probes_20261109T064627.431181Z.json`: fresh A100 clocked-chain measurements, with no generalized scheduler promotion.

## Decision rule

An idea is `implemented` only when a current source path, numerical/resource
contract and explicit CLI or compiler option exist. A mixed or shape-specific
candidate is opt-in and does not change the default. A result is
`tested-not-selected` when the current evidence gives no production compiler
benefit, fails its gate, duplicates an existing stream, or is only a policy or
application-level change. Retired probe protocols remain documented with their
raw evidence; they are not described as fresh hardware replays.

The complete per-idea matrix is in
`validated-not-integrated_20261109.json`. The catalogue and each dossier are
the user-facing status records; this file is the experiment ledger.
