# Transformer GEMM role and shape coverage

This experiment audits representative transformer GEMM roles against the
current-at-generation `GemmPlan` contract (the 128×128 reference). It is a
static coverage audit, not a performance claim and not a replacement for A100
numerical/timing validation. The later 64×64 family and exact dispatch table
are documented separately; this matrix is not a role-level production route.

Run it from the repository root:

```sh
.venv/bin/python exp/shape_coverage/role_coverage.py
.venv/bin/python -m pytest -q exp/shape_coverage/test_role_coverage.py
```

The generated [`role-coverage-manifest.json`](role-coverage-manifest.json)
separates four decisions that must not be conflated:

* dimensions fit the full-tile 128-aligned GEMM generator;
* operand layouts are already contiguous RCR without transpose/packing;
* the role can be launched as one OA GEMM rather than a batched/per-head
  operation; and
* an optional policy has exact evidence for that role and shape.

The current audit covers the pasted report's `B=8,S=128,H=512` case, plus small,
medium, long-context and ragged microbatch cases. It includes QKV, attention
output, FFN up/down, vocabulary output, QKV/output/FFN dX/dW, single-head attention QK/AV and
FP16/BF16. Many dX/dW shapes pass the dimension test but fail the direct
role route because transposes and packing are outside the current contract.
Attention QK/AV additionally require per-head batching, masking and layout
semantics. These are the next contracts to implement and benchmark; padding
alone must not be treated as support.

## Current A100 role run

`a100_role_gemm_modal.py` measures the direct forward roles that fit the current
RCR contract. The source-hashed run
[`a100-role-gemm_20261109T075039.758591Z.json`](a100-role-gemm_20261109T075039.758591Z.json)
covered QKV, attention output, FFN-up and FFN-down in FP16/BF16, with full
outputs, three input families, canaries, input immutability, matched-ptxas
bitwise checks, poisoned graph replay, disassembly and resource checks before
timing. All 156 method/input checks pass.

For the medium `M=1024,H=1024` roles, same-source ptxas remains faster. In the
second timing round, OA/ptxas was approximately 1.15x for QKV, 1.48x for the
output projection, 1.40x for FFN-up and 1.43x for FFN-down. The load-next
candidate reduced FFN-up OA time by about 9% in both dtypes, while grouped-N
was mixed and Morton was neutral or slower. These are role-specific measurements
on one A100 worker, not a whole-transformer result or an automatic dispatch rule.

Reproduce the current role run:

```sh
modal run exp/shape_coverage/a100_role_gemm_modal.py --rounds 2 --samples 8
modal run exp/shape_coverage/a100_role_gemm_modal.py --heldout \
  --role ffn_up --dtype f16 --rounds 1 --samples 4
```

The report’s suggested percentages should be interpreted carefully: GEMM,
attention and reduction numbers in the whole-stack profiler are overlapping
recognized-kernel-family sums, not additive shares of a training step. The
resident-data stage table remains the appropriate whole-step reference; it
excludes H2D, data loading, compilation and distributed communication.

A held-out FFN-up run across small, medium and large model shapes is retained
in [`a100-role-gemm_20261109T080749.220619Z.json`](a100-role-gemm_20261109T080749.220619Z.json).
All 54 method/input checks pass. Load-next is selected only for the measured
medium 1024×4096×1024 role; the small 128×2048×512 and large 4096×8192×2048
roles correctly fall back to the other policies and remain slower than ptxas.

Held-out QKV, output-projection and FFN-down runs likewise pass all 84 checks
per role: [QKV](a100-role-gemm_20261109T081712.382539Z.json),
[output](a100-role-gemm_20261109T082039.525666Z.json), and
[FFN-down](a100-role-gemm_20261109T082356.053685Z.json). Their small and large
shapes remain behind ptxas and do not justify applying a medium-shape policy by
role name alone.

## Invalidated pointer-policy labels

The initial pointer sweep is retained in
[`a100-role-gemm_20261109T083147.885491Z.json`](a100-role-gemm_20261109T083147.885491Z.json);
the BF16 record is
[`a100-role-gemm_20261109T083359.463266Z.json`](a100-role-gemm_20261109T083359.463266Z.json).
Those runs passed their numerical checks, but **their direct-pointer labels
are invalid**. The harness renamed the entry and omitted the checked generated
source annotation, so the compiler did not receive the pointer-home policy.
The preliminary `082832` record has the same flaw. These records cannot support
pointer dispatch decisions; they do not invalidate independent production tests.

The corrected runner preserves `source_for(plan)` and the `gemm` entry, asserts
the policy fields in the compiled native manifest, and includes cuBLAS GemmEx
with matched input/accumulation/output precision. It checks cuBLAS by the
declared FP32 reference gate, rather than requiring the same reduction order.

```sh
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --role ffn_up --dtype f16 --pointer-sweep --rounds 3 --samples 12
```

## Exact experimental dispatch gate

[`measured_dispatch.py`](measured_dispatch.py) validates normalized proposals
for another integration experiment. It requires exact role/shape/layout/
alignment/workspace/numerical and implementation keys; current compiler and
harness identities; complete numerical, binary and resource gates; matched
ptxas/library comparisons; and independent selection/held-out sessions. At
least two randomized paired rounds with eight samples each must exceed the OA
baseline by 2% in every round. That threshold is an experiment acceptance rule,
not a hardware property. It reports losses to ptxas/cuBLAS explicitly and always
returns `production_promoted: false`.

Existing raw records are not silently converted into dispatch entries.
The selector verifies normalized evidence consistency, not the truth of manual
assertions: an audited importer still has to connect each claim to raw data.

```sh
.venv/bin/python -m pytest -q exp/shape_coverage
.venv/bin/python -m exp.shape_coverage.measured_dispatch normalized-proposal.json
```

## Fragment placement and hardware counters

The [FP16/BF16 fragment replay](a100-role-gemm_20261109T111245.728446Z.json)
checks split-2/4/6 load-next placements against source-order, production
load-next, pointer policies, same-policy ptxas, baseline ptxas and matched cuBLAS.
All 126 method/input checks and 42 poisoned-graph checks pass. Split-4 exactly
matches the production load-next body. Split-2 and split-6 do not beat that
placement on the medium FFN-up role, despite split-6 using four fewer registers.

The [counter replay](a100-role-gemm_20261109T111551.304398Z.json) retains
raw CSV, units, commands, metric queries and per-kernel records for baseline,
direct+early, load-next, split-6, ptxas and cuBLAS. All 20 requested counters
were collected. Counters use separate fresh driver contexts after numerical
gates and benchmark timing; replay duration is not used as a timing result.
The profiler preserves all kernels in one library invocation instead of assuming
one skipped launch is one skipped library call.

```sh
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --role ffn_up --pointer-sweep --fragment-sweep --rounds 3 --samples 12
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --role ffn_up --dtype f16 --pointer-sweep --fragment-sweep --profile \
  --rounds 3 --samples 12
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --heldout --role ffn_up --pointer-sweep --fragment-sweep --rounds 3 --samples 12
```

The schema-3 source/manifest guard and strict fragment certificate are in
[`gemm_policy.py`](gemm_policy.py) and
[`fragment_split_search.py`](../tensor_scheduler/fragment_split_search.py).
Experimental split sources deliberately have no inferred production contract;
the harness validates and launches their bounded manual contract explicitly.

## Small-grid kernel family

[`small_tile_gemm.py`](small_tile_gemm.py) generates an explicit 64×64×64 CTA
experiment with four 32×32 warp tiles, 32 accumulator values per thread and
32 KiB shared memory. Its layout checks exhaustively cover output ownership,
copy coverage and LDSM bounds/alignment. It uses the baseline HMMA8/drain15
policy and requires a separate manual launch; no production contract is changed.

The [first 128×512×512 FP16 replay](a100-role-gemm_20261109T112122.548949Z.json)
passes all 15 method/input and five graph checks. OA takes 11.01 µs versus
21.25–21.38 µs for the 128-tile baseline. The old-tile ptxas control takes
14.85 µs, but matched tile64 ptxas is 8.45 µs and cuBLAS is 7.68 µs.
A kernel-geometry win is therefore not a backend superiority claim.

The [32-case role/shape sweep](a100-role-gemm_20261109T113850.802215Z.json)
passes all 1,296 method/input and 432 graph checks. Tile64 improves 18 cases
and regresses 14; the large-role regressions reach about 49%. The
[independent frozen replay](a100-role-gemm_20261109T113154.637725Z.json) of the
small output case passes 33 method/input and 11 graph checks. See the
[canonical dossier](../../docs/research/ideas/small-grid-tensor-cta-tiles.md)
for the cohort table, source protocol, counter analysis and promotion boundary.

```sh
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --heldout --model-case small_fp32_context --role attention_output_projection \
  --dtype f16 --tile-sweep --rounds 3 --samples 12
modal run --detach exp/shape_coverage/a100_role_gemm_modal.py \
  --heldout --tile-sweep --pointer-sweep --rounds 3 --samples 12
```

The model-case label `small_fp32_context` identifies its model geometry, not
the GEMM input precision: each recorded case explicitly says FP16 or BF16.
Likewise the `mid_gqa_style` audit label does not implement GQA semantics.

## Raw-evidence inventory

[`evidence-ledger.json`](evidence-ledger.json) inventories raw run hashes,
device cohorts, actual method/input counts, recorded gates, policy/source scope
and counter completeness. It flags the invalid pointer labels even though
those numerical checks passed. It does not infer winners or promotions from
names, booleans or aggregated timings.

```sh
.venv/bin/python -m exp.shape_coverage.audit_role_evidence
```

The source-identity-locked production table is
`openassembler/kernels/sm80_gemm_dispatch.json`. It is validated separately
from the historical role JSONs. Original output arrays/cubin bytes were not
retained in those older role JSONs; the audit records that limit and does not
claim independent hardware rerunning. The candidate remains slower than
matched tile64 ptxas and cuBLAS. Current table entries must use their own
source/body hash and full-gate evidence.

```sh
modal run --detach exp/shape_coverage/a100_small_tile_counters_modal.py
```
