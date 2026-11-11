# Experiment drivers and evidence

This directory contains isolated research drivers and raw results. It is not an
alternative documentation tree and is not imported by the production compiler.
The canonical findings and exact reproduction commands are in the
[per-idea catalogue](../docs/research/README.md).

Updated November 11, 2026. The latest production promotion evidence is the
[completed exact-shape replay](dispatch_tuning/RESULTS.md): 42 GEMM contracts
install 34 nonbaseline policies and eight measured baseline retentions. The
separate rowwise sweep installs five exact softmax/LayerNorm CTA geometries.
These are paired improvements over OpenAssembler's reference plans. Same-source
ptxas remains faster for every nominal-selected GEMM in the 42-case campaign;
none of these kernel results establishes a whole-transformer training gain.

| Workspace | Use |
|---|---|
| [a100_counters](a100_counters/README.md) | Matched native/library counter and graph controls |
| [whole_stack](whole_stack/README.md) | Original transformer and hybrid-MLP application/stage study |
| [weaknesses](weaknesses/README.md) | Validated compiler/operator follow-ups and rejected controls |
| [dispatch_tuning](dispatch_tuning/README.md) | Current production-source selection, frozen independent replay, exact GEMM/rowwise promotion evidence and durable recovery |
| [instruction_overhead](instruction_overhead/README.md) | Copy-address reuse, pointer homes and tile-geometry experiments; historical pilots feed the current exact-shape replay |
| [nn_round2](nn_round2/README.md) | Native arithmetic, dense training products, composed networks, legal source-template candidates and counter-guided follow-ups |
| [shape_coverage](shape_coverage/README.md) | November 9 model-role audit and exploratory A100 forward-role measurements; current promotions use the later production replay |
| [tensor_scheduler](tensor_scheduler/README.md) | Validated bounded SASS dependency probes, unpromoted general scheduler and explicit fragment-overlap experiments |
| [gemm_dispatch](gemm_dispatch/README.md) | CTA traversal models and historical shape/alignment/workspace candidate matrix |
| [revalidation](revalidation/README.md) | Source-bound early November 9 status-review ledger |

The critical interpretation and procedure review for the pasted transformer
report is in [shape_coverage/report_review.md](shape_coverage/report_review.md).

Keep raw samples, failed gates and identities; do not create another phase
report that duplicates a dossier. Generated build/payload caches remain
noncanonical. [Development conventions](../docs/DEVELOPMENT.md) explain evidence
levels, promotion requirements and retained documentation provenance.

Superseded narrative reports were removed after their unique findings moved
into the canonical dossiers. Exact prior text is retained once in
[documentation provenance](../docs/evidence/documentation-history/README.md);
historical raw success and failure records remain evidence, rather than being
advertised as tests of the latest compiler.
