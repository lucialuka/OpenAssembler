# OpenAssembler reverse-engineering results

This directory is the canonical technical record for the reverse engineering
that underlies OpenAssembler. It explains what was learned from NVIDIA's
`ptxas`, `nvdisasm`, CUDA cubins and A100 execution, how each result was
validated, what became production code, and what remains experimental.

The raw records remain in `corpus/`, `docs/evidence/` and `exp/`. Those files
contain large JSON payloads, SASS listings, cubins, hashes and machine-readable
measurements. This directory provides the readable interpretation and points to
the exact evidence instead of duplicating those payloads.

## Reading order

1. [Executive summary](00_EXECUTIVE_SUMMARY.md) — the conclusions and status.
2. [Scope, method and provenance](01_SCOPE_METHODS_AND_PROVENANCE.md) — how the black-box observations were made and what counts as evidence.
3. [ptxas and frontend pipeline](02_PTXAS_AND_FRONTEND_PIPELINE.md) — the compiler boundary and reference strategy.
4. [SASS encoding and control fields](03_SASS_ENCODING_CONTROL_FIELDS.md) — instruction words, control bundles and field validation.
5. [CUBIN/ELF metadata and splicing](04_CUBIN_ELF_METADATA_AND_SPLICING.md) — how loadable binaries are preserved and edited.
6. [Scalar latency, scheduling and registers](05_SCALAR_LATENCY_SCHEDULING_AND_REGISTERS.md) — fixed-pipe latency, scoreboards, allocation and WAR.
7. [SM80 MMA pipeline](06_MMA_TENSOR_PIPELINE_SM80.md) — HMMA dependency classes and clocked A100 probes.
8. [Async copy, LDSM and shared memory](07_ASYNC_COPY_LDSM_SHARED_MEMORY.md) — `cp.async`, barriers, address-WAR and swizzling.
9. [Arithmetic and division](08_ARITHMETIC_DIVISION_AND_FRAGMENTS.md) — recovered arithmetic fragments, division strength reduction and numerical contracts.
10. [Reductions and NN arithmetic](09_REDUCTIONS_AND_FUSED_NN_ARITHMETIC.md) — backward reductions, FMA contraction and counter findings.
11. [Layouts, CUTLASS and cuBLASLt](10_LAYOUT_SWIZZLE_CUBLAS_CUTLASS.md) — public-library observations and tested A100 candidates.
12. [Counters, performance and validation](11_COUNTERS_PERFORMANCE_AND_VALIDATION.md) — how timing and Nsight results are interpreted.
13. [Reproduction map](12_CLI_REPRODUCTION_MAP.md) — commands, evidence paths and validation gates.
14. [Open questions](13_OPEN_QUESTIONS_AND_NEXT_STEPS.md) — unresolved technical work.

## Status vocabulary

Each result is classified separately for correctness, performance and shipping:

| Status | Meaning |
|---|---|
| Implemented | The behavior is available in production code under an explicit contract. |
| Validated, not integrated | The experiment is correct or informative but is not a production optimization. |
| Tested, not selected | The candidate was tested and rejected, regressed, or failed a gate. |
| Open or unavailable | The idea is proposed, incomplete, unprofiled, or requires hardware that A100 does not have. |

“Implemented” never means universally faster. “Validated” never means
production-ready. A result becomes a production optimization only when its
numerical, binary, resource, counter and held-out timing gates are all stated
and satisfied.

## Evidence roots

| Evidence root | Purpose |
|---|---|
| [`corpus/sm_80`](../../corpus/sm_80) | Committed probe inputs, recovered fields, machine models and archived GPU records. |
| [`specs/sm_80`](../../specs/sm_80) | Encoding templates, field maps and generated instruction specifications. |
| [`docs/evidence/a100-latency`](../evidence/a100-latency/README.md) | Corrected 42-case A100 scalar latency sweep. |
| [`docs/evidence/a100-mma`](../evidence/a100-mma/README.md) | A100 MMA, GEMM, layout, counter and integration evidence. |
| [`docs/research/ideas`](../research/README.md) | Per-feature contracts and implementation status. |
| [`exp`](../../exp) | Current experiment drivers and raw candidate measurements. |
| [`tools`](../../tools) | Probe generation, corpus construction, GPU replay and analysis utilities. |

The catalogue remains the per-feature status index. This directory is the
cross-cutting reverse-engineering narrative that connects those dossiers.
The machine-readable topic map is [`EVIDENCE_INDEX.json`](EVIDENCE_INDEX.json).

