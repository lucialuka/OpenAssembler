# Native NN arithmetic and composition campaign

This isolated A100 campaign builds native OpenAssembler and exact-source ptxas
controls, checks their complete outputs before timing, and measures explicit
networks as well as individual kernels. It changes no production compiler code.

Documentation updated November 11, 2026. The records below are the source-bound
November 7–9 campaign, not a replay of every current production policy. Later
GEMM and rowwise promotion evidence lives in
[`dispatch_tuning`](../dispatch_tuning/README.md): 34 nonbaseline GEMM choices,
eight baseline retentions and five exact rowwise geometries. The backward
reduction-overlap/FMA candidates and native composition failures here retain
their original experiment-only admission; successful unrelated shape tuning
does not promote a failed full-state network.

The canonical methods, results, exclusions and reproduction commands are in the
[campaign dossier](../../docs/research/ideas/nn-arithmetic-campaign.md). Individual
new ideas have their own dossiers: [backward FMA reductions](../../docs/research/ideas/fused-backward-reductions.md),
[stable GELU backward](../../docs/research/ideas/stable-gelu-backward.md),
[early tensor pointer updates](../../docs/research/ideas/early-gemm-pointer-advance.md),
[joint tensor options](../../docs/research/ideas/joint-tensor-options.md), and
[inference weight packing](../../docs/research/ideas/inference-weight-packing-cache.md).

| Files | Responsibility |
|---|---|
| `coverage.json`, `coverage_execution_20261107.json`, `methodology.md` | Prospective arithmetic inventory, observed coverage and comparison rules |
| `operator_suite.py`, `operator_worker.py`, `operator_analysis.py` | Broad arithmetic source generation, complete GPU checks and compiler comparison |
| `operator_candidates*.py`, `operator_fma_*.py`, `operator_standard_fma*.py` | Explicit local FMA candidate, independent confirmation and standard widths |
| `operator_gelu_stable*.py` | Analytic derivative repair under the original gate |
| `tensor_suite.py`, `tensor_analyze.py`, `tensor_followup*.py` | Dense forward/dX/dW products, option combinations, legal pointer-placement experiment |
| `whole_suite.py`, `whole_worker.py`, `whole_analysis.py` | Six explicit NN configurations, full-state training, inference and graph/cache controls |
| `counter_plan.py`, `counter_fma_plan.py`, `counter_worker.py` | Bounded Nsight cohorts, output proofs, raw CSV and supported-metric checks |
| `campaign.py` | Modal transport; no GPU work happens on import or local build |
| `runs/` | Immutable successful/failed GPU attempts, timings, checks, logs and runtime identities |
| `source_snapshot.json`, `source_history/`, `evidence_manifest.json` | Initial production-source identity, retained harness revisions and evidence custody |

Generated build directories are local caches. Rebuild them using the dossier's
commands before replay; JSON identities and measurement records are retained.
`completed` means the worker returned, not that every numerical gate passed.
Look at each method's admission field and its actual reference group. Capture
errors and the initial CSV-parser failure are retained rather than overwritten.

For the frozen campaign's read-only admission and source-custody checks:

```sh
PYTHONPATH=. .venv/bin/python exp/nn_round2/check_campaign.py
.venv/bin/python tools/check_docs.py
```

The local campaign tests passed **100/100**. These exercise emitted arithmetic,
emulated outputs, gradients, launch contracts and evidence parsing; they are
distinct from the recorded A100 gates. The existing production regression
record remains separately documented in STATUS. Generated caches can be rebuilt;
an old manifest should not be re-sealed to conceal changed measurements. Source
snapshot mismatches after later integration indicate drift from that recorded
campaign, not necessarily a current compiler failure.
