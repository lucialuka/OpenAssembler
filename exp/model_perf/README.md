# Model kernel performance campaign

The November 26 continuation follows the attached kernel-research agenda under
the same `exp/` workflow. Separate-pack reuse covers a repeated row-major 128×128
FP32 source converted to FP16 within one forward call. The allocator experiment
measures compile cost with exact executable equality. Results and retained
negative controls are in [RESULTS.md](RESULTS.md).

```sh
modal run exp/model_perf/modal_run.py --phase pack-reuse
modal run exp/model_perf/modal_run.py --phase pack-reuse-integrated
modal run exp/model_perf/modal_run.py --phase allocator-build
.venv/bin/python -m exp.model_perf.allocator_gate --integrated --focused --output output/allocator-warm.json
.venv/bin/python -m exp.model_perf.allocator_gate --integrated --samples 1 --output output/allocator-all.json
.venv/bin/python -m exp.model_perf.allocator_gate --cold --output output/allocator-cold.json
.venv/bin/python -m exp.model_perf.retain_kernel_research
python -m exp.model_perf.audit_pack_reuse exp/model_perf/runs/pack-reuse-integrated-20261126T084247Z --output output/pack-reuse-audit.json
```

`pack-reuse` uses the experimental adapter. `pack-reuse-integrated` uses the
production runtime, exercises four storage configurations and adds a ptxas arm
with the same reuse schedule. Each workload runs in a fresh process. Both arms
load ordinary verified bundles, with an explicitly disabled reuse schedule for
the incumbent control. The scripts retain randomized graph-event samples,
complete output/gradient/update checks, poisoned replay, changed inference
weights, guards and independent profiler launch audits.

The independent audit command requires a CPU Torch installation and the retained
raw run archive. It reloads complete expected/actual state snapshots, verifies
field hashes and tolerance gates, checks cubin text/registers and recomputes
every shuffled timing median. Raw archives remain under ignored `runs/`; their
identities and compact envelopes are retained under `evidence/`.

`allocator_gate` retains the original interval scan and early-scheduling
candidate. Production uses the streaming interval scan. The cold samples start
fresh subprocesses and encoders. `allocator-build` compares complete cubin bytes
after toolkit compilation and packaging and runs the relevant toolkit/model
tests. `allocator_continuity` verifies all installed policy binaries before
rebinding dispatch identities. Its `--update` option writes only verified table
identity/provenance fields, preserving policy and measurement records.

`native-pack` is an unpromoted packed-conversion experiment. Its partial and
failed envelopes remain retained with the later successful reuse experiments.

November 19, 2026. Starts from revision `91e3f94` and the user's frozen-task
review. This work targets performance within existing coverage; it does not
extend language/operator support. See [RESULTS.md](RESULTS.md).

The repository survey inventories tracked source modules, tests and active
Markdown, reviews changes since the earlier FP16 implementation, and reads the
model/backend/benchmark paths relevant to the measured bottlenecks. This is a
structural survey and targeted source review, not a claim to have manually read
every historical source snapshot or binary. Its inventory is retained in
[repository-survey.json](repository-survey.json).

```sh
modal run exp/model_perf/modal_run.py --phase experiment
modal run exp/model_perf/modal_run.py --phase operators
modal run exp/model_perf/modal_run.py --phase gemm
modal run exp/model_perf/modal_run.py --phase confirm
modal run exp/model_perf/modal_run.py --phase integrated
modal run exp/model_perf/modal_run.py --phase comparisons
modal run exp/model_perf/modal_run.py --phase cache
modal run exp/model_perf/modal_run.py --phase tests
python -m pytest -q tests/test_model_training_kernels.py tests/test_model_cache_tools.py
python tools/check_docs.py
```

`candidate.py` contains four explicit ablations: incumbent, transpose-pack,
small tile and combined. Reduction ablations test incumbent, tile-only,
reduction-only and both. `baseline_tensor.py` and `baseline_kernels.py` freeze
the pre-change source for reproducible comparisons after promotion. `integrated`
compares those frozen emitters with production code; production bundles still
validate generated source and native executable ownership. `production` uses
only current production dispatch, without experimental patches.

The pinned Modal worker ships local compiler/data and uses Python 3.12,
Torch 2.7.1/cu128 and A100. `comparisons` invokes the existing frozen ML suite;
Triton, same-PTX ptxas and manual CUDA-operator controls remain research arms.
No comparator cubin enters OA production execution. The worker treats every
nonzero child exit as failure, retains source hashes, samples and unsuccessful
outcomes, and saves generated build trees only under ignored `runs/`. Compact
results, plans and executable/source hashes are retained under `evidence/`.
