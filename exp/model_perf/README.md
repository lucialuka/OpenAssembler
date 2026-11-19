# Model kernel performance campaign

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
