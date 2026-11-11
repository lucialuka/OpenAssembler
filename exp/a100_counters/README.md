# A100 counter-study workspace

The canonical methods, command sets and conclusions are in
[Hardware-counter-guided tensor analysis](../../docs/research/ideas/tensor-counter-profiling.md).
The [Morton dossier](../../docs/research/ideas/morton-cta-order.md) and
[direct-pointer dossier](../../docs/research/ideas/direct-pointer-homes.md) state
which later measurements support actual implementation choices.

`run.py` is the current bounded driver; `counter-validation_*.json` are raw
records. Preflight can request a GPU. NCU availability/permissions are checked;
a failed profile is not zero stall activity. Counter replay perturbs caches,
so its duration is not the separate validated CUDA-graph benchmark latency.
No experimental candidate is automatically installed in production by this driver.

Documentation updated November 11, 2026. Counter records here retain their
original source/device identities and describe those measured methods. The
separate [completed production replay](../dispatch_tuning/RESULTS.md) installs
34 exact nonbaseline GEMM choices and eight baseline retentions using full
numerical/binary gates and paired graph timing. It does not contain a new NCU
cohort for all 42 promoted contracts. Do not attribute older counters to a new
policy or assume a shorter instruction stream explains every timing gain.
The generalized tensor dependency scheduler remains unpromoted.
