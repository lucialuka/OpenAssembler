# Supplied independent model-coverage matrix

`input-coverage.json` is the unmodified user-provided `closed_exp/coverage.json`.
`audit.py` checks expected admission decisions and all seven timing repetitions.
It keeps unsupported rows outside accepted CPU timing cohorts and records the
input SHA-256 and original environment in `audit.json`.

`replay.py` defines a new explicit corpus from the supplied shape metadata.
The successful replay keeps 27 accepted/12 rejected decisions with seven samples
and all four storage settings. It cannot reconstruct missing original factories
or establish GPU/training-state correctness; its timings include four planning
and preparation combinations per repetition. The initial varargs fixture failure
is retained alongside the corrected fixed-arity replay.

The canonical interpretation and open implementation work are in
[coverage priorities](../../docs/compiler/coverage-priorities.md). These are CPU
capture/planning/PTX-preparation records; this directory does not introduce new
GPU performance or full-state equivalence claims.

```sh
.venv/bin/python compiler_exp/coverage_matrix/audit.py
.env/bin/python compiler_exp/coverage_matrix/replay.py
```
