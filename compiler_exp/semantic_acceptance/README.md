# Scalar semantic acceptance and source diagnostics

The canonical contract, implementation decisions and evidence limits are in
[semantic acceptance](../../docs/compiler/semantic-acceptance.md). The broader
priority list is [coverage priorities](../../docs/compiler/coverage-priorities.md).

`input-review.txt` preserves the November 19 user-supplied independent review.
`probe.py` compares encoded scalar instruction streams with independent integer
and FP32-bit formulas under clean/poisoned registers at all optimization levels.
`before.json` retains the failing baseline. `after-final.json` records the repaired
stream. The A100 JSON/ZIP retains full outputs, input/guard checks, native and
same-source ptxas binaries and source identities. It makes no timing claim.
The exact original hardware harness/probe snapshots match their recorded hashes.

The first import and register-type fixture failures are retained in the logs;
the corrected hardware run passes. The first pinned-test payload lacked Git
and the welcome example; its failures are payload omissions, not waived tests.
The corrected pinned payload includes those dependencies and reruns every test.

`before-valid-bit-registers.json` additionally replays the corrected fixture on
an isolated unmodified `91e3f94` export. Use it for baseline comparison; the older
CPU-only fixture's register declaration was unsuitable for the hardware oracle.
The final local full suite retains one independent existing division-emulator
failure. `emulator-limit-baseline.json` and `emulator-limit-current.json` prove
the same fixed counterexample on both revisions. It is not waived by the passing
targeted or pinned tests, and is not a newly demonstrated native hardware defect.

```sh
.venv/bin/python compiler_exp/semantic_acceptance/probe.py NEW_REPORT.json
.venv/bin/python -m pytest -q tests/test_backend_contracts.py tests/test_source_diagnostics.py
modal run compiler_exp/semantic_acceptance/run_modal.py --action hardware
modal run compiler_exp/semantic_acceptance/run_modal.py --action tests
```

Preserve all original success/failure records. The live harness can evolve;
use frozen snapshots and source hashes when interpreting historical results.
