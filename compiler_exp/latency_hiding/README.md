# A100 memory/latency policy experiments

The canonical contract and interpretation live in
[docs/compiler/memory-latency.md](../../docs/compiler/memory-latency.md).
This directory retains experiment sources, negative outcomes, frozen selections,
timing samples and complete checked artifact archives.

The descriptive `runs/20261115T051124Z` experiment evaluates three exact shapes
in two sequential sessions. Its frozen policy is assessed without replacing it
by the second session's winner. These sessions may reuse a worker/GPU; this
experiment alone does not authorize a production promotion.

The separate `runs/admission-20261115` campaign evaluates two uncovered exact
contracts in distinct selection/heldout Modal applications through the normal
production auditor. Frozen selections precede heldout execution. Only the
existing auditor can admit entries; the experiment never changes its gates.

```sh
modal run compiler_exp/latency_hiding/run_modal.py --action timing
.venv/bin/python compiler_exp/latency_hiding/analyze.py compiler_exp/latency_hiding/runs/20261115T051124Z
modal run compiler_exp/latency_hiding/run_modal.py --action selection --campaign NEW_NAME
modal run compiler_exp/latency_hiding/run_modal.py --action heldout --campaign NEW_NAME
.venv/bin/python compiler_exp/latency_hiding/promote.py compiler_exp/latency_hiding/runs/NEW_NAME
```

The last command audits and writes decisions only. `--install` additionally
appends admitted exact contracts, rejects replacements and requires the existing
dispatch identity to be current. Do not rerun selection into an existing
campaign or install records without reviewing their complete results.

`--action entries` is a separate full-output/guard regression for the new PTX
entry forms. Its results live under `compiler_exp/repository_coverage/`. The
first remote import failed because the mounted script's location differed
from the checkout; that failed log is preserved alongside the successful retry.
