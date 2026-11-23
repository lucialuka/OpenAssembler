# Native code continuity across compiler repairs

Status: implemented

## November 21 backend and model integration

The [current audit](../../compiler_exp/gap_closure/integration_20261121/audit-20261121T124658Z.json)
again checks all 44 GEMM and five rowwise policies against 98 original
selection/heldout cubins. The admitted source changes are limited to build
instrumentation and the separately hardware-validated register-selector and
scalar-contract repairs. Current and frozen code, labels, full instruction
words/controls, exits and register counts all match those retained policy
executables. Policy choices, measurements and production identity guards are
unchanged; only compiler binding and continuity history advance.

The [revised driver](../../compiler_exp/gap_closure/integration_20261121/continuity.py)
loads frozen scalar-contract, register-selector and tensor modules together,
including the tensor selector's base class and configuration type. It restores
the live modules after each comparison. Review caught cached tensor-module
references in an initial version, followed by a configuration-class identity
error; those runs stopped before publication and their logs/correction are
retained. A later audit was stopped when independent review found packed value
snapshot defects, which were repaired and hardware-gated before the final run.

The separate [model executable audit](../../compiler_exp/gap_closure/integration_20261121/hardware-continuity-20261121T125602Z.json)
checks 106 archived bundles and 866 kernel occurrences from the final operator
and repeated-profiling A100 gates. It regenerates 239 unique scalar kernel
inputs and proves full native-word/control, register-count and EXIT-metadata
equality against the saved OA cubins. The twelve required model/runtime source
identities match their reports. Backend source and dispatch metadata drift is
explicitly named and bound to final source hashes; matching executable bytes,
rather than source similarity, permits reuse of those hardware observations.
This audit is not new hardware execution or a new timing measurement.

Run the current read-only audits from the repository root:

```sh
python compiler_exp/gap_closure/integration_20261121/continuity.py --baseline 0fdbe2e1791445c6edec22430f06dbf77fa5eade
python compiler_exp/gap_closure/integration_20261121/hardware_continuity.py --help
python -m pytest -q tests/test_rowwise_dispatch.py tests/test_gemm_dispatch.py
```

## Retained November 20 reporting-only audit

Adding G47 build-phase timing changed the content identity of `build.py` and
therefore invalidated measured dispatch tables. The fail-closed policy correctly
retained baseline policies until the new source could be checked. Changing or
excluding identity inputs would weaken that policy.

The selected solution was a binary continuity audit followed by an explicit
identity update. The [audit script](../../compiler_exp/gap_closure/continuity/check.py)
reconstructs the exact Git baseline, requires its identities to match the frozen
installed tables, and checks that only `openassembler/build.py` changed among
both dispatch identity input sets. It verifies unchanged scoped CLI/runner
sources and an identical `generate()` function AST, then loads the frozen build
module separately and compares it with the current build.

The [retained audit](../../compiler_exp/gap_closure/continuity/audit-20261120T153239Z.json)
passed all 44 installed GEMM and five rowwise contracts. For each policy,
generated PTX, native instruction words including scheduling controls and padding,
labels/exits, and register counts matched between frozen and current compilers.
The generated native words/register counts also matched each policy's original
selection and independent heldout cubin, for 98 retained-session comparisons.
All referenced raw-result and artifact-archive hashes were checked.

Only the installed compiler identities and continuity metadata changed. Every
policy choice, generated PTX hash, original measurement identity, numerical gate,
speedup, device record and artifact identity remains unchanged. Previous
continuity metadata is retained in `source_continuity_history`. Production
identity functions and stale/corrupt-table rejection rules are unchanged.
The GEMM and rowwise regression files passed 31 tests after the update.

The original reporting-only audit can be reproduced in its recorded checkout;
its strict source-change guard intentionally rejects the later backend repairs:

```sh
python3 compiler_exp/gap_closure/continuity/check.py --baseline 0fdbe2e1791445c6edec22430f06dbf77fa5eade
python3 -m pytest -q tests/test_rowwise_dispatch.py tests/test_gemm_dispatch.py
```

`--update` explicitly rebinds the tables only after every check passes. The audit
also refuses changes outside its narrow build-instrumentation scope or a compiler
identity change during the run. This is source/binary continuity under Python
3.12.10; it is neither new hardware validation nor a new timing result. It
preserves the original A100 measurement boundaries and device cohorts.
