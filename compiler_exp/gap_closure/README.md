# Supplied gap register: experiments and validation

This directory follows gap definition → candidate solution → independent
experiment/validation → bounded production promotion. The
[register](gap-register.json) retains all 51 supplied exit gates; **the full
program is not complete**. The unchanged [input review](input-gap-register.txt)
is the original task material, not a current support claim.

| Work | Evidence | Boundary |
|---|---|---|
| G01 exact division and G51 seeds | [Verification](verification/), [dossier](../../docs/compiler/gaps-verification.md) | Exact retained counterexample repaired; general FCHK truth remains partial |
| G02 cross-stage contracts / G12 observations | [1,023-row ledger](contracts/ledger.json), [audit](contracts/audit.json), [dossier](../../docs/compiler/gaps-contracts.md) | 426 declared forms, representative operand domains; numerical/toolkit/GPU coverage explicitly not-run |
| G04 device helper reuse | [Python candidate/replay](python/), [dossier](../../docs/compiler/gaps-python.md) | Same-file typed straight-line scalar helpers; no general Python/library or device-call ABI |
| G25 batch-broadcast matmul / G51 diagnostics | [Model candidate/CPU/A100](model/), [dossier](../../docs/compiler/gaps-model.md) | New rank2..4 contiguous broadcast route; existing equal-batch higher ranks retained |
| G47 compile attribution | [Source/cache probes](measurement/), [dossier](../../docs/compiler/gaps-measurement.md) | Nested worker-local phases; remote startup/transport and per-pass tensor attribution remain open |
| Installed-policy continuity | [Audit](continuity/), [dossier](../../docs/compiler/gaps-continuity.md) | All 44 GEMM and five rowwise policies match frozen generation and 98 retained session artifacts; no new speed measurement |

The [structural map](repository-map.json) reads and hashes every Git-tracked
file, with module imports/definitions for Python. It is not a proof of every
historical experiment or arbitrary workload. [inventory.py](inventory.py)
regenerates it; [update_register.py](update_register.py) attaches current bounded
outcomes to the unchanged original exit gates.

## November 21 continuation

The [next integration](integration_20261121/README.md) records helper control flow, global memory/predicate/snapshot repairs, sigmoid/static reductions and repeated profiling. Its own versioned validation and source/native-code audits supersede the November 20 counts only for their named new contracts. All earlier records below remain historical.

## Environments and run custody

The supported local model checks use Python 3.12.10 and Torch 2.7.1 on macOS
15.6.1 arm64, with CUDA 12.8 tools through Colima. The compiler-only environment
uses Python 3.14.5 without Torch; its skips are recorded separately. A100
experiments use Python 3.12.1, CUDA toolkit 12.8.1 and actual
A100-SXM4-40GB devices; model Torch is 2.7.1+cu126 (runtime CUDA 12.6). Hardware
reports identify actual environment versions and source/binary hashes.

[supported-suite.log](supported-suite.log) is the first successful integrated
model/compiler run (1,277 passed), before the last equal-batch compatibility
adjustment. [final-supported-suite.log](final-supported-suite.log) and
[compiler-only-suite.log](compiler-only-suite.log) record the stable-source
final checks. [validation.json](validation.json) binds final results and hashes.
The final supported suite passes **1,278 tests with no skips**; the separate
compiler-only run passes **1,117 with 11 explicit skips**. Both live and exported
documentation checks pass. The no-Torch run supplements the supported model
environment; it cannot substitute for it.

[baseline-suite.log](baseline-suite.log) is a frozen Git export of commit
`0fdbe2e1791445c6edec22430f06dbf77fa5eade`, tested from a temporary directory
outside Colima's mounted home. It records 984 passes, 199 explicit skips and
three failures: the stale broadcast assertion, a cache test that cannot build
with its toolkit unavailable, and a dry-run fixture absent from the Git export.
It is neither a fully supported baseline-toolkit run nor a green baseline.
The unchanged stale assertion independently reproduces there with supported
Python/Torch.

[aborted-baseline-setup.log](aborted-baseline-setup.log) retains an earlier
incorrect-working-directory collection failure; it is excluded from repository
validation. A concurrent working-tree run observed 1,173 passes and 13 failures
(12 policy identity mismatches during reporting edits and the stale assertion).
Its full log was accidentally overwritten during setup; the observation is
retained in [initial-concurrent-summary.json](initial-concurrent-summary.json),
without claiming complete raw-log custody or treating the run as a baseline.
Policy identity continuity was subsequently checked before rebinding. Remote
harness path failures remain beside their successful retries in each hardware
subdirectory. Forced-fast division failures remain diagnostic negative controls.

## Reproduction

Use Python >=3.12 with pytest 8.4.2; install Torch 2.7.1 for model checks. Native
build tests need the toolkit, and the retained emulator corpus enables additional
cases. The fixed division regression also includes its own frozen SASS fixture.

```sh
python -m pytest -q
python compiler_exp/gap_closure/memory_widths/validate.py compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json /tmp/oa-memory-audit.json
python compiler_exp/gap_closure/inventory.py
python compiler_exp/gap_closure/update_register.py
python tools/check_docs.py
```

Dossiers contain each experiment's independent CPU/Modal commands, supported
operands/shapes/precision, raw outputs and limits. Exclusive-output probes require
a fresh result path. Production code never imports experiment drivers. No whole
model speedup, universal semantic coverage, arbitrary application support or
completion of all 51 gaps is claimed.
