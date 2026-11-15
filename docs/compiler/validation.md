# Validation of the November 15 compiler changes

The final installed implementation passes **1,028 tests with five explicit
skips** in the compiler environment. The separate local CPU Torch run passes
**75 tests** covering graph algebra/derivatives, numerical diagnostics and model
discovery. That CPU run uses Torch 2.15.0, outside the production model worker's
pinned 2.7.1 contract. No new general model/operator capability is claimed.

The [final verification record](../../compiler_exp/repository_coverage/validation-final.json)
binds commands, logs and retained evidence. Scope matters:

| Check | Outcome and boundary |
|---|---|
| Full compiler suite | 1,028 passed, five skipped; covers the installed parser, build, launch and 44-entry dispatch state |
| New entry tests | 24 passed, including actual toolkit builds/disassembly, scalar ABI rejection and launch/metadata integrity |
| CPU Torch regressions | 75 passed in the separate available interpreter; not pinned-worker hardware validation |
| A100 entry regression | Nine full-output/guard checks, 2,304 output elements, actual A100-SXM4-80GB |
| Original policy continuity | Every original policy retains native instruction words, controls and register counts |
| Final installed continuity | [522 candidate contracts, 1,044 session checks](../../exp/native_indexing/dispatch-continuity-20261115T053430Z.json), plus 40 rowwise checks and six tensor goldens |
| New exact-policy admission | Two FP16 contracts pass the unchanged production auditor, frozen choice and distinct selection/heldout apps; see [results](memory-latency.md) |
| Documentation | Link/catalogue/CLI checks pass; compiler experiment READMEs are now included in the maintained scan |

The entry [artifact archive](../../compiler_exp/repository_coverage/entry-artifacts.zip)
retains source PTX, native/reference cubins, disassembly and manifests. GEMM
records retain their own complete source/binary archives. Sources, binary
identities, raw timing samples and session metadata remain available for audit.
The [original entry/descriptive harness snapshot](../../compiler_exp/latency_hiding/runs/20261115T051124Z/harness-source.py.snapshot)
matches the exact harness SHA-256 recorded by both experiments; later admission
orchestration changes do not replace that source record.

## Retained failures and rejected alternatives

- The first remote harness import assumed the checkout's path depth; Modal's
  script mount differed. The failed log and repaired retry are retained. This
  was a harness failure before kernel validation.
- The first complete local run overlapped dispatch identity refresh and caught
  a stale identity. After complete binary auditing, the affected test and two
  subsequent full runs pass. The original failed log remains intact.
- The first heldout dispatch was stopped by the experimental runtime snapshot
  check after a documentation helper and table provenance changed. The retry
  reconstructs the exact frozen helper bytes from Git and records the permitted
  table-only provenance differences. The unchanged production auditor checks
  compiler identity, every source/binary, numeric gates, freeze timing and
  distinct sessions; those checks were not relaxed.
- A wider PTX regex still accepted a quoted false entry; the balanced scanner
  passes all nine parser fixtures. Static CUDA-header discovery can overselect
  an unused qualifier macro, while nvcc-based discovery incurs frontend work.
- The 1024³ timing alternative does not establish a 2% incumbent improvement,
  so its incumbent is retained. Combined-loop and early-only pointer variants
  lose on the measured shapes. No deeper-stage or warp-role rewrite is promoted.

This completes the documented entry/ABI/discovery repair and two bounded
dispatch additions. Dependency/build graph resolution, CPU application
packaging, custom launch integration and general dynamic/operator coverage
remain open in the [research plan](research-plan.md).
