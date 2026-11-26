# Local-package device helpers: research and production integration

Status: production integrated; complete source-exact evidence retained.
The canonical [helper dossier](../../../../docs/compiler/gaps-python.md#static-local-package-device-helpers)
contains the exact current contract and remaining G04/G42 scope.

The retained [normalization prototype](../README.md) reproduced three source
call failures and passed emulator/toolkit controls. It lacked package initializer
identity and original diagnostic handling. This cycle uses each helper's own
namespace and source AST, then statically specializes pure scalar calls through
the existing emitter. No user package/entry/helper module is imported or executed.
The exact baseline/candidate source snapshots are retained here.

The graph handles direct/module aliases, transitive relative imports and static
re-exports under the inferred enclosing package root or `--python-source-root`.
It snapshots reachable entry/helper/package bytes, body/signature/call identities
and original lines, validates before/after compilation, and binds the complete
graph to PTX/kernel/native ownership. The normal remote bundle includes ignored
reachable helpers, uses the same source root, checks returned graphs and rebases
package entry paths before publication. Source/module/call-size/depth limits and
invalid type/effect/binding/import/recursion diagnostics remain explicit.

## Independent experiments

- [Offline gate](candidate-offline.json): 1,080 complete executions / sixty rows;
  five original source-tree variants × three entries × four levels × two seeds ×
  three lengths × PTX/normal/poisoned SASS. Independent whole modulo64/FP32 bytes,
  unchanged inputs, guards/tails and repeated single-entry source snapshots pass.
- [Negative controls](negative-controls.json): 29 import/type/effect/recursion/
  binding/diagnostic/source-snapshot/symlink/ambiguity controls. A standalone
  [production fixture corpus](module_helper_fixtures.json) supplies golden bytes.
- [Toolkit control](oracle-probe.json): five original source trees, fifteen
  emitted kernels, all accepted by actual ptxas; 42 raw artifact identities.
- Original-source A100 gates use ordinary CLI builds and a serialized worker,
  three 128-thread CTAs, two seeds and lengths 0/1/259/384. Direct imports,
  re-exports, entry-package and explicit-root graphs all pass. The final
  source-exact gate has 600 launches, 5,529,600 independently checked bytes and
  sixty exact native word/register/resource/ownership comparisons. Original
  source/graph/cubin/disassembly/manifest/input/output archives are retained.
- [Focused production tests](targeted-suite.log): 73 pass. All 273 earlier frontend
  tests and 100 earlier build/ownership/database tests pass with the candidate.
- [Frontend continuity](frontend-continuity.json): sixteen prior source files,
  62 exact unchanged kernels and unchanged rejection results.
- [Serialized local worker](remote-probe.json): reachable ignored helper included,
  five exact dependency identities, package entry path preserved and ownership.

## Retained failures and corrections

`first-install-failure.txt` explains the stale package-facade function reference
in the isolated installer. The first namespace experiment also misclassified a
package initializer as a file/package ambiguity. Neither affected production.
`first-single-entry-comparison-failed.log` retains a harness comparison of AST
source-line coordinates between full-module and single-entry PTX; the corrected
gate compares complete native words/registers and unchanged source graph identity.
The first existing frontend tests retain a changed recursion-message spelling;
module-specific cycle identity now preserves its established user diagnostic.
The negative-control logs retain path-alias/expected-diagnostic harness mismatches;
all corrected controls still require the actual source/type/effect rejection.
`first-publication-path-check-failed.log` retains a check that treated a relative
manifest source as a process-relative path; publication already succeeded, and
the corrected checker uses the manifest's own root.

The initial 600-launch gate preceded imported syntax-line and single-snapshot
hardening. Later source-exact gates retain those corrections, and a final rerun
includes the updated frontend contract docstring. Earlier successes are retained
separately; their byte counts are not pooled as independent coverage or timing.
Production imports no experiment module. General package/host execution, mutable
captures, runtime import hooks and arbitrary helper effects remain open.

```sh
python compiler_exp/gap_closure/python_module_helpers/promotion/offline.py
python compiler_exp/gap_closure/python_module_helpers/promotion/negative_controls.py
python compiler_exp/gap_closure/python_module_helpers/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_module_helpers/promotion/run_modal.py
python compiler_exp/gap_closure/python_module_helpers/promotion/normal_remote_check.py
python -m pytest -q tests/test_pyfront_module_helpers.py
```

The first full supported and compiler-only suites each found one real integration
regression in `test_failed_frontend_retains_completed_phase_time`: the initial
Python builder bypassed the shared frontend adapter. `first-supported-suite.log`
and `first-compiler-only-suite.log` retain those failures (3,536 and 3,051 other
passes respectively). Production now routes Python through `to_ptx_module`,
passing the explicit source root through that adapter and retaining existing
failure/timing behavior. The focused routing/measurement suite passes. A further
regression keeps remote directory discovery within the requested target when
an explicit source root sits above it; the A100 gate also builds a directory.
The final source-exact report is `candidate-a100-20261123T224420Z.json`, with the
same 600-launch independent numerical gate and final source hashes.

The complete prior-native replay passes **2,090 occurrences**. The retained
pre-final-adapter audit preserves its original source hashes and comparison
records. A [supplemental binding](final-adapter-continuity.log) checks identical
frontend executable AST and unchanged build/remote AST outside the entry adapter
and client target discovery, then requires the final source-exact A100 admission
before rebinding those transport-only sources. Existing native generation stays
structurally identical; no numerical report or earlier binary is rewritten.
The [public CLI/Modal check](normal-remote-audit.json) passes an explicit-root
directory through actual sender/worker/local publication. The installed wheel
recreates all five original trees and sixty exact PTX/native outputs. The rerun full suites pass **3,538 supported tests** (Python 3.12.10/Torch 2.7.1,
zero skips) and **3,053 compiler-only tests with fifteen skips** (Python 3.14.5,
no Torch). Failed first suites remain retained. The [integration record](validation.json)
binds final sources, complete byte/native/ownership evidence, original failed runs
and live/exported documentation.
