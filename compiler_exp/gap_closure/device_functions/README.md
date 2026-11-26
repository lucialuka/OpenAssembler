# Direct PTX device-function specialization

Status: validated production promotion for the scalar direct-call contract;
G16, G10 and the original 51-gap program remain partial. The canonical
[device-function dossier](../../../docs/compiler/gaps-device-functions.md)
owns the contract, methods, commands, evidence and limits.

## Gap and solution

The parser/build was entry-local: direct `call` operands rejected even when
nvcc successfully emitted a scalar `__noinline__` helper. `baseline-parse.py`,
`baseline-build.py`, `baseline-native-ownership.py` and `baseline.json` retain
the original sources and fifteen rejected source cases. `noinline.cu` and its
exact toolkit PTX/cubin/listing demonstrate actual `.func`, scalar `.param`,
call-local lexical blocks and `call.uni` syntax. That first wide CUDA helper
also exposes unsupported `selp.b64`; function expansion does not imply adding
an unrelated scalar operation. `nvcc_scalar.cu` supplies a complete supported
CUDA noinline route, compiled both locally and in the final A100 worker.

`inlining.py` resolves module-local direct functions, validates scalar slot
arity/width/initialization and performs static specialization. Each call copies
arguments into fresh formal values, allocates fresh registers/labels/local
objects and converts returns into branches. Lexical blocks preserve reused
call slot names. Alias and false-call output cases keep their old values.
Nested calls and loops are admitted under depth/statement bounds; recursion,
indirect/unresolved calls, aggregate slots, multiple parameter returns and
escaping parameter storage reject. Native call-stack and cross-module/library
linking are not implemented.

The candidate parser retains original instruction lines after expansion. The
module and per-kernel manifest inventory record function signatures, body hashes
and expanded call counts; ownership validation binds that manifest contract to
executable bytes. No experiment imports enter production. Existing local-frame
allocation/resource metadata accounts for independent static callee objects.

## Experiments, failures and corrections

- `initial-candidate.log` retains a missed guard rename; `second-candidate.log`
  reaches the unrelated `selp.b64` rejection in the first CUDA fixture.
- The first representative matrix admitted two `.param` return slots, while
  the actual toolkit rejected that ABI (`oracle/multi_return.tool.log`). The
  candidate now rejects it. `initial-fixtures.py` and corresponding reports
  retain the original unsuccessful domain.
- Initial focused tests used source parameter names for nvcc's renamed ABI
  symbols and expected an overly narrow diagnostic. `first-candidate-tests-failed.log`
  and `first-nvcc-offline-failed.log` retain those harness failures; corrected
  tests bind the actual parsed ABI. A later prototype first used `.extern`
  with a local definition; actual toolkit rejection and the original source
  remain in `first-expanded-*` and `prototype-*`. A bare matching-signature
  prototype with different parameter names passes the toolkit and candidate.
- `snapshot-reference-failed.log` retains an incorrect reference for a fixture
  that explicitly overwrites its low result register. The corrected reference
  checks that zero while the copied argument still feeds the callee.
- `candidate-a100-20261123T173558Z.json` passes the first 512 hardware launches;
  `first-hardware-*` preserve its exact candidate, fixtures and harness. It
  predates source-line, rejection and expanded lexical/prototype controls.
  `first-hardware-audit.json` checks its complete 3,670,016 bytes and exact
  original candidate/harness/fixture identities without rebinding it.
- `candidate-offline.json` passes the final **1,080 complete executions**:
  fifteen sources × four levels × two seeds × three active lengths ×
  PTX/normal/poisoned SASS. Independent references use fixed-width integer and
  exact simple FP32 operations, complete buffers, guards and immutable inputs.
- `oracle-probe.json` and `oracle-audit.json` preserve fifteen actual local
  ptxas V12.8.61 sources, binaries, listings, logs and executable inventories.
  These observations do not replace hardware or independent value references.
- `candidate-a100-20261123T181957Z.json` passes the final **960 launches** over
  fifteen sources × four levels × OA/same-original-PTX controls × four active
  lengths × two seeds. Three 128-thread CTAs cover nested/repeated/guarded calls,
  lexical names, aliased slots, copied argument snapshots, forward prototypes,
  early returns, loops, side effects, FP32 and per-call local objects. The
  `nvcc_cuda` case starts from ordinary `.cu`; the worker invokes nvcc and
  retains that emitted PTX separately. Actual device: A100-SXM4-40GB; Python
  3.12.1, CUDA 12.8.1, ptxas V12.8.93.
- `promotion-audit.json` checks **6,881,280 retained bytes**, **60 exact native
  word/register comparisons**, original/emitted source identities, full
  manifest function inventories, resources and ownership. Two local callee
  objects reserve 64 bytes in OA, independently of ptxas optimization. Inputs,
  outer guards and inactive output lanes stay exact. Corrupted-output and
  mutated call-inventory controls reject.
- `wheel-audit.json` checks exact installed source/data and all sixty native
  comparisons in an isolated wheel subprocess. Prior installed policies and
  application native occurrences are checked separately before their source
  identities are carried forward. `base-application-continuity.json` retains
  1,598 complete prior comparisons; `complete_continuity.py` separately adds
  eighty prior Python-source kernels from their `generated.ptx` archive, for
  1,678 in total. The base report is preserved rather than relabeled. `validation.json` binds full suites/docs.

All earlier raw sources, failed comparisons and toolkit rejections remain
retained. No performance claim is made and no vendor fallback is enabled.

The final evidence review found that the void-call fixture's caller overwrote
the helper store. `pre-effect-*` retains that weaker source/gate. The final
fixture removes the caller overwrite; `void-effect-negative.json` confirms
that removing the sole callee store fails the independent full-output check.
A fresh source-exact 960-launch A100 gate and both full suites validate that
observable effect. The compiler modules are unchanged by this fixture repair.

## Reproduction and remaining boundaries

```sh
python compiler_exp/gap_closure/device_functions/offline.py --candidate
python compiler_exp/gap_closure/device_functions/oracle_probe.py
modal run compiler_exp/gap_closure/device_functions/run_modal.py
python compiler_exp/gap_closure/device_functions/audit.py compiler_exp/gap_closure/device_functions/candidate-a100-20261123T181957Z.json /tmp/device-functions-audit.json
python -m pytest -q tests/test_ptx_device_functions.py
uv build --wheel --out-dir compiler_exp/gap_closure/device_functions/wheel
python compiler_exp/gap_closure/device_functions/audit_wheel.py
```

Use the supported Python 3.12.10/Torch 2.7.1 environment for the full suite;
compiler-only Python 3.14.5 keeps its separate skips. Helpers must resolve
inside one PTX module. Input arguments and at most one return use 32/64-bit
scalar parameter slots at offset zero, with maximum 32 slots, depth 32 and
100,000 expanded statements. Static per-call local storage uses the existing
65,536-byte total-frame bound; ordinary register/spill limits still apply.
Device shared/module storage, recursion, indirect calls, register-passing or
aggregate call ABI, native stacks and separate device-library linking remain
open. Unused external declarations are inventoried; an unresolved reachable
symbol rejects. Unsupported instructions inside a helper retain their ordinary
backend diagnostics at the original source line.

The next [two-translation-unit experiment](../device_linkage/README.md) retains
actual original nvcc relocatable objects/device linking and an isolated source
merge. It passes a separate 48-launch A100 research gate and 196,608-byte audit
against original linked and merged-PTX controls. This is not production linkage;
symbol/visibility/dependency rules and normal source/CLI integration remain open.
