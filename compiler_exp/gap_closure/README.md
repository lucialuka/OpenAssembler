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

## November 23 continuation

The [current integration](integration_20261123/README.md) passes 2,710 supported
tests and 2,253 compiler-only tests with 14 skips. Ordered/shared atomics and
two value-lifetime fixes pass separate 472/768-launch A100 gates. Existing
wide/buffer/primitive/ABI/build/optimizer/checkpoint dossiers are integrated into
the register. All 49 measured policies retain native bytes; changed model
executables pass fresh full-state/CLI hardware execution. Raw failures and
interrupted runs remain explicitly separate from completed gates.

The subsequent [decode cycle](decode_specialization/README.md) promotes explicit
bounded exact-length specialization after 85 focused CPU checks and complete
A100 prediction/cache/ownership gates. Its full suite passes 2,738 supported
tests and 2,253 compiler-only tests with 15 skips. The corresponding dossier
keeps G39 partial beyond the declared factory/context/budget contract.

The next [full-warp shuffle cycle](warp_collectives/README.md) adds bounded
indexed/up/down forms using recovered templates, passes 1,792 A100 launches,
and preserves installed policies plus 1,234 earlier application kernel
occurrences. Final integration passes 2,810 supported tests and 2,325 compiler-only
tests with 15 skips. A stale indexed-shuffle rejection is replaced with a real
partial-mask rejection; both initial failed logs remain retained.

The [static local-storage cycle](local_storage/README.md) adds explicit private
frames, scalar32 access and donor-independent frame metadata after 704 A100
launches. Full integration passes 2,868 supported tests and 2,383 compiler-only
tests with 15 skips. Policy and application audits preserve original native
executables; G13 remains partial beyond the declared local contract.

The subsequent [scalar spill cycle](spilling/README.md) recovers explicit and
automatic register pressure with 384 A100 launches, complete private-state and
resource/ownership gates. Full integration passes 2,883 supported tests and
2,398 compiler-only tests with 15 skips. A matched synthetic resource comparison
retains the cost of additional spilling; no lower-register policy is selected.

The [aligned-group/predicate cycle](spilling_groups/README.md) extends that
allocator after retaining twenty baseline failures. Another 320 A100 launches
pass full guarded vector/predicate state, resources and ownership checks;
read-only audits bind 83,787,776 retained bytes and twenty native comparisons.
Successful prior scalar code remains identical. Slot reuse and general
pressure/occupancy policy remain open.
Its [integration](spilling_groups/validation.json) passes 2,898 supported tests
and 2,413 compiler-only tests with 15 explicit skips.

The [pressure scheduling cycle](pressure_schedule/README.md) next compares
virtual operation order before allocation, preserving dependencies and fixed
boundaries. A100 records contain 2,144 correctness, 1,323 timed and 189 warmup
launches across frozen/current/same-PTX controls and final-source admission.
Complete bytes, resources and paired medians are audited independently.
Additional host compilation cost, the retained weaker heuristic and a guarded
exit admission repair are recorded explicitly. A separate
[decimal-literal counterexample](float_literals/README.md) starts the subsequent G11 cycle.

That [literal cycle](float_literals/README.md) implements binary64 decimal
representation followed by use conversion, raw payload/width behavior and
invalid-context rejection. Six oracle matrices, independent rational controls,
360 PTX/SASS executions and 320 A100 launches precede promotion. The production
audit checks 28,737,536 bytes and twenty native comparisons; broader G11 remains
partial under its original numerical and operand-conformance exit gate.
Its [integration](float_literals/validation.json) passes 2,974 supported tests
and 2,489 compiler-only tests with 15 explicit skips.

The [typed-expression cycle](constant_expressions/README.md) adds a bounded
scalar operand parser with integer precedence/casts and binary64 intermediate
rounding. Independent typed trees and full CPU/A100 bytes precede promotion.
The first hardware report remains failed, exposing local/worker toolkit
differences; final diagnostics retain unresolved cases explicitly. A separate
depth counterexample led to a new exact-source 384-launch A100 gate and final
integration. The original broad G11 exit gate remains partial.
Its [integration](constant_expressions/validation.json) passes 3,042 supported
tests and 2,557 compiler-only tests with 15 explicit skips.

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

The [scalar shared-width cycle](shared_widths/README.md) adds exact byte/halfword/
64-bit exchanges after four recovered store forms, 12,096 varied disassemblies,
a failed barrier schedule gate and the repaired 1,216-launch A100 gate. The
production audit checks 34,865,152 complete retained bytes and 76 native
comparisons. A separate 336-row width matrix, isolated wheel check and final
[integration](shared_widths/validation.json) keep encoding, hardware and prior
code continuity distinct. Full suites pass 3,099 supported tests and 2,614
compiler-only tests with 15 skips. G12 and the full 51-gap program remain partial.

The [typed Python shared-array cycle](python_shared_wide/README.md) adds
naturally aligned i64/u64 arrays and padding-aware 48 KiB admission. A failed
conditional-writer A100 gate led to an independently isolated forward-phase
BSSY/BSYNC repair. The final 1,280-launch gate, 20,971,520-byte production audit,
eighty native comparisons, isolated wheel and 62-kernel frontend continuity
precede full integration. Suites pass 3,176 supported tests and 2,691 compiler-
only tests with fifteen skips. General cyclic convergence and memory safety
remain open; no performance claim is made.

The [direct PTX function cycle](device_functions/README.md) resolves scalar
module-local helpers by static specialization with lexical slots, copied
arguments, early returns and distinct local objects. Fifteen original-source
cases pass 960 A100 launches, including ordinary CUDA noinline input. The
production audit checks 6,881,280 bytes and sixty native comparisons; a wheel
replay and 1,678 prior native occurrences remain separate. Final suites pass
3,229 supported tests and 2,744 compiler-only tests with fifteen skips. Native stacks and device
linking retain their original G16/G10 exit gates.

The [source-library promotion](device_linkage/promotion/README.md) adds public
PTX/CUDA `--device-library` integration, deterministic scalar resolution and
per-unit build/dependency custody. The original-linked-control gate passes 576
A100 launches; complete production auditing checks 4,128,768 bytes and 48 native
comparisons. Normal CLI/remote publication and an isolated wheel pass. Full
suites pass 3,274 supported tests and 2,789 compiler-only tests with 15 skips.
Binary object/archive libraries and general host/device linking remain open.

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

## November 24 scalar 64-bit selection

The [wide-select cycle](wide_select/README.md) promotes the exact validated
`selp.b64/u64/s64/f64` lowering using existing scalar SEL words. Original CUDA
noinline and legal typed PTX controls pass 1,480 A100 launches. The independent
production audit checks 15,155,200 bytes and 148 native comparisons. Initial
invalid FP64 sources and the focused diagnostic mismatch remain retained.
The [canonical dossier](../../docs/compiler/gaps-wide.md#scalar-64-bit-conditional-selection)
keeps the admitted bit-copy contract and broader G11 limits explicit.

## November 24 low64 multiplication in PTX/CUDA and typed Python

The [product promotion cycle](wide_multiply/promotion/README.md) extends the
retained backend experiment through typed Python `i64/u64` products and `*=`,
including helpers/loops/casts and source/parameter squares. Two independent A100
gates pass 1,560 launches, 15,974,400 complete bytes and 156 native outputs.
The exact validated selector/admission/frontend sources are production code;
full integration gates remain separate from the hardware evidence. The
[canonical dossier](../../docs/compiler/gaps-wide.md#low-64-bit-integer-multiplication)
keeps broader G03/G11 limits explicit.

## November 24 static local-package Python helper graphs

The [promotion cycle](python_module_helpers/promotion/README.md) preserves
per-module scalar semantics and original source/dependency/diagnostic identity
through alias/relative/re-export helper calls and ordinary local/remote builds.
The final A100 gate passes 600 launches, 5,529,600 bytes and sixty native outputs.
The complete earlier-native replay passes 2,090 occurrences. First full-suite
frontend-routing failures remain retained alongside the production repair; reruns
pass 3,538 supported tests and 3,053 compiler-only tests with fifteen skips. The [canonical dossier](../../docs/compiler/gaps-python.md#static-local-package-device-helpers)
keeps runtime packages/captures/host effects outside this validated source subset.
The next [static-value hypothesis](python_static_values/README.md) has independent
finite functional evidence and remains outside production.

## November 24 explicit static numeric and tuple declarations

The [original-source promotion](python_static_values/promotion/README.md) replaces
the isolated normalization hypothesis with defining-module Final specialization,
finite arithmetic/tuple indices/slices/len and exact original source/value/use
identity. The final A100 gate passes 1,440 launches, 9,338,880 independent bytes
and 144 native outputs. Full suites pass 3,656 supported tests and 3,171 compiler-only tests with fifteen
skips; the installed wheel and 2,150 earlier native occurrences pass their distinct
source/code/ownership gates. The [canonical dossier](../../docs/compiler/gaps-static-values.md)
keeps mutable captures/runtime containers/general metaprogramming outside this
validated source contract; broad G05 and the 51-gap program remain incomplete.

## November 24 original finite tuple iteration

The [promotion cycle](python_static_loops/promotion/README.md) implements static
numeric tuple iteration with observable target, empty/nested/break/continue/else
semantics in original source. All 2,400 A100 launches, 16,465,920 independent bytes
and 240 native outputs pass; the old normalized prototype remains separate.
Full suites pass 3,806 supported tests and 3,321 compiler-only tests with fifteen
skips; the installed wheel regenerates 240 outputs and all 2,294 earlier native
occurrences retain exact words/registers in the linked integration record.
The [canonical dossier](../../docs/compiler/gaps-static-loops.md) keeps runtime
containers/arbitrary iterables/general control and host effects outside this
bounded G05/G06 support.

## November 24 original pure scalar helper returns

The [original-source cycle](python_helper_returns/promotion/README.md) implements
branch-complete, early and loop return values with every-path typing, live-path
initialization, copied callers and exact original source/native ownership. Its
final A100 gate passes 2,400 launches, 16,465,920 independent bytes and 240 native
outputs; 4,320 offline executions, 768 generated-control executions and 425 focused
regressions pass. Full suites pass 3,962 supported tests and 3,477 compiler-only
tests with fifteen skips. The installed wheel checks 132 identities/240 outputs;
240 prior tuple outputs regenerate exactly and 2,294 earlier replay results bind
to unchanged backend inputs. The earlier normalized prototype and first passed
GPU payload remain separate. The [canonical dossier](../../docs/compiler/gaps-helper-returns.md)
retains runtime range/while else and broader Python effects/ABI as open G04/G06 work.

## November 24 finite ranges and runtime loop else

The [original-source cycle](python_range_control/promotion/README.md) guards
finite i32 range exhaustion before nonunit counter wrap and preserves range/while
else with original target/bounds snapshots, nearest-loop exits, helper returns,
conservative initialization, convergence and source/native ownership. It passes
2,400 A100 launches, 17,203,200 complete independent bytes, 240 native outputs,
4,320 offline executions, 128 generated cases / 1,536 executions and 920 frontend
tests. The first bounded counterexamples and minimized O0/O3/runtime/static loop
study remain separate. Full suites pass 4,112 supported tests and 3,627
compiler-only tests with fifteen skips; the installed wheel checks 132 identities
and 240 native regenerations. The linked integration record binds final sources,
proofs and documentation. Dynamic steps/wide bounds and arbitrary control/effects
remain open in the [canonical dossier](../../docs/compiler/gaps-range-control.md).

## November 24 original typed condition chains

The [original-source cycle](python_comparison_chains/promotion/README.md) implements
condition-chain single evaluation, short-circuit atomic/memory snapshots,
pairwise exact literal contexts and complete original source/expression/native
ownership. Final gates pass 2,400 A100 launches, 15,482,880 independent bytes,
240 exact native outputs, 4,320 offline executions, 128 generated chains /
1,536 executions and 159 new tests. The prototype and first numerical GPU payload
remain separate from the final standalone ownership repair. Full suites pass
4,271 supported tests and 3,786 compiler-only tests with fifteen skips. All 1,079
frontend tests, installed wheel and public Modal publication pass. 240 preceding
outputs regenerate exactly, while 2,774 earlier occurrences retain their separate
proof/archive/backend-input binding. The linked integration record binds final
sources and evidence; general Boolean values and arbitrary Python effects remain open in the
[canonical dossier](../../docs/compiler/gaps-comparison-chains.md).

## November 24 Boolean values and predicate control-flow readiness

The [original-source cycle](python_boolean_values/promotion/README.md) promotes
contextual Boolean values and lazy scalar operand selection with full value/source/
native ownership. The first GPU gate exposes 54 real predicate-loop mismatches;
a CFG readiness repair fixes them without changing references, instructions or
allocation. The final gate passes 2,400 A100 launches, 16,957,440 independent bytes
and 240 exact current outputs. Only twelve outputs change stall/yield fields in
the failed/repaired transition. All 167 new regressions / 1,246 focused tests and
49-policy/98-session exact continuity pass. Full suites pass 4,438 supported tests
and 3,953 compiler-only tests with fifteen skips. A fresh direct replay passes all
3,254 earlier native comparisons exactly. Installed-wheel and public Modal checks
retain original ownership, and the integration record binds final sources/proofs/
documentation. Boolean ABI/type unions and the full 51-gap program remain open in the [canonical dossier](../../docs/compiler/gaps-boolean-values.md).

## November 24 logical Boolean helper signatures

The [logical helper cycle](python_boolean_abi/helpers/README.md) admits exact
Boolean scalar parameters/returns in pure static helper expansion, caller/return
isolation and original source/native custody. Its original-source A100 gate passes
2,400 launches / 16,220,160 independent complete bytes / 240 exact native outputs;
all 155 new regressions pass. Existing tests and backend/installed-policy inputs
are unchanged. Full suites pass 4,593 supported tests and 4,108 compiler-only tests
with fifteen skips. Wheel/public Modal/live/exported-documentation gates pass; the
[validation record](python_boolean_abi/helpers/validation.json) binds final identities.

The [next mixed-width layout experiment](python_boolean_abi/parameter_layout/README.md)
keeps the physical byte-parameter boundary in research. It observes actual toolkit
slots and explicit original parser/form rejections, without a production candidate
or numerical/GPU claim. All 51 original exit gates remain tracked and incomplete.

## November 25 narrow scalar entry ABI

The [narrow-entry cycle](narrow_parameters/README.md) promotes byte/halfword scalar
parameter layout/extraction/extension with matching verifier and normal explicit
launch bindings. Its original-source gate passes 14,400 A100 launches and 240
explicit-run launches; the independent audit checks 67,461,120 complete bytes and
240 exact native outputs. All 171 new tests pass, with no existing test changes.
49 measured policies and 98 sessions remain exact before identity rebinding;
public Modal publication passes. Fresh native replay passes all 3,734 direct
comparisons exactly. Full suites pass 4,764 supported tests and 4,279 compiler-only
tests with fifteen skips. Installed-wheel/live/exported-docs gates pass;
[validation.json](narrow_parameters/validation.json) binds final identities. The next
[logical Boolean scalar](python_boolean_abi/scalar/README.md) and original
[CUDA narrow-operation](narrow_scalars/README.md) studies retain separate boundaries.

## November 25 logical Boolean kernel ABI

The [Boolean parameter cycle](python_boolean_abi/scalar/README.md) promotes source
bool annotations, canonical byte launch values and exact typed metadata/native
custody. Its original-source A100 gate passes 9,600 launches plus 960 actual public
Boolean run-spec launches / 69,746,688 complete bytes / 240 exact native outputs.
All 164 new regressions pass; two exact stale rejection fixtures now use unsupported
Boolean array annotations. Policy binaries/controls/resources and 98 sessions stay
exact before rebinding. Complete integration is running and will be bound by
[validation.json](python_boolean_abi/scalar/validation.json). Boolean storage remains open.
