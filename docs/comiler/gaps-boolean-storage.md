# Boolean array, shaped-buffer and shared storage

Status: implemented and integrated for the contract below.

## Original baseline and admission contract

The [logical scalar launch ABI](gaps-boolean-parameters.md) provides Boolean
parameters and host values. The [original probe](../../compiler_exp/gap_closure/python_boolean_abi/storage/baseline.json)
retains pre-implementation Array/Buffer/shared annotation rejections as baseline
tests, with exact source diagnostics and no user-code execution. Its preserved
CUDA artifacts establish the original byte-memory boundary; the production
evidence below establishes the current implementation.

A logical predicate uses one-byte physical storage, decodes nonzero loads and
stores canonical zero/one values. Array addressing, shaped strides and bounds,
shared-memory resources, Boolean store typing, initialization and barrier
convergence follow one storage contract. Evaluation order and side effects are
preserved. Numeric arithmetic, mixed Boolean/numeric unions and Boolean atomics
remain outside this implementation.

The [read-only layout probe](../../compiler_exp/gap_closure/python_boolean_abi/storage/layout-baseline.json)
preserves pre-implementation byte-memory forms and Boolean Buffer constructor
rejections as regression inputs. The [experiment dossier](../../compiler_exp/gap_closure/python_boolean_abi/storage/README.md)
defines global/shared encoding, aliasing, partial-grid, shaped-view, shared-memory
and public-run controls. Original CUDA setp.s16/selp.u16 remains a separate
[narrow-operation boundary](gaps-narrow-scalars.md). The broader 51-gap compiler
program remains incomplete.

## Validated implementation

The five staged modules now implement one shared byte-width rule for logical
predicates in raw arrays, shaped buffers and static shared arrays. Loads decode
nonzero bytes; stores require Boolean values and materialize canonical zero/one.
Checked loads yield false outside the view and checked stores skip. Rank 1..4
shape, explicit nonnegative stride, read-only broadcast, writable nonoverlap,
alignment/span and existing shared limit/convergence rules remain enforced.
Public `dtype: bool` buffers require JSON true/false initialization and canonical
random values; wrong numeric, scalar or raw-pointer roles reject. Native ownership
binds the type/shape contract and exact original source graph through normal builds.

The [primary A100 gate](../../compiler_exp/gap_closure/python_boolean_abi/storage/candidate-a100-20261126T065029Z.json)
and [additional shape gate](../../compiler_exp/gap_closure/python_boolean_abi/storage/shapes-a100-20261126T065356Z.json)
pass **5,280 original-source launches** and **528 actual public buffer launches**.
Independent production audits check **22,302,720 complete bytes** and **264 exact
native outputs**, with no timing claim. Raw noncanonical input bytes, canonical
stores, aliases, lazy atomic effects, partial grids, sparse/transposed/broadcast
views, checked false loads/skip stores and mixed-width shared permutations retain
independent full-byte references. All **208 focused tests** and **12,672 offline
PTX/normal/poisoned executions** pass. Eight stale rejection cases have exact
contract updates across three fixtures; no numerical bits or tolerance changes.

The [promotion record](../../compiler_exp/gap_closure/python_boolean_abi/storage/promotion.json)
binds the exact candidate. Full suites pass **5,136 supported tests** and
**4,651 compiler-only tests with fifteen skips**. Fresh replay reproduces all
**4,214 preceding native outputs** exactly. Installed-wheel gates preserve 132
source/data identities and all 264 native outputs; public remote and documentation
gates pass. The [validation record](../../compiler_exp/gap_closure/python_boolean_abi/storage/validation.json)
binds exact final sources and evidence. Direct predicate comparisons, helper arrays, Boolean arithmetic,
Boolean atomics and re-export-only kernel entry discovery remain separate boundaries.
Static shared memory retains its existing uninitialized-storage contract and
conservative possible-writer/barrier checks; these tests do not prove arbitrary
initialization or race freedom. The full 51-gap program remains incomplete.


The [measured-policy continuity audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261126T070403Z.json)
passes all 49 installed binaries, controls and resources and 98 retained sessions
against the original frozen commit before rebinding current source identities.
The actual public CLI/Modal publication also passes with exact source/logical/native
custody. These gates make no fresh numerical GPU or timing claim.

The next [Boolean comparison study](gaps-boolean-comparisons.md) retains four original
source rejections for direct scalar/storage/helper/chained comparisons after storage
admission. It has no candidate or numerical/hardware support claim.
