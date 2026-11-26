# Boolean global, shaped and shared storage

Status: three original storage annotation rejections recorded after logical scalar
launch admission; no candidate or production change in this subcycle.

The [Boolean scalar launch contract](../scalar/README.md) connects canonical host
Boolean values to internal predicates over u8 parameter slots. Storage additionally
requires byte addressing/alignment, load decoding, canonical stores, array and
shaped-buffer metadata, verification allocation, host buffer binding and shared
initialization/convergence semantics. Treating predicate registers as ordinary
u32 elements would give the wrong byte span and neighboring memory effects.

The [original source](source.py) contains Array(bool), shaped Buffer(bool) and a
cross-thread shared Boolean exchange, plus a top-level exception. The
[baseline](baseline.json) records the earliest original diagnostics without
executing the module. Exact prior independent CUDA/PTX/cubin/listing/tool logs
already establish byte Boolean global storage; their hashes are checked without
rewriting them or claiming new numerical/GPU execution.

The solution must define a logical predicate with one-byte physical storage,
nonzero load decoding and canonical zero/one stores. Byte offset/stride/span and
shared padding accounting must agree. Every store requires an exact Boolean
value; numeric zero/one must use explicit conversion. Reads and selected operands
evaluate once in original order, including atomic or helper effects elsewhere.
Boolean arithmetic/unions and Boolean atomics remain explicit separate contracts.

The typed verifier must independently initialize canonical Boolean arrays and
check all byte outputs/unchanged inputs/guards/tails. The public buffer schema
needs a logical Boolean dtype and true/false values with exact byte initialization;
numeric or wrong-dtype storage cannot silently acquire that role. Raw byte buffers
retain their own contract. Native/source ownership must bind the Boolean array
type and shaped-buffer layout, including read-only/overlap/bounds and zero/skip
semantics. Tampered metadata or bindings reject.

Actual global/shared byte encodings already exist, but their use for Boolean
source arrays must pass independent original sources, poisoned/partial-grid
neighbors, read/write aliases, lazy accesses, static shapes/strides, shared
permutations/initialization/barriers, canonical host data, actual ptxas/A100 and
public local/remote run gates. Prior native/policy/installed/full-suite/docs gates
remain required before production promotion.

CUDA-emitted setp.s16/selp.u16 is the separately retained
[narrow register-operation gap](../../narrow_scalars/README.md). Source storage
support must not imply those original CUDA operations are implemented.

The [read-only layout probe](layout-baseline.json) additionally records the
existing declared byte load/store forms and actual predicate/Boolean Buffer
constructor rejections. It does not bypass them with a synthetic accepted
Buffer. A future implementation should carry an explicit physical element kind
alongside the logical predicate kind, use one shared element-width rule across
raw/shaped/shared addressing and metadata, and validate that rule in host buffer
bindings. Reusing an internal predicate kind as an implicit four-byte numeric
storage type is outside the required contract.

The first shared probe reversed the documented `(type, count)` arguments and
rejected the integer annotation before reaching `bool`. Its source, observer and
raw record remain in `first-reversed-shared-arguments-*`. The corrected baseline
uses `shared_array(bool, 128)` and requires every diagnostic to identify `bool`.

## Candidate, validation and production

Five exact staged modules implement shared logical/physical width rules, Boolean
Array/Buffer annotations, nonzero byte decoding, canonical byte stores, one-byte
shared alignment and checked false-load/skip-store semantics. Type metadata and
existing native custody retain the logical Boolean role. Public buffer dtype
`bool` requires actual true/false initialization; random values are canonical and
wrong numeric or untyped pointer bindings reject. Production imports no experiment.

The [primary original-source A100 gate](candidate-a100-20261126T065029Z.json) passes
4,320 launches and 432 actual public Boolean buffer launches across 18 raw/shaped/
shared/effect kernels, three unexecuted same-file/import/package trees, four OA
levels and an independently assembled ptxas control. The [separate shape gate](shapes-a100-20261126T065356Z.json)
passes another 960 launches and 96 public launches for rank 1/3/4 views and observable
out-of-bounds false loads. Independent [primary](candidate-audit.json) and
[shape](shapes-audit.json) audits check 22,302,720 complete bytes and all 264 native
word/register/frame outputs. Noncanonical raw inputs, canonical stores, sparse/
transposed/broadcast layouts, skip stores, byte guards/tails, mixed-width shared
padding, shared permutations/barriers, exact raw alias and lazy atomic effects
retain independent references. No timing or speed claim.

[Primary offline](candidate-offline.json) and [shape offline](shapes-offline.json)
gates pass 12,672 complete PTX/normal/poisoned executions. All 208 focused tests
pass. [Frozen original-source replay](baseline-replay.json) retains 54 original
annotation rejections; the corrected first source baseline separately retains all
three original type boundaries.

The exact candidate is now promoted, with [promotion record](promotion.json),
[production audit](promotion-audit.json) and [production shape audit](promotion-shapes-audit.json).
Eight old rejection cases have [explicit type-contract updates](test-contract-updates.json)
across three fixture files. Helper arrays and bare array values retain their actual
semantic rejections; formerly unsupported storage annotations now exercise numeric
stores into Boolean storage. No positive reference bits or tolerances changed.
Full suites pass 5,136 supported tests and 4,651 compiler-only tests with fifteen
skips. Fresh replay reproduces all 4,214 preceding native outputs exactly. The
installed wheel preserves 132 source/data identities and all 264 native outputs;
public remote publication and live/exported docs pass. The [validation record](validation.json)
binds exact final sources and evidence; the full 51-gap program remains open.

## Retained rejected probes and observer corrections

The first checked-buffer candidate tried to render Python `False` as an integer
immediate. Exact PTX/compiler rejection is retained in `first-false-immediate-*`;
checked missing loads now form zero in an integer register and explicitly decode
it into a predicate. Direct predicate comparisons remain unsupported; the retained
`first-predicate-comparison-*` source needs explicit `i32` conversions to compare
Boolean values in the admitted comparison path. This cycle does not claim direct
Boolean comparison or re-export-only kernel-entry discovery. A first third source
tree contained only re-exports and found no kernels; `first-reexport-only-entry-*`
retains it, while the final tree has actual top-level kernels in a package.

The first shared-bounds negative reached missing-producer analysis earlier than
the bounds diagnostic; its retained source/log show that rejection. The final
negative performs an out-of-range store and read and reaches the bounds contract.
A fixture observer cached the old exception class while installing snapshots; the
retained failure is resolved by importing the class after each installation.
The first shape auditor checked all 960/96 exact domains and bytes but copied the
primary count into its summary; promotion rejected it. The corrected summary
observer reran the entire audit. The first policy observer stopped at a stale
intermediate source snapshot; the next observer composes the exact validated
baseline/candidate chain before checking all measured binaries and sessions.


The [measured-policy audit](../../foundations_20261121/audit-20261126T070403Z.json)
passes all 49 frozen/current exact policy executables and 98 retained sessions before
source identity rebinding. The [public CLI/Modal build](normal-remote-audit.json)
passes exact original dependency/native/logical storage contracts. Its local client
reported a final-log polling timeout after artifacts were retrieved; the successful
manifest and independently validated native publication are retained.
