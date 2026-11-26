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
