# Logical Boolean scalar launch contract

Status: promoted; full integration, installed/public remote and documentation gates pass.

The [physical narrow-entry cycle](../../narrow_parameters/README.md) establishes
u8 parameter storage and host argument width. This subcycle adds the logical
meaning of a Python `bool` annotation. The [original source](source.py) includes
adjacent flags, numeric neighbors, wide slots, ordinary pure helpers, local
reassignment and guarded output; a top-level exception forbids dynamic discovery.
The [original baseline](baseline.json) retains all three frontend diagnostics and
separately confirms physical u8 entries compile at every level. No fresh GPU or
numerical result is claimed by this baseline.

The solution must keep logical predicates distinct from physical bytes. Kernel
entry loading should decode a byte into the predicate kind, while typed source
metadata records the original Boolean role and physical u8 slot. Numeric arrays
and scalar neighbors retain their exact types/offsets. Annotation shadowing,
helper/return typing, caller copies and source/native custody remain required.

The public launch schema should require an actual Boolean JSON value for a
Boolean source parameter and encode exactly zero/one as one byte. An integer,
buffer or wrong-width scalar must not silently acquire that logical role. Raw PTX
u8 remains an arbitrary byte type. Typed verification needs independent canonical
Boolean values and complete numeric/guard/tail checking. A parameter named `n`
must not become an integer dimension merely because its logical type is Boolean.

Normal manifests/native ownership must retain and validate this logical/physical
parameter contract; parsing a byte declaration alone cannot reconstruct a source
Boolean role. Tampered contract, source/dependency metadata or host binding must
fail rather than fall back. Original same-file/imported/remote sources, every flag
combination, numeric signed/wide neighbors, partial grids, local reassignment and
helper calls need independent complete-byte PTX/native/A100/public-run gates.
Installed/remote/full-suite/native continuity and docs remain promotion gates.

Boolean Array/Buffer/shared byte storage has additional load/store, layout,
canonicalization, safety and launch-buffer contracts. It remains separately open.

## Validated implementation

The [canonical dossier](../../../../docs/compiler/gaps-boolean-parameters.md) records
logical predicate versus physical byte roles. Five exact candidate modules now
implement source annotation/entry decoding, canonical typed verification, normal
build metadata, deep value/native ownership and public Boolean argument binding.
Production imports no experiments; physical byte layout and native selection are
unchanged.

The [original A100 gate](candidate-a100-20261125T074133Z.json) passes **9,600 original
source launches** plus **960 actual public Boolean run-spec launches**. The
independent [production audit](promotion-audit.json) checks **69,746,688 complete
bytes** and **240 exact native word/register/frame outputs**, including every
flag pair, source signature and host Boolean role. [Offline](candidate-offline.json)
passes 17,280 complete executions; all 164 new regressions and fourteen negative
source controls pass. The [original replay](original-replay.json) preserves the
three initial rejections; [full frozen replay](baseline-replay.json) retains sixty
original rejections across the source trees without rewriting earlier observations.

Two existing [rejection fixtures](test-contract-updates.json) now use genuinely
unsupported Boolean array parameters; their earlier scalar-parameter rejection
became stale. New positive numerical/host/metadata/native tests validate scalar
behavior; no reference bit, tolerance or numerical class is weakened.

The [policy audit](../../foundations_20261121/audit-20261125T074730Z.json) keeps all
49 measured binaries/controls/resources and 98 original sessions exact before
identity rebinding. A fresh replay reproduces all 3,974 previous native outputs
exactly. Full suites pass 4,928 supported tests and 4,443 compiler-only tests with
fifteen skips. Installed/public remote and live/exported documentation gates pass.
The final [validation record](validation.json) binds their exact identities.

## Retained observer failures

The first A100 client lost its configured local API proxy and retrieved no result.
The observed app stopped with zero tasks before the exact direct-API rerun. The
first independent auditor completed every byte/native check but shadowed its
parsed argument object while forming the report; its source/log remain retained.
The corrected observer only renames that local variable and reruns every check.
No source, candidate, input, reference or tolerance changes to resolve either issue.

The first separate public Modal build lost its API heartbeat and result polling
connection. Its log and compile-error payload remain retained. This is transport
failure without a numerical or compiler diagnosis; only a retrieved successful
build can satisfy that integration gate.

The subsequent [storage cycle](../storage/README.md) retains three original
Array/Buffer/shared rejections and the complete byte/storage/host-buffer/convergence
contract. Source scalar admission does not imply those storage forms are supported.
The maintained original-source replay command is `replay_original.py`; the original
`probe.py` remains the dated pre-admission observer and is not rerun as a current
rejection check after promotion.

The [generated-expression gate](control-properties.json) passes 128 Boolean
parameter/effect trees and all four flag pairs / 6,144 complete executions,
seed 67439, independently evaluated truth/atomic/read state. The installed wheel
preserves 132 source/data identities and 240 exact native regenerations. These
checks retain explicit CPU/installation scope rather than imply additional GPU work.

The final successful public build uses `normal-remote-build/manifest.json` and its
exact native ownership ledger. That directory also retains the earlier
`compile-error.json` transport diagnostic; the manifest has no `build_failed`
entry and the successful audit binds its current hash. The separately copied
first-failure payload/log keep the earlier transport state explicit.
