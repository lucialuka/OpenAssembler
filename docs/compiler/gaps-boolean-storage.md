# Boolean array, shaped-buffer and shared storage

Status: original source boundaries established; no candidate.

The [logical scalar launch ABI](gaps-boolean-parameters.md) supplies Boolean
parameters and canonical host values. G03/G07/G09/G12 additionally need Boolean
Array/Buffer/shared storage. The [original probe](../../compiler_exp/gap_closure/python_boolean_abi/storage/baseline.json)
retains three annotation rejections with exact file/line diagnostics and no user
execution. It checks the original independent CUDA byte-storage artifact identities
without rewriting them or claiming fresh numerical/hardware execution.

A logical predicate needs one-byte physical storage, nonzero load decoding and
canonical zero/one stores. Source array addressing, shaped strides/span/bounds,
shared padding/resources, exact Boolean store typing, initialization and barrier
convergence must agree. Reads and effects retain original order and once-only
evaluation. Numeric arithmetic/unions and Boolean atomics remain separate work.

The public Boolean buffer dtype must validate true/false initialization and exact
byte layout; numeric/wrong-dtype buffers cannot silently acquire the role. Typed
verification needs independent canonical byte data and full output/input/guard/
tail snapshots. Source/manifest/native ownership must preserve the logical array
type and shaped layout through local/remote publication and reject tampering.

The [experiment dossier](../../compiler_exp/gap_closure/python_boolean_abi/storage/README.md)
defines global/shared encodings, alias/lazy/poisoned/partial-grid/shaped/read-only/
overlap/shared-exchange controls and actual ptxas/A100/public-run/full integration
gates. Existing byte instruction coverage does not by itself establish Boolean
source semantics. Original CUDA setp.s16/selp.u16 remains a separate
[narrow-operation boundary](gaps-narrow-scalars.md). The full 51-gap program remains
incomplete.

The [read-only layout probe](../../compiler_exp/gap_closure/python_boolean_abi/storage/layout-baseline.json)
records existing declared byte memory forms and actual predicate/Boolean Buffer
constructor rejections. The solution needs an explicit logical/physical element
mapping and one width rule across raw/shaped/shared addressing, metadata,
verification and host bindings; it must preserve the present gate until the
complete storage contract is validated.
