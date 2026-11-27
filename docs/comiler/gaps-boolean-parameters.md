# Logical Boolean kernel parameters

Status: implemented; full integration, installed/public remote and documentation gates pass.

## Gap and contract

The [physical narrow-entry ABI](gaps-narrow-parameters.md) supports a raw byte
parameter, while Python kernels still reject `flag: bool`. The [logical helper
extension](gaps-boolean-helpers.md) only carries internal predicates through pure
static calls. This G03/G10 cycle supplies the missing source-to-launch contract.

An unshadowed built-in `bool` annotation denotes a logical predicate in a kernel
body and a physical `.param.u8` slot. Entry code loads the byte and compares it
to zero. Source type metadata explicitly records `dtype: bool`, separately from
the physical parameter declaration. Existing numeric scalar/array neighbors,
alignment, helper/caller copying, local reassignment, convergence and effect rules
remain required. Array/Buffer/shared Boolean storage and arbitrary type unions
remain separate work.

The public `--spec` argument is `{"bool": true}` or `{"bool": false}`. It requires
an actual JSON Boolean and a matching logical source contract, then encodes a
canonical zero/one byte. Integer zero/one, a buffer, a raw u8 key or a wrong-width
scalar cannot acquire the Boolean role. A Boolean argument also cannot relabel a
raw PTX byte. Raw byte parameters retain their arbitrary integer value contract.
The auto-harness requires an explicit spec for logical Boolean parameters.

Typed verification generates canonical Boolean values and packs one byte. A
Boolean named `n`, `count` or another dimension spelling stays Boolean. Host
dimension heuristics do not override the logical type. Original numeric array
contents, inactive tails, counter effects and guards remain independently checked.

## Candidate implementation and ownership

The [experiment](../../compiler_exp/gap_closure/python_boolean_abi/scalar/README.md)
stages five exact snapshots: Python annotation/entry lowering, typed verification,
normal build metadata, native ownership and explicit launch binding. It adds no
physical ABI, opcode, template, scheduler or resource limit.

The builder validates source type metadata even without `--verify` and records a
`logical_parameter_contract` for Boolean entries. Native ownership validates and
deep-copies that contract along with the physical signature. Its launch validator
checks the exact retained PTX hash and matching parsed logical contract; missing,
mutated, stale or mismatched contracts reject. Checked Buffer validation remains
separate, and generic verification keeps its explicit shaped-buffer boundary.

## Independent experiments

Three original same-file/import/package trees contain twenty kernels and top-level
exceptions, proving static discovery must not execute user code. Controls include
every flag pair, adjacent byte slots, numeric/wide neighbors, helper copies,
predicate loops, local reassignment, FP32 NaN/infinity/signed-zero selection and
short-circuit atomic effects. The [offline gate](../../compiler_exp/gap_closure/python_boolean_abi/scalar/candidate-offline.json)
passes **17,280 complete executions / 240 rows** through four levels and
PTX/normal/poisoned engines against independent raw-byte references.

The [frozen replay](../../compiler_exp/gap_closure/python_boolean_abi/scalar/baseline-replay.json)
reproduces sixty original annotation rejections. All fourteen original source
type/storage/effect/convergence negatives reject. **164 candidate regressions**
pass, including exact host roles/ranges, malformed metadata, a Boolean dimension
name and source/native custody. Existing capability rejection fixtures require
exact contract updates only when the formerly unsupported scalar ABI is admitted;
positive numerical and host/ownership controls remain mandatory.

The [A100 gate](../../compiler_exp/gap_closure/python_boolean_abi/scalar/candidate-a100-20261125T074133Z.json)
passes **9,600 original-source launches** and **960 actual public Boolean run-spec
launches** using ordinary CLI builds and the real serialized worker. The
[independent production audit](../../compiler_exp/gap_closure/python_boolean_abi/scalar/promotion-audit.json)
checks **69,746,688 complete bytes** and **240 exact native word/register/frame
outputs** with original source/type/native identities and complete output/input/
counter/guard/tail state. Two exact [existing fixture updates](../../compiler_exp/gap_closure/python_boolean_abi/scalar/test-contract-updates.json)
replace the newly stale scalar annotation rejection with genuinely unsupported
Boolean array annotations, without changing numerical reference bits or tolerances.
Fresh replay reproduces all **3,974 previous native outputs** exactly. All 49 measured
policies and 98 original sessions retain exact binaries, controls and resources.
Full suites pass **4,928 supported tests** and **4,443 compiler-only tests with
fifteen skips**. Installed/remote builds and live/exported documentation pass;
the [validation record](../../compiler_exp/gap_closure/python_boolean_abi/scalar/validation.json)
binds their final identities.
No timing or general Python/storage/ABI claim follows from this candidate.

The first A100 client disconnected from its local API proxy before a result was
retrieved. Its log and authoritative stopped app state remain distinct from
numerical evidence; the direct-API rerun reuses the exact candidate and references.
The full 51-gap program remains incomplete.

All 49 measured policy binaries/controls/resources and 98 original sessions remain
exact in the [current policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261125T074730Z.json)
before compiler identity rebinding. The first public remote-build attempt lost its
API heartbeat/result-polling transport; its retained failure is not compiler or
numerical evidence. A separate successful publication is still required.

The subsequent [Boolean storage study](gaps-boolean-storage.md) retains the current
original Array/Buffer/shared annotation rejections and defines the byte-layout,
canonical load/store, host buffer type and shared initialization/convergence gates.
A logical Boolean parameter does not imply Boolean array storage support.

The additional [generated-expression gate](../../compiler_exp/gap_closure/python_boolean_abi/scalar/control-properties.json)
passes 128 Boolean parameter/effect trees across all four flag pairs, with 6,144
complete PTX/normal/poisoned executions and an independent recursive truth/atomic/
read-state reference (seed 67439). This adds CPU semantic evidence, without a new
hardware or performance claim. The installed-wheel gate preserves all 132 source/
data identities and 240 exact original source/native regenerations.
