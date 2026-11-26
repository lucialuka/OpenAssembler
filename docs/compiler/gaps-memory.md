# Global/shared memory widths and predicate assignment

Status: implemented for the bounded contracts below

## Gap and numerical/effect contract

G12's earlier admission/selection mismatch blocked vector-two global loads and
vector-four stores despite existing recovered SM80 instructions. Scalar global
byte, halfword and 64-bit stores also lacked a maintained lowering. G11's
admitted `mov.pred` reached generic register MOV and rejected; predicate
reassignment across guards and branches additionally lacked fixed predicate
homes. Guarded vector loads did not count all destination definitions.

The repaired global-memory matrix is:

| Operation | Element types | Required address alignment | Native instruction |
|---|---|---:|---|
| Vector-two load | `b32/u32/s32/f32` | 8 bytes | `LDG.E.64` |
| Vector-four store | `b32/u32/s32/f32` | 16 bytes | `STG.E.128` |
| Scalar byte store | `b8/u8/s8` | 1 byte | `STG.E.U8` |
| Scalar halfword store | `b16/u16/s16` | 2 bytes | `STG.E.U16` |
| Scalar 64-bit store | `b64/u64/s64/f64` | 8 bytes | `STG.E.64` |

Stores copy exact low-order bytes; signedness does not change those bytes.
The 64-bit floating route copies bits without introducing FP64 arithmetic.
Register values and integer/IEEE binary64 immediates have explicit low/high-word
materialization; a binary32 float literal used as a 64-bit source rejects with
`NotSupported` rather than an internal exception.
Existing load cache modifiers (`nc/cg/cs/lu/ca`) and store cache modifiers
(`wb/cg/cs/wt`) retain the same value contract and may lose their performance
hint. Ordering/scope/volatile semantics are still rejected. Alignment, valid
addresses and sufficient allocation lengths remain caller invariants; this is
not a new runtime bounds, race or alignment checker. The subsequent shared-memory extension below adds scalar narrow/64-bit
accesses. Vectors of 64-bit elements remain unsupported.

Predicate MOV and logical/setp assignments preserve snapshots, false-guard old
values, branch joins, loop-carried values and self-guarded writes. Freshly
computed predicates write directly into a fixed home when needed. Vector load
destinations participate in the existing fixed-home analysis, so an inactive
load preserves every old vector element. The allocator still rejects excessive
predicate/register pressure; this work does not add spilling.
Packed and unpacked 64-bit MOV values also preserve snapshots when a scalar or
wide source is reassigned later; mutable home components are copied before
renaming or destination updates.

## Candidate, validation and promotion

The isolated [candidate generator](../../compiler_exp/gap_closure/memory_widths/candidate.py)
started from recorded baseline hashes and generated separate candidate modules.
Production code remained unchanged while the
[experiment](../../compiler_exp/gap_closure/memory_widths/offline.py) encoded and
executed all fixtures against independent byte-level expected buffers. The
[baseline](../../compiler_exp/gap_closure/memory_widths/baseline-fixture-v2-20261121.json)
rejects all 23 cases across four optimization levels. The
[initial candidate](../../compiler_exp/gap_closure/memory_widths/candidate-direct-predicate-20261121.json)
passes its 92 encoded cases under normal and poisoned-register execution.
Independent review then found two untested pack/unpack aliasing failures and
an uncaught binary64 immediate operand exception. The
[snapshot controls before repair](../../compiler_exp/gap_closure/memory_widths/snapshot-before-20261121.json)
retain eight wrong-result failures across four levels; the
[immediate controls before repair](../../compiler_exp/gap_closure/memory_widths/immediate-before-20261121.json)
retain eight operand exceptions. Separate
[snapshot](../../compiler_exp/gap_closure/memory_widths/candidate_snapshot.py) and
[immediate](../../compiler_exp/gap_closure/memory_widths/candidate_immediates.py)
candidates fix those concrete defects. The
[final candidate](../../compiler_exp/gap_closure/memory_widths/immediate-candidate-20261121.json)
passes all 112 encoded cases under normal and poisoned-register execution.

The [final A100 report](../../compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json)
records **224 launches**: 28 cases × four OA optimization levels × OA and
same-PTX ptxas controls. Python is 3.12.1, CUDA toolkit 12.8.1 and the actual
device is NVIDIA A100-SXM4-40GB. Every output byte, both outer guards, inactive
lanes, narrow-store neighbors and input immutability pass. Each kernel launches
96 threads for 67 valid lanes. Input words include signed/unsigned boundaries,
infinity/NaN bit patterns and fixed-seed random words. No floating arithmetic
tolerance is involved. The report's archive retains combined source, input and
expected buffers, complete outputs, manifests, cubins and SASS; the source and
binary hashes identify each implementation. There is no performance claim.

The earlier [copy-based predicate candidate](../../compiler_exp/gap_closure/memory_widths/a100-20261121T122940Z.json)
also passes. The selected candidate computes directly into predicate homes,
avoiding its extra predicate copies. Production
[selector](../../openassembler/backend/ra.py) and
[admission schema](../../openassembler/backend/contracts.py) were promoted only
after their exact hashes matched the final hardware report.
The [promotion audit](../../compiler_exp/gap_closure/memory_widths/final-promotion-audit-20261121.json)
checks all 458,752 retained output bytes, binary identities and the exact
promoted backend hashes. Earlier successful reports retain their narrower
fixture sets and earlier backend identities.

The first toolkit attempt rejected a fixture's incorrectly typed `f64` source
register; the corrected fixture declares a raw `b64` register. The rejected
[toolkit log](../../compiler_exp/gap_closure/memory_widths/a100-candidate-20261121.log)
and both original offline snapshots remain retained. They must not be counted
as successful toolkit validation. A separate
[harness correction](../../compiler_exp/gap_closure/memory_widths/harness-correction.json)
records an aborted manifest-indexing attempt before hardware results.

## Reproduction and remaining boundaries

```sh
.venv/bin/python compiler_exp/gap_closure/memory_widths/offline.py /tmp/oa-memory-validation.json
.venv/bin/python -m pytest -q tests/test_backend_memory_widths.py tests/test_backend_contracts.py tests/test_backend_loops.py tests/test_backend_ops.py tests/test_verify.py
modal run compiler_exp/gap_closure/memory_widths/run_modal.py
.venv/bin/python compiler_exp/gap_closure/memory_widths/ledger_snapshot.py
.venv/bin/python compiler_exp/gap_closure/memory_widths/validate.py compiler_exp/gap_closure/memory_widths/a100-20261121T124129Z.json /tmp/oa-memory-audit.json
```

The hardware command uses the retained exact candidate modules, matching the
promoted source hashes. The generator records baseline anchors and is intended
for the original baseline checkout; rerunning it on the promoted source rejects
anchor drift. The later snapshot/immediate candidate constructors read retained
prior candidate sources, keeping each repair's baseline available. The
[regression fixtures](../../tests/data/backend_memory_widths.json) are retained
independently of experiment imports. Offline regression compares every byte,
including false-guard and tail bytes. Negative tests keep unsupported widths,
memory ordering and malformed vectors rejected. The relevant suite passes
327 backend and Python frontend tests locally.

The [new scalar ledger](../../compiler_exp/gap_closure/memory_widths/ledger-memory-20261121-v3.json)
retains a new version and leaves the original 426-declaration snapshot intact.
All 496 current declarations select in its representative domains; 490
register-backed and 482 parameter-backed fixtures encode. Its independent
CPU/toolkit/packaging/hardware/performance stages remain explicitly `not-run`:
the 28 hardware fixtures above have separate exact source identities. This
repairs the named mismatches and expands G12's bounded support. Arbitrary shared operand/alignment domains, ordered memory, general G11
arithmetic and complete G02 semantic conformance remain open.

## Scalar shared byte, halfword and 64-bit memory

### Gap, solution and exact contract

The [baseline](../../compiler_exp/gap_closure/shared_widths/oracle-probe.json)
retains seven same-source toolkit-accepted shared exchanges that the selector
rejected. Existing load/64-bit-store templates covered several observed forms;
four narrow-store forms needed recovery. The maintained scalar matrix is now:

| Element types | Bytes/alignment | Load | Store |
|---|---:|---|---|
| `b8/u8` | 1 | `LDS.U8` zero extension | `STS.U8` low byte |
| `s8` | 1 | `LDS.S8` sign extension | `STS.S8` low byte |
| `b16/u16` | 2 | `LDS.U16` zero extension | `STS.U16` low halfword |
| `s16` | 2 | `LDS.S16` sign extension | `STS.S16` low halfword |
| `b64/u64/s64/f64` | 8 | `LDS.64` pair | `STS.64` pair |

The wide route preserves exact bits, including floating zeros, infinities and
NaN payloads; it adds no FP64 arithmetic. Guarded loads preserve false-path old
values and aliased wide destinations. Stores preserve every adjacent byte.
Literal offsets must be naturally aligned. Direct shared-symbol accesses must
fit their declared size and alignment. Dynamic bases must designate aligned,
allocated and initialized shared storage; this remains a caller invariant.
Vector narrow/wide-element forms, volatile/ordered accesses, arbitrary alias
and initialization/race proofs remain outside this contract. Existing
`v2.u32` uses the same 64-bit native store/load and is retained as a control.

### Encoding recovery and hardware scheduling

The [recovery](../../compiler_exp/gap_closure/shared_widths/template-recovery.json)
uses actual oracle words and the existing field learner with `canonical=False`.
The [field sweep](../../compiler_exp/gap_closure/shared_widths/field-sweep.json)
checks **12,096 native words** across source/base registers, signed 24-bit
literal offsets, address scales and guards. Deliberately misaligned offsets are
encoding controls only. The [custody audit](../../compiler_exp/gap_closure/shared_widths/template-audit.json)
binds the source, cubin, listing and log for all four recovered forms.
Production loads these exact rows through the ordinary encoder from
[`memory_templates.json`](../../specs/sm_80/memory_templates.json); existing
generated tables are not hand-edited.

The [first hardware attempt](../../compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T143055Z.json)
failed toolkit parsing of negative-offset PTX before execution. Retained
[syntax observations](../../compiler_exp/gap_closure/shared_widths/negative-syntax-oracle.json)
show the accepted `+-10` spelling. The next
[960-launch gate](../../compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T143647Z.json)
failed 90 OA launches: optimized scalar 64-bit exchanges loaded the earlier
initialization although a later complete shared snapshot was correct. Safe OA
and all same-PTX controls passed. The
[failed-output audit](../../compiler_exp/gap_closure/shared_widths/failed-execution-audit.json)
checks all 27,525,120 retained bytes without relabeling those failures.

The scalar latency model had shortened the barrier issue gap to one cycle.
The validated repair retains at least six cycles at CTA barriers in ordinary
code containing `LDS.64`/`STS.64`, including existing `v2.u32` exchanges. This
bounds the observed scheduling repair; it is not a general minimum-latency
proof or permission to relax another protocol. The
[repaired initial gate](../../compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T144011Z.json)
passes 960 launches. The final expanded gate adds false-load preservation,
aliased wide destinations, raw FP64 edges and the existing vector64 control.

### Validation, promotion and reproduction

The fresh [width matrix](../../compiler_exp/gap_closure/shared_widths/width-matrix.json)
retains 336 scalar/vector rows over fourteen types, global/shared spaces and
register/parameter-backed inputs. All 172 admitted representatives encode;
164 unsupported rows reject at admission. These are encoding observations,
with numerical/toolkit/hardware stages explicitly not run for those fixtures.
Their source identities differ from the execution corpus below. A separate
[read-only audit](../../compiler_exp/gap_closure/shared_widths/width-matrix-audit.json)
replays all 172 exact native encodings and checks every retained fixture,
rejection and stage count.

The [offline gate](../../compiler_exp/gap_closure/shared_widths/candidate-offline.json)
passes **1,368 executions** over nineteen fixtures, four compiler levels, two
seeds, three active lengths and PTX/normal/poisoned SASS engines. The
[final A100 report](../../compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T144327Z.json)
passes **1,216 launches**: nineteen fixtures × four levels × OA/same-PTX ptxas
× four active lengths × two seeds. Three 128-thread CTAs exchange across warps.
All threads reach each barrier, including empty and partial-tail domains.
Signed extension, low/high words, guarded loads/stores, negative offsets,
symbol bounds and every neighbor byte have independent expected buffers.

The [production audit](../../compiler_exp/gap_closure/shared_widths/promotion-audit.json)
checks **34,865,152 complete retained bytes**, input immutability/guards,
**76 native word/register comparisons**, 2,048 shared bytes per CTA, zero local
frames and manifest ownership. Actual device: A100-SXM4-40GB; worker Python
3.12.1, CUDA 12.8.1, ptxas V12.8.93. Local recovery uses ptxas V12.8.61.
Source and harness hashes bind the exact promoted implementation. There is no
performance claim. The [integration](../../compiler_exp/gap_closure/shared_widths/validation.json)
and [continuity dossier](gaps-continuity.md) retain full-suite and prior-native
checks separately from the new hardware execution.

```sh
python compiler_exp/gap_closure/shared_widths/audit_templates.py
python compiler_exp/gap_closure/shared_widths/offline.py
modal run compiler_exp/gap_closure/shared_widths/run_modal.py
python compiler_exp/gap_closure/shared_widths/audit.py compiler_exp/gap_closure/shared_widths/candidate-a100-20261123T144327Z.json /tmp/shared-audit.json
python -m pytest -q tests/test_backend_shared_widths.py tests/test_backend_memory_widths.py
```

Use the supported Python/Torch environment described in [DEVELOPMENT](../DEVELOPMENT.md).
The [experiment](../../compiler_exp/gap_closure/shared_widths/README.md) retains
all intermediate failures, candidate snapshots and raw artifacts. Production
and [regression fixtures](../../tests/data/shared_width_fixtures.json) import no
experiment modules. The original G12 and G09 exit gates remain partial.

The subsequent [typed Python shared-array extension](gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
adds i64/u64 with eight-byte storage/scaling and mixed-layout padding. Its own
failed hardware gate led to a separate native CTA reconvergence repair before
the final source-exact A100 admission. Broader memory-safety and control-flow
contracts remain open.
