# Scalar, aligned-group and predicate spilling

Status: implemented for bounded scalar, pair/quad and predicate spills

G17's original scalar allocator rejected register exhaustion and could not satisfy restrictive
`.maxnreg` directives. The [baseline pressure record](../../compiler_exp/gap_closure/spilling/baseline-pressure.json)
retains 40/96-value kernels exceeding their 32-register limits and a 260-value
kernel exhausting the existing allocator at all four optimization levels.
This research uses the independently validated [private local frames](gaps-storage.md)
to preserve scalar values instead of rejecting those workloads. A second
[baseline](../../compiler_exp/gap_closure/spilling_groups/baseline.json) retains
twenty rejected pair/quad, predicate and combined pressure configurations.

## Candidate and exact boundaries

The [initial scalar candidate](../../compiler_exp/gap_closure/spilling/spill.py)
first attempts ordinary allocation. On actual register exhaustion, it selects
long-lived scalar virtual values and assigns each a distinct four-byte private
slot. It reloads values before their uses and stores after definitions, then
reallocates. Aligned pairs/quads remain intact and are excluded from scalar
spilling. That validated scalar policy remains the first allocation plan.

The [extended candidate](../../compiler_exp/gap_closure/spilling_groups/spill.py)
retries an unsuccessful scalar plan with whole two/four-register groups. Every
member receives a separate word slot, and a wide consumer reloads an entire
fresh tuple; normal allocation preserves its consecutiveness and alignment.
Predicate exhaustion independently spills long-lived predicates. Existing
`SEL` encodes zero for true and one for false, and `LDL` plus `ISETP.EQ` restores
the predicate. Definitions, predicate operands and positive/negated guards all
receive the same replacement at each instruction. This composes existing
recovered instructions and does not add encoding overrides.

This path serves the ordinary scalar backend, including its wide memory/value
groups. The specialized tensor recipe keeps its separate allocation/resource
rules. Tensor-fragment spilling and async-pipeline pressure changes require
independent dependency and hardware validation.

A guarded definition reloads the previous slot value unconditionally, computes
the update under the original predicate, then stores the resulting scalar.
This preserves the old value when the predicate is false and provides an
unconditional temporary definition for conservative CFG liveness. An initial
design guarded the temporary definition/store separately; conservative
backedge liveness then kept its scratch values live around the loop and failed
to reduce pressure. The [failure log](../../compiler_exp/gap_closure/spilling/candidate-offline-first.log)
and [first implementation](../../compiler_exp/gap_closure/spilling/first-spill.py)
remain retained. That run was stopped during the largest case after its earlier
failures; it is not counted as a completed passing gate.

The allocator admits at most 256 total word slots and a total local frame of 65,536
bytes, including declared user storage. Allocator slots follow the aligned
user frame and cannot overlap local objects. Entry stack adjustment and both
per-function metadata records use the final total reservation. The ordinary
builder receives explicit generated resource metadata and retains spill-slot,
frame and register-limit information in the manifest. Existing native frame
ownership checks bind the deployed reservation. Production
[allocation](../../openassembler/backend/spill.py) and
[builder resource propagation](../../openassembler/build.py) were promoted from
the exact tested source bytes; they import no experiment module.

Explicit `.maxnreg` limits below 16 reject. The candidate honors limits through
its existing SM80 register-allocation/metadata convention; without a directive,
it retains the existing 250-physical-register allocation domain and only spills
after that domain exhausts. No-spill kernels retain their original native code.
This adds neither a tuning policy nor a claim that fewer registers run faster.
Slot reuse, general pressure-aware scheduling and occupancy/performance
selection remain original G17 work. Spilling also rejects an instruction whose
simultaneous operands cannot fit the physical register/predicate domain, even
after every eligible original value has a slot. It never splits a physical
tuple or silently exceeds the slot/frame bounds.

## Independent validation

The [offline reference](../../compiler_exp/gap_closure/spilling/candidate-offline.json)
checks 40/96/260-value guarded loop trajectories against independent integer
arithmetic and normal/poisoned SASS, at all four compiler levels. The
[control gate](../../compiler_exp/gap_closure/spilling/candidate-controls.json)
checks exact no-spill native continuity, preserved existing rejections, a tight
16-register scalar limit and failure on unsupported group pressure. It retains
resource boundaries without treating an unsupported scalar form as a success.

The [expanded A100 run](../../compiler_exp/gap_closure/spilling/candidate-a100-20261123T095515Z.json)
passes **320 launches**, plus
[64 tight-limit launches](../../compiler_exp/gap_closure/spilling/tight-a100-20261123T095932Z.json).
Sources cover explicit 32/64-register limits, automatic spilling of the
previously exhausted kernel, a 16-register limit and coexistence with declared
local storage. Each domain uses all four OA levels and same-PTX ptxas controls,
three 128-thread CTAs, active lengths 0/1/259/384 and two changed input seeds.
Complete output/input/guard/tail checks pass. Both workers identify
A100-SXM4-40GB, Python 3.12.1 and CUDA toolkit 12.8.1. Driver resources match
16/32/64/252 native registers and 80/112/144/304/832-byte local frames. The
automatic case preserves the kernel that previously exhausted allocation;
it uses 18 spill slots. The explicit 260-value/64-register case uses 206 slots.

The [full audit](../../compiler_exp/gap_closure/spilling/promotion-audit.json)
and [tight audit](../../compiler_exp/gap_closure/spilling/promotion-tight-audit.json)
recheck **73,334,784 complete retained bytes**, native/source/manifest/driver
identities, independent integer references, deliberate corruption rejection and
24 current native word/register/resource comparisons. These audits ran against
both the isolated candidate and promoted production. The
[15 regressions](../../tests/test_backend_spilling.py) retain complete loop,
predicate, private-state, local-object coexistence and ordinary builder checks,
plus explicit too-low-limit rejection. The original group-pressure rejection
is superseded by the validated extension below; a simultaneous-operand budget
failure remains an explicit negative regression.

## Aligned groups and predicate extension

The [offline gate](../../compiler_exp/gap_closure/spilling_groups/candidate-offline.json)
passes all twenty configurations with full normal/poisoned SASS outputs,
independent integer references and guards. Sources retain 48-word pair/quad
groups at a 16-register limit, 260 words with automatic pressure, 24 live
predicates, and combined group/predicate pressure beside a declared local
object. Three loop iterations exercise guarded, self-guarded and negated
predicate definitions/uses. Alternate lanes preserve vector outputs, and
inactive tail lanes retain their sentinels. The
[controls](../../compiler_exp/gap_closure/spilling_groups/candidate-controls.json)
check exact prior scalar executables/resources, aligned reloaded tuples,
predicate budgets and impossible operand/frame failures. An initial control
harness stopped on an already unsupported `fma.rm.f32`; its
[log](../../compiler_exp/gap_closure/spilling_groups/controls-first.log) remains
retained. The final harness compares existing rejection diagnostics too.

The [A100 gate](../../compiler_exp/gap_closure/spilling_groups/candidate-a100-20261123T102437Z.json)
passes **320 launches** on A100-SXM4-40GB, driver 13030, Python 3.12.1 and CUDA
toolkit 12.8.1. It uses all four OA levels, same-PTX ptxas controls, three
128-thread CTAs, active lengths 0/1/259/384 and two input seeds. The
[read-only audit](../../compiler_exp/gap_closure/spilling_groups/promotion-audit.json)
checks **83,787,776 retained bytes**, all source/binary/ownership identities,
driver frame/register resources and twenty current native word comparisons.
OA uses 16/16/250/16/31 registers and 352/272/608/80/320 local bytes for the
five fixtures. Ptxas uses 24 registers for the two sources requesting 16;
that actual donor result remains recorded. No equal-resource performance
comparison is inferred. The [regressions](../../tests/test_backend_spill_groups.py)
use a separate predicate truth-table reference and complete guarded outputs.

Current [policy continuity](gaps-continuity.md) and
[application continuity](../../compiler_exp/gap_closure/spilling_groups/application-continuity.json)
retain all 49 measured policies and 1,366 earlier native kernel occurrences,
including the original scalar-spill corpus. Historical numerical reports stay
unchanged; changed allocator identity is bound to this separate hardware gate.

The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T100450Z.json)
preserves all 49 measured policies across 98 original native sessions. The
[application audit](../../compiler_exp/gap_closure/spilling/application-continuity.json)
checks 1,342 earlier local/shuffle/atomic/decode/checkpoint kernel occurrences
against current generation, without rewriting their independent numerical
proofs. The new allocator participates in current source identities and remains
absent from the frozen Git inventory. Full integrated suites and the resource
comparison are recorded beside the experiment; no optimal register-limit
policy is inferred from successful spilling.

The [paired resource experiment](../../compiler_exp/gap_closure/spilling/resources-20261123T100759Z.json)
compares the same 260-value workload, inputs, three-CTA launch and exact output
under an explicit 64-register limit and the ordinary automatic domain. Each
variant has three warmups and thirteen resident CUDA-event samples in randomized
order on one A100-SXM4-40GB worker. Every timed output passes complete byte and
guard checks. The [resource audit](../../compiler_exp/gap_closure/spilling/resource-audit.json)
checks all 52 recorded outputs, exact executable/source identities, sample
inventory/order and median recomputation.

| Executable | Driver registers | Local bytes/thread | Median kernel interval (μs) |
|---|---:|---:|---:|
| OA, explicit limit 64 | 64 | 832 | 111.36 |
| OA, automatic domain | 252 | 80 | 35.87 |
| Same-PTX ptxas, explicit limit 64 | 64 | 848 | 43.33 |
| Same-PTX ptxas, automatic domain | 255 | 88 | 31.65 |

The automatic OA case still spills eighteen values. Fewer registers permit a
higher driver maximum block size here (1024 versus 256 threads), but the actual
measured launch stays 128 threads; achieved residency is not inferred. Additional
spill traffic is substantially slower in this constructed kernel. This is a
resource/cost observation, not a selected register policy, held-out workload
distribution, host-inclusive measurement or general OA speed claim.

The [scalar-stage validation](../../compiler_exp/gap_closure/spilling/validation.json)
passes **2,883 supported tests**, plus **2,398 compiler-only tests with 15 explicit
skips**. Live/exported documentation checks and all source/evidence hashes are
bound separately. Broad G17 remains partial under its original exit gate.

The [current integration](../../compiler_exp/gap_closure/spilling_groups/validation.json)
passes **2,898 supported tests** and **2,413 compiler-only tests with 15 explicit
skips**, binding the extended allocator, continuity and documentation checks.
No new timing policy is selected by the group/predicate extension.

## Pressure-aware virtual scheduling

The next [baseline](../../compiler_exp/gap_closure/pressure_schedule/baseline.json)
retains valid kernels with avoidable 18/74/77/85/19/29-word spill plans. The
allocator sees long overlapping ranges when loads and predicate definitions
precede all their independent consumers. The staged
[virtual pass](../../compiler_exp/gap_closure/pressure_schedule/pressure.py)
orders ready instructions by estimated live scalar/group/predicate cost before
physical allocation. It retains register/predicate RAW, WAR and WAW edges,
guards, source load order, and fixed stores/synchronization/control boundaries.
Starting incomplete pairs/quads receives an extra cost so many address copies
cannot displace their wide consumers. No arithmetic expression is reassociated.
The first passing but weaker heuristic remains in the experiment directory.

The proposed allocation replaces a spilled baseline only with strictly fewer
slots, no larger frame, and no more local loads/stores in any corresponding
basic block. Equal-resource proposals retain the exact earlier allocation.
An earlier resource rejection may be recovered by a successful scheduled plan.
Successful no-spill kernels retain their original path. The final admission
recognizes guarded exits and every other control-flow terminator;
[the retained repair](../../compiler_exp/gap_closure/pressure_schedule/terminator-repair.json)
and a negative regression prevent transfer counts being aggregated across a
guarded early exit. This is a resource heuristic, not a general speed predictor.

The [offline gate](../../compiler_exp/gap_closure/pressure_schedule/candidate-offline.json)
passes 24 complete normal/poisoned configurations against independent integer
references and guards. Six new kernels use zero spill slots, including v2/v4
memory groups, predicate fan-out and a three-iteration local-object loop.
The [order/resource controls](../../compiler_exp/gap_closure/pressure_schedule/final-controls.json)
independently check pairwise dependencies, fixed boundaries, source load order,
unchanged operation identities, deliberately corrupt orders and actual
allocations for seventeen sources. Eleven older spill kernels retain exact
code/resources. These controls do not replace hardware scheduling validation.

The [main A100 report](../../compiler_exp/gap_closure/pressure_schedule/candidate-a100-20261123T105809Z.json)
passes **1,632 correctness launches**, comparing frozen OA, candidate OA and
same-PTX ptxas over seventeen sources, four compiler levels, active lengths
0/1/259/384, two seeds and three 128-thread CTAs. It then records 21 randomized
resident CUDA-event samples for each of 51 opt variants, with three checked
warmups each: **1,071 timed and 153 warmup launches**. Every launch retains full
input/output/guard/tail bytes by content hash, and the audit checks each byte
again rather than treating matching hashes as a numerical reference.

| Source | Frozen OA median (μs) | Candidate OA median (μs) | Same-PTX ptxas median (μs) |
|---|---:|---:|---:|
| Scalar40 | 20.320 | 15.776 | 14.656 |
| Scalar96 | 39.840 | 25.696 | 24.512 |
| Pair96 | 33.760 | 20.448 | 18.848 |
| Quad96 | 30.912 | 17.792 | 16.160 |
| Pred24 | 14.944 | 14.528 | 13.984 |
| Local-loop48 | 41.856 | 23.872 | 18.976 |

The [frozen-pass controls](../../compiler_exp/gap_closure/pressure_schedule/heldout-freeze.json)
vary count/register limits and add a finite FP32 FMA chain. Their
[A100 report](../../compiler_exp/gap_closure/pressure_schedule/heldout-a100-20261123T110440Z.json)
passes **384 correctness, 252 timed and 36 warmup launches**. Baseline/candidate
median intervals are 30.752/21.312 μs for scalar64, 41.792/21.088 μs for quad160,
19.040/15.968 μs for pred40 and 29.248/25.824 μs for the FP32 chain. That chain
uses exactly representable integer/half-integer trajectories with an
integer-derived reference; it does not prove arbitrary IEEE behavior. These
are four parameter/arithmetic controls, not a held-out application distribution.
Their original decimal-literal parser failure remains retained separately.

The exact final allocator has a separate
[128-launch A100 gate](../../compiler_exp/gap_closure/pressure_schedule/final-a100-20261123T112918Z.json)
for guarded early returns and a 65,536-byte declared local frame. The latter
previously rejected because added spill slots exceeded the bound; the candidate
uses zero slots and reserves the original maximum frame. All three workers
identify A100-SXM4-40GB and driver 13030, Python 3.12.1 and CUDA toolkit 12.8.1.
Prototype source bytes remain retained, and final production auditors require
every measured native executable to match final generation exactly.

The stable [main production audit](../../compiler_exp/gap_closure/pressure_schedule/promotion-audit.json),
[separate-control audit](../../compiler_exp/gap_closure/pressure_schedule/promotion-heldout-audit.json)
and [final-source audit](../../compiler_exp/gap_closure/pressure_schedule/promotion-final-audit.json)
recheck **627,347,456 retained bytes** and **176 native comparisons**, along with
source/ownership/frame/register custody, every sample order and every median.
The initial candidate auditor file was edited while its rehearsal process ran,
so its return-time source hash cannot identify the executing auditor. That
[custody limitation](../../compiler_exp/gap_closure/pressure_schedule/candidate-audit-custody.json)
excludes the rehearsal from promotion authority. The stable production audits
repeat all byte/native/resource/timing checks independently; original hardware
reports and archives remain unchanged.

The [host code-generation experiment](../../compiler_exp/gap_closure/pressure_schedule/compile-cost.json)
shows a separate cost: baseline/candidate median 2.574/2.661 seconds for scalar96,
0.238/0.254 seconds for pred24 and 9.761/25.510 seconds for legacy group4_auto,
whose native code remains unchanged. Three randomized paired samples use a warm
encoder and parsed PTX; parsing, toolkit/packaging, GPU work and complete host
contention are outside the measurement. The candidate evaluates two allocation
plans in spilled kernels. The kernel interval gains above are not a measured
whole-application or compile-amortized speedup, and ptxas remains faster on all
six new nominal kernel comparisons. Register counts and driver maximum block
sizes are recorded; achieved occupancy is not inferred from those attributes.

Canonical [experiment custody](../../compiler_exp/gap_closure/pressure_schedule/README.md)
links the prototype, final admission, regressions and integration results.
The [current integration](../../compiler_exp/gap_closure/pressure_schedule/validation.json)
passes **2,927 supported tests** and **2,442 compiler-only tests with 15 explicit
skips**. Source identities, complete production audits and live/exported docs
checks are bound in that record. Prior application continuity retains 1,386
native occurrences and all 49 measured policies remain byte-identical.
Slot reuse, tensor-fragment spilling, cross-region scheduling and a general
occupancy/performance policy remain original G17 work.

```sh
python compiler_exp/gap_closure/spilling/pressure_baseline.py
python compiler_exp/gap_closure/spilling/offline.py
python compiler_exp/gap_closure/spilling/control_gate.py
modal run compiler_exp/gap_closure/spilling/run_modal.py
modal run compiler_exp/gap_closure/spilling/run_tight_modal.py
python compiler_exp/gap_closure/spilling_groups/offline.py --candidate
python compiler_exp/gap_closure/spilling_groups/controls.py
modal run compiler_exp/gap_closure/spilling_groups/run_modal.py
```

The drivers stage the candidate explicitly; their results do not establish
general G17 completion, new architectures or a whole-application speedup.
