# Range counter overflow research

Status: finite i32 ranges and runtime loop else implemented and integration validated.
The first sections retain the bounded pre-promotion study and its original limits.

## Gap and bounded reproduction

The supplied G06 review explicitly identifies fixed-width loop-counter wrap.
Production accepts i32 range bounds and a nonzero constant i32 step. Near the
i32 limits, a nonunit step can wrap before the exhaustion comparison observes
that the mathematical next value has crossed stop. The original Python range is
finite, but OA executes extra iterations. The
[bounded baseline](../../compiler_exp/gap_closure/python_range_control/baseline.json)
retains four counterexamples and two ordinary/empty capability controls across
four PTX/normal/poisoned levels: **72 observations**.

An explicit eight-trip watchdog prevents an unbounded test. Independent Python
range values determine the complete checksum, trip count and last-target bits.
The boundary cases cover positive/negative overflow, INT_MAX positive step and
INT_MIN negative step. For example, `range(2147483646, 2147483647, 2)` must visit
one value; the retained OA baseline reaches its watchdog after eight trips.
PTX and encoded emulators agree on the wrong source lowering, so their agreement
alone is insufficient validation.

## Isolated solution and hardware evidence

The [candidate](../../compiler_exp/gap_closure/python_range_control/candidate.py)
checks whether the next mathematical counter would leave i32 before applying a
nonunit step. With admitted i32 bounds, that next value has necessarily crossed
stop, so branching to exhaustion avoids the wrap while preserving the last
selected target. Unit-step lowering remains unchanged. This is an isolated
hypothesis, with no production imports or raised bounds. All six candidate cases
match independent bytes in **72 offline observations**.

The [A100 record](../../compiler_exp/gap_closure/python_range_control/candidate-a100-20261124T060602Z.json)
retains **60 bounded original-source observations**: frozen baseline/candidate,
six ranges, emitted PTX controls and four OA levels. Its
[audit](../../compiler_exp/gap_closure/python_range_control/candidate-audit.json)
separates **30 numerically correct candidate launches**, **16 required baseline OA
mismatches**, **eight correct baseline OA controls**, and **six observational
baseline ptxas results**. It independently reconstructs **9,120 complete output/
guard bytes** and checks **48 exact native word/register outputs** plus corrupted
output rejection. No timing or production integration claim is made.

The [first GPU attempt](../../compiler_exp/gap_closure/python_range_control/candidate-a100-20261124T060330Z.json)
failed because its baseline control assumed ptxas would reproduce the OA wrap.
Actual ptxas produced the Python-reference result for the first boundary case.
The retained final observations show that all six baseline ptxas controls match
the Python reference, while baseline OA still reproduces all four counterexamples
at every level. The cause of that toolkit/native/emulator divergence remains
unresolved. Baseline ptxas is consequently recorded as six explicit observations;
the candidate numerical gate still requires all thirty outputs to match every
reference byte. The first failed harness and full archive remain retained. No
numerical tolerance, candidate expectation or old observation is rewritten.

## Remaining promotion gates

Original helper/caller and nested control, runtime bounds and snapshotted values,
positive/negative/nonunit boundary generation, break/continue and exhaustion else,
source/native ownership, regression and whole-application continuity still need
separate validation. The toolkit divergence needs a minimized instruction-level
investigation before making a general PTX semantic claim. Runtime range/while
else and dynamic steps remain separate open G06 work. The complete 51-gap program
remains incomplete.

```sh
python compiler_exp/gap_closure/python_range_control/replay_baseline.py
python compiler_exp/gap_closure/python_range_control/candidate.py
python compiler_exp/gap_closure/python_range_control/probe.py --candidate
modal run compiler_exp/gap_closure/python_range_control/run_modal.py
python compiler_exp/gap_closure/python_range_control/audit.py compiler_exp/gap_closure/python_range_control/candidate-a100-20261124T060602Z.json
```

## Original-source production extension

The [promotion cycle](../../compiler_exp/gap_closure/python_range_control/promotion/README.md)
now guards a nonunit next counter before wrap and preserves range/while exhaustion
else with separate break labels. `continue` reaches the next range increment or
while test; break skips else. Else uses the enclosing loop context, so a break or
continue inside an inner else correctly targets an outer loop. Helper and kernel
early returns retain their respective exit scopes. Unit-step loops retain their
old emission path. Original range bounds remain evaluated once and snapshotted;
mutation of a visible target or bound variable does not change the iteration
sequence. A final target mutation survives break/exhaustion as in Python.

Start/stop are still i32 expressions and the step a compile-time nonzero i32
numeric integer. The loop counter never applies an increment that would leave
i32. Because stop is i32, a mathematical next value outside that interval has
necessarily crossed stop, so the range exhausts. This fixes finite range
iteration without changing scalar arithmetic's wrapping semantics. Runtime or
Boolean steps, wide bounds, arbitrary iterables and general control graphs remain
outside the existing admitted source contract. Loop termination is not inferred
for arbitrary while bodies. There is no new ISA form, backend template, native
call ABI, resource limit or vendor fallback.

The helper initialization analyzer keeps general range/while zero-trip paths;
body-only assignments remain uninitialized after the loop. Else assignments reach
later code only on paths that exhaust, while break paths bypass them. Returning
paths leave subsequent initialization joins. Thread-dependent bounds or loop
exits make barriers in exhaustion else reject before execution. Uniform else
barriers retain their existing admission contract. Original entry/helper/package
hashes, body signatures, call identities, static values and native ownership
continue through local/remote source snapshots. No user module executes during
compilation.

The [frozen classifications](../../compiler_exp/gap_closure/python_range_control/promotion/baseline.json)
retain 36 original loop-else rejections and 24 admitted but deliberately unexecuted
potentially overflowing original kernels. Actual baseline wrap evidence comes
from the separate bounded watchdog study above. The final
[offline gate](../../compiler_exp/gap_closure/python_range_control/promotion/candidate-offline.json)
passes **4,320 complete executions / 240 rows** over three original trees, twenty
kernels, four levels, two seeds, three lengths and PTX/normal/poisoned SASS.
Independent full-bit references cover boundary steps, runtime bounds, empty loops,
mutation/snapshots, break/continue, nearest-loop else exits and early returns.
The [generated gate](../../compiler_exp/gap_closure/python_range_control/promotion/control-properties.json)
adds **128 deterministic boundary/control combinations / 1,536 whole-byte
executions**, with exact original sources and Python range sequences retained.
Sixteen [negative controls](../../compiler_exp/gap_closure/python_range_control/promotion/negative-controls.json)
keep range-domain, path/initialization and convergence failures explicit.

The final [A100 gate](../../compiler_exp/gap_closure/python_range_control/promotion/candidate-a100-20261124T111006Z.json)
passes **2,400 launches** on A100-SXM4-40GB, Python 3.12.1/CUDA 12.8.1, with original
emitted PTX controls and four OA levels, two seeds, lengths 0/1/259/384 and three
128-thread CTAs. The [production audit](../../compiler_exp/gap_closure/python_range_control/promotion/promotion-audit.json)
checks **17,203,200 complete independent bytes / 240 exact native word/register/
frame outputs**, zero local frames, original source/helper/native ownership and
corrupted-output rejection. Actual ptxas accepts sixty kernels in three original
trees, with 22 retained raw artifact identities. This is correctness evidence;
no timing or universal external optimizer interpretation follows.

The minimized [O0/O3 A100 study](../../compiler_exp/gap_closure/python_range_control/minimized-a100-20261124T061630Z.json)
and [independent audit](../../compiler_exp/gap_closure/python_range_control/minimized-audit.json)
now isolate the earlier disagreement: O0 and runtime-bound loops retain explicit
i32-wrap outcomes; O3 folds the four static-bound overflow loops to mathematical
range results with either signed or unsigned increments. Isolated signed/unsigned
adds wrap at both levels. All **40 observations / 6,080 complete bytes** and raw
sources/cubins/disassembly remain retained. This establishes a bounded external
loop-optimization distinction, without claiming its general legality or internal
implementation. The retained [NVIDIA add specification](../../compiler_exp/gap_closure/python_range_control/manual/source.json)
provides the instruction context. Production avoids the counter overflow entirely,
and all final original-source candidate/reference launches must match independently.

The [150 new regressions](../../tests/test_pyfront_range_control.py) pass along with
all **920 frontend tests**. Five old loop-else rejection assertions are replaced
by genuine missing-path/value/initialization or unsupported dynamic-step failures,
with positive original numerical controls. Their
[exact updates](../../compiler_exp/gap_closure/python_range_control/promotion/test-contract-updates.json)
and first 765-pass/five-stale-assertion run remain retained. A fixture initially
read an uninitialized runtime-loop target, then initialized it explicitly; another
used a Boolean step outside the admitted contract and now uses numeric one.
Neither failure weakened production flow/type admission.

The [continuity audit](../../compiler_exp/gap_closure/python_range_control/promotion/application-continuity.json)
newly regenerates 240 preceding helper-return source/native outputs exactly;
2,534 earlier direct/inherited occurrences retain proof/archive hashes and
unchanged backend inputs. The total 2,774 explicitly separates those counts.
The [frontend comparison](../../compiler_exp/gap_closure/python_range_control/promotion/frontend-continuity.json)
checks eighteen source files / 62 kernels; one existing semantic fixture changes
PTX for nonunit guards and keeps its independent Python/PTX/SASS numerical gate.
The [integration record](../../compiler_exp/gap_closure/python_range_control/promotion/validation.json)
passes **4,112 supported tests** and **3,627 compiler-only tests with fifteen skips**.
The [wheel audit](../../compiler_exp/gap_closure/python_range_control/promotion/wheel-audit.json)
checks 132 exact installed identities and 240 native regenerations. The
[public Modal build](../../compiler_exp/gap_closure/python_range_control/promotion/normal-remote-audit.json)
retains original graph/native ownership. All 49 measured policies and 98 original
sessions retain their exact compiler inputs without a policy update. Final
source/test/experiment/research/proof/log identities and live/exported docs checks
remain bound in the record. Broad G06 and the full 51-gap program remain incomplete.

The subsequent [condition-chain cycle](gaps-comparison-chains.md) now preserves
middle-operand snapshots and short-circuit effects under its separate typed
source/expression/native ownership gates.

The post-promotion [frozen watchdog replay](../../compiler_exp/gap_closure/python_range_control/baseline-replay.json)
reproduces all 72 exact original observations using the frozen frontend; it does
not overwrite historical results or describe current production as still failing.
