# Typed condition comparison chains

Status: implemented and integration validated for the original-source contract below.

## Gap and selected solution

G06 excluded conditions such as `a < b < c`. Python evaluates the middle operand
once and evaluates later operands only after preceding comparisons succeed.
Duplicating its source can repeat atomic effects or observe a different memory
snapshot. The [initial rejection](../../compiler_exp/gap_closure/python_comparison_chains/baseline.json)
and [prototype](../../compiler_exp/gap_closure/python_comparison_chains/README.md)
remain retained, with their original finite hardware/CPU limits.

Production now evaluates and saves each nonliteral middle scalar, emits its
existing typed comparison and branches out on false before any later operand
evaluates. The last comparison reaches the requested true/false branch. Negation
and existing while/and/or conditions retain short-circuit polarity. No predicate
Boolean ISA operation, separate call ABI, backend template, resource limit or
vendor fallback is added. Ordinary single comparisons keep their emitted code.

```python
from openassembler.pyfront import Array, i32, kernel, tid_x

@kernel
def classify(a: Array(i32), out: Array(i32), n: i32):
    i = tid_x()
    if i < n:
        value = 11
        if -3 < a[i] < 7:
            value = 17
        out[i] = value
```

## Type, evaluation, effect and source contract

Every adjacent relation uses the existing fixed scalar rules: i32/u32/i64/u64/
f32 values, exact contextual literals and explicit conversions for mixed runtime
kinds. A pure numeric literal can adopt the next adjacent operand's kind, so
`0 < 1 < x` works for admitted wide/unsigned/floating `x`. A nonliteral expression
keeps its one evaluated scalar kind and snapshot. Inexact or out-of-range literals
reject; this introduces no general Python mixed-type promotion or unbounded ints.

The six numeric relations `<`, `<=`, `>`, `>=`, `==`, `!=` compose in source order.
FP32 ordered comparisons and inequality retain existing NaN/infinity/signed-zero
semantics, including negated chains. A false relation skips remaining operands;
therefore skipped global loads and atomics have no effects. A middle atomic is
executed once and its old value is retained for both adjacent relations. A later
load can observe the completed atomic update; it does not replace the saved old
value. Independent counter checks validate this distinction.

Chains work directly in if/while conditions, including nested helpers, early
returns and existing and/or/not composition. Helpers retain their pure scalar
rules, copied caller arguments, return typing and recursion/expansion limits.
Kernel atomics retain their existing primitive/memory contracts. Thread-dependent
chains still make controlled barriers reject. Subsequent
[Boolean values](gaps-boolean-values.md) admit chain results as locals with typed
integer conversion. Membership, identity, mixed Boolean/numeric comparisons,
Boolean ABI/type unions and general Python effects remain outside the admitted
contracts and reject explicitly.

Each original chain records path/function/line/column, ordered operator kinds,
a digest of its specialized AST and per-kernel counts in the versioned source
graph. Original file hashes and helper/static-value identities remain present.
Standalone kernels retain chain ownership even without a helper or static
constant. The validator checks location, operator, digest and positive count
shape before build/native ownership; manifests bind the complete graph. Changed
source/expressions invalidate old dependency snapshots and refreshed kernel
references rebuild them. Local, installed and remote builds share the frontend;
no user module/package top-level code executes during discovery or compilation.

## Experiments and validation

The [frozen baseline](../../compiler_exp/gap_closure/python_comparison_chains/promotion/baseline.json)
retains **60 original source/entry rejections**. The
[offline matrix](../../compiler_exp/gap_closure/python_comparison_chains/promotion/candidate-offline.json)
passes **4,320 complete executions / 240 rows** over same-file, imported and
remote-package sources, twenty kernels, four levels, two seeds, three lengths
and PTX/normal/poisoned SASS. Independent references cover relation/negation/
while/and/or truth, literal bridges, wide signed/unsigned boundaries, FP32 special
values, nested helpers/returns and old/new memory/atomic snapshots. Every output,
atomic counter, input, guard and inactive-tail byte is checked.

The [generated gate](../../compiler_exp/gap_closure/python_comparison_chains/promotion/control-properties.json)
adds **128 deterministic 2..5-link chains / 1,536 executions**, using all six
relations, literals/negation and independently counted atomic evaluations.
Sources, seed 86713, input bits, output/counter references remain retained. The
[16 negative controls](../../compiler_exp/gap_closure/python_comparison_chains/promotion/negative-controls.json)
keep mixed types, invalid literals/value forms, unsupported membership/identity,
initialization, helper effects and convergence failures explicit. Actual ptxas
accepts all three original trees / sixty kernels with 22 raw artifact identities.

The final [A100 gate](../../compiler_exp/gap_closure/python_comparison_chains/promotion/candidate-a100-20261124T113417Z.json)
passes **2,400 launches** on A100-SXM4-40GB, Python 3.12.1/CUDA 12.8.1: three source
trees, twenty kernels, emitted PTX controls plus four OA levels, two seeds,
lengths 0/1/259/384 and three 128-thread CTAs. Original source/PTX/graphs/manifests/
cubins/disassembly and complete before/after input/output/counter bytes are
archived. The [production audit](../../compiler_exp/gap_closure/python_comparison_chains/promotion/promotion-audit.json)
checks **15,482,880 complete independent bytes**, **240 exact native word/register/
frame outputs**, zero local frames, source/expression/helper/native ownership and
corrupted-output rejection. These are correctness gates with no speed claim.

All [159 new regressions](../../tests/test_pyfront_comparison_chains.py) pass before
promotion, including adjacent literal type contexts, standalone chain ownership,
source refresh, malformed expression metadata and manifest tampering. All 920
preceding frontend tests pass under the candidate; no existing test is changed.
Full suites, installed-wheel regeneration, public Modal publication and prior-code
continuity have separate integration gates in the
[experiment record](../../compiler_exp/gap_closure/python_comparison_chains/promotion/README.md).

The [integration record](../../compiler_exp/gap_closure/python_comparison_chains/promotion/validation.json)
passes **4,271 supported tests** and **3,786 compiler-only tests with fifteen skips**.
All **1,079 frontend tests** pass. The
[wheel audit](../../compiler_exp/gap_closure/python_comparison_chains/promotion/wheel-audit.json)
checks 132 installed identities and 240 native regenerations; the
[public Modal build](../../compiler_exp/gap_closure/python_comparison_chains/promotion/normal-remote-audit.json)
preserves original expression/source/native ownership. The
[frontend comparison](../../compiler_exp/gap_closure/python_comparison_chains/promotion/frontend-continuity.json)
checks nineteen source files, 62 identical accepted PTX kernels and original
rejections. The [application audit](../../compiler_exp/gap_closure/python_comparison_chains/promotion/application-continuity.json)
newly regenerates 240 preceding original range/else outputs and separately inherits
2,774 earlier direct/inherited occurrences after exact proof/archive and unchanged
backend-input checks. The total 3,014 explicitly separates those counts. All 49
measured policies and 98 original sessions retain their existing source binding;
no historical binary, policy, measurement or numerical report is rewritten.

The first numerically passed 2,400-launch payload omitted standalone-chain source
ownership. Its failed focused probe, exact candidate and raw payload remain
retained; the final graph repair reruns the complete source-exact hardware gate.
A negative fixture initially expected an operator diagnostic before its unsupported
tuple operand; it now asserts the precise tuple rejection. A report-only Path
error is retained separately. No numerical tolerance, resource limit, type rule,
old observation or existing rejection assertion is weakened.

The earlier [30-launch prototype](../../compiler_exp/gap_closure/python_comparison_chains/candidate-a100-20261124T111942Z.json)
and [11,080-byte / 24-native audit](../../compiler_exp/gap_closure/python_comparison_chains/candidate-audit.json)
remain finite first-stage evidence. Their 12 atomic and 60 typed emulator
executions are separate from the final original-source promotion gate.

```sh
python compiler_exp/gap_closure/python_comparison_chains/promotion/baseline.py
python compiler_exp/gap_closure/python_comparison_chains/promotion/offline.py
python compiler_exp/gap_closure/python_comparison_chains/promotion/negative_controls.py
python compiler_exp/gap_closure/python_comparison_chains/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_comparison_chains/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_comparison_chains.py
```

Production paths: [condition lowering](../../openassembler/pyfront/compile.py),
[source/expression records](../../openassembler/pyfront/modules.py) and
[regressions](../../tests/test_pyfront_comparison_chains.py). General Boolean values,
dynamic range steps, arbitrary Python effects and the full 51-gap program remain
incomplete.

The subsequent [Boolean-value cycle](gaps-boolean-values.md) implements
operand-returning and/or and explicit truth values with contextual literals,
source/native ownership and an independently validated predicate-path repair.
Boolean ABI and general mixed-type unions remain separate work.
