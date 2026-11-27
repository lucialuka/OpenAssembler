# Boolean values and lazy scalar operand selection

Status: implemented and integration validated for the contract below.

## Gap and selected solution

G06 excluded general Boolean-valued expressions after admitting
[condition chains](gaps-comparison-chains.md). Python and/or return an operand;
forcing every output to zero/one changes numeric programs. The
[first-stage experiment](../../compiler_exp/gap_closure/python_boolean_values/README.md)
retains original rejections and finite CPU/toolkit evidence. Its leading-literal
rejection was a prototype limitation, not the intended production contract.

Production now supports Boolean locals/literals, not, bool(x), chain results,
typed integer conversions and operand-returning scalar and/or. Each nonliteral
operand compiles into an isolated fragment to learn its fixed device kind. Those
fragments emit only on the original live runtime paths. Numeric literals adopt
the common exact kind, so `0 or wide_value` works without duplicating expressions
or evaluating later effects early. The compiler never executes user source code.

```python
from openassembler.pyfront import Array, i32, kernel, tid_x

@kernel
def flags(a: Array(i32), out: Array(i32), n: i32):
    i = tid_x()
    if i < n:
        inside = 0 < a[i] < 7
        fallback = a[i] or 9
        out[i] = fallback + i32(inside)
```

## Type, value, effect and source contract

Nonliteral and/or operands must have one fixed device kind: i32/u32/i64/u64/f32,
or internal Boolean predicates. A selected numeric operand preserves its exact
bits. FP32 NaN remains truthy; signed zeros and NaN payloads survive selection.
False/True, not, bool(x) and comparison-chain results produce internal Boolean
values. Explicit i32/u32/i64/u64 conversion maps them to zero/one. Numeric and/or
results retain their selected operand and are not implicitly normalized to flags.

Numeric literals use the common nonliteral kind under existing exact/range rules;
pure literal-only numeric expressions retain i32/f32 defaults. Mixed runtime kinds,
Boolean/numeric unions and inexact/out-of-range literals reject. Boolean array/
parameter annotations and general implicit Boolean arithmetic remain outside this
extension. Plain int()/float() conversions retain their existing numeric contracts;
typed integer casts provide the Boolean conversion route. This is the declared
fixed device language, not arbitrary Python type unions or unbounded integers.

Later operands are skipped when the prior operand decides and/or. Every evaluated
expression runs once. Atomic old values, later reads and nested helper calls retain
source order and distinct snapshots. Helpers keep scalar annotations, copied caller
arguments, pure effects and recursion/expansion limits. Boolean locals may carry
state through existing loops. Initialization remains conservative; thread-dependent
Boolean values continue to make controlled barriers reject.

Each original value expression records path/function/line/column/category,
specialized AST digest, result kind and per-kernel expansion counts. Versioned
source graphs bind the complete original files/helper/static-value records and
native ownership, including standalone expressions without helper/static values.
Shape validation and changed-source/snapshot checks remain active through normal
local, remote and installed builds. No production import points into experiments.

## Actual native predicate failure and repair

The [first original-source A100 gate](../../compiler_exp/gap_closure/python_boolean_values/promotion/candidate-a100-20261124T115739Z.json)
runs all 2,400 cases but has **54 predicate_while mismatches** at mined/sched/opt.
Safe and ptxas pass. The independent loop reference requires two trips; failing
native code executes three. Functional emulators agree with the reference because
they do not model predicate readiness. All failed raw sources/outputs, original
inputs and expected bits remain retained; no tolerance or emulator is changed.

The linear native guard-readiness pass sees the definition before a guard in
instruction order. A loop's backward edge can carry a newer predicate definition
to that guard too soon. The repair keeps the existing linear pass and additionally
checks direct branch/fallthrough predecessor paths against the existing measured
**13-cycle predicate-to-guard requirement**. It only increases stall/yield bits.
Predicated writes preserve both write and skipped-write possibilities. Search is
bounded by the latency threshold and terminates across cycles. Instruction forms,
operands, order, register allocation, frames and resource limits remain unchanged.

The [first CFG candidate](../../compiler_exp/gap_closure/python_boolean_values/promotion/candidate-a100-20261124T120206Z.json)
missed quoted assembler labels and retained the same 54 failures. Its exact source
and logs remain retained. The corrected parser passes explicit backward-branch,
alternate-predecessor, predicated-definition and control-validity regressions.
The final [A100 rerun](../../compiler_exp/gap_closure/python_boolean_values/promotion/candidate-a100-20261124T120418Z.json)
passes **2,400 launches**. The
[failure-transition audit](../../compiler_exp/gap_closure/python_boolean_values/promotion/failure-transition-audit.json)
binds identical original source/PTX/inputs/counters/guards and independently checks
all repaired outputs. Exactly **12 of 240 native outputs** change stall/yield bits;
their instruction/operand/order/register identity stays exact. This repairs every
observed failure under unchanged numerical references. No new encoding template,
vendor fallback or speed claim is introduced.

## Experiments and validation

The [frozen baseline](../../compiler_exp/gap_closure/python_boolean_values/promotion/baseline.json)
retains sixty original source/entry rejections. The
[offline matrix](../../compiler_exp/gap_closure/python_boolean_values/promotion/candidate-offline.json)
passes **4,320 complete executions / 240 rows** across three original source trees,
twenty kernels, four levels, two seeds and three lengths with PTX/normal/poisoned
SASS. Independent raw-bit truth/operand/effect references compare every output,
input, atomic counter, guard and inactive tail. Sixty literal-context executions
cover all five scalar kinds and special values. The
[generated gate](../../compiler_exp/gap_closure/python_boolean_values/promotion/control-properties.json)
adds **128 original lazy value/effect trees / 1,536 executions**, with full sources,
seed 97531, input bits and independently interpreted atomic/read states retained.
Sixteen [negative controls](../../compiler_exp/gap_closure/python_boolean_values/promotion/negative-controls.json)
keep kind/literal/effect/initialization/convergence and ABI limits explicit.
Actual ptxas accepts sixty original kernels with 22 raw artifact identities.

The final A100 gate uses A100-SXM4-40GB, Python 3.12.1/CUDA 12.8.1, three source
trees/twenty kernels, emitted PTX plus four OA levels, two seeds, lengths
0/1/259/384 and three 128-thread CTAs. The
[production audit](../../compiler_exp/gap_closure/python_boolean_values/promotion/promotion-audit.json)
checks **16,957,440 complete independent bytes**, **240 exact current native
word/register/frame outputs**, zero local frames, original value/source/helper/
native ownership and corrupted-output rejection. All original sources/PTX/graphs/
manifests/cubins/disassembly and complete before/after input/output/counter bytes
remain archived.

The [159 frontend regressions](../../tests/test_pyfront_boolean_values.py) and
[eight native path regressions](../../tests/test_native_predicate_paths.py) pass
before promotion. All **1,246 focused tests** pass afterward. Three older negative
fixture rows retain genuine mixed-kind/ABI boundaries with positive independent
value/conversion/effect controls. Their
[exact contract updates](../../compiler_exp/gap_closure/python_boolean_values/promotion/test-contract-updates.json)
and first test outcome remain retained. The measured
[policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261124T121203Z.json)
preserves all 49 policy binaries/controls/resources and 98 original sessions before
source identity rebinding. Its first source whitelist rejection is retained;
only the exact independently A100-validated backend snapshot is newly admitted,
while every frozen executable comparison remains mandatory.

The [fresh native replay](../../compiler_exp/gap_closure/python_boolean_values/promotion/application-continuity.json)
passes **3,254 direct comparisons, all exact**, including every native word/control/
register identity. No prior timing field changes and no inherited-only shortcut
are used. The [integration record](../../compiler_exp/gap_closure/python_boolean_values/promotion/validation.json)
passes **4,438 supported tests** and **3,953 compiler-only tests with fifteen skips**.
The [wheel audit](../../compiler_exp/gap_closure/python_boolean_values/promotion/wheel-audit.json)
checks 132 installed identities and 240 source/native regenerations; the
[public Modal build](../../compiler_exp/gap_closure/python_boolean_values/promotion/normal-remote-audit.json)
preserves original value/source/native ownership. The
[frontend comparison](../../compiler_exp/gap_closure/python_boolean_values/promotion/frontend-continuity.json)
checks twenty source files, 62 identical accepted PTX kernels and unchanged
rejections. Exact final source/test/experiment/proof/log identities and live/exported
documentation checks remain bound. Boolean ABI/type unions and the full 51-gap
program remain incomplete.

```sh
python compiler_exp/gap_closure/python_boolean_values/promotion/baseline.py
python compiler_exp/gap_closure/python_boolean_values/promotion/offline.py
python compiler_exp/gap_closure/python_boolean_values/promotion/negative_controls.py
python compiler_exp/gap_closure/python_boolean_values/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_boolean_values/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_boolean_values.py tests/test_native_predicate_paths.py
```

Production paths: [value lowering](../../openassembler/pyfront/compile.py),
[expression/source records](../../openassembler/pyfront/modules.py),
[native predicate readiness](../../openassembler/backend/compile.py). General
Boolean ABI, mixed-type unions, dynamic range steps, arbitrary host/Python effects
and the complete 51-gap program remain incomplete.

The next [Boolean ABI study](gaps-boolean-abi.md) records original scalar/array/
helper rejections and actual CUDA byte parameter/storage forms. It has no
production candidate and keeps narrow-parameter parser admission explicit.
