# Original Boolean values and lazy operand selection

Status: implemented and integration validated for the bounded original-source contract.
The [canonical dossier](../../../../docs/compiler/gaps-boolean-values.md) defines
source/type/effect/ownership and native predicate-readiness contracts.

## Gap and solution

The [first-stage experiment](../README.md) admitted homogeneous scalar and/or and
predicate values but rejected leading numeric literal contexts. The final frontend
compiles each nonliteral operand into an isolated code fragment to learn its fixed
kind without executing user code. It then emits those fragments only on their
original live runtime paths. Numeric literals adopt that common exact scalar kind,
including `0 or wide_value`; mixed runtime kinds and Boolean/numeric unions reject.
Pure literal-only numeric expressions follow existing i32/f32 defaults.

Python and/or return a selected operand, preserving exact numeric bits. Boolean
locals, True/False, not, bool(x), chained-comparison results and typed integer
casts keep separate truth semantics. Internal predicates convert to i32/u32/i64/
u64 as zero/one. Boolean array/parameter ABI, arbitrary mixed-kind unions and
unrestricted Python numeric conversions remain separate work.

Original value expressions record path/function/line/column/category, specialized
AST digest, fixed result kind and per-kernel expansion counts. Full source/helper/
static-value/native ownership and normal local/remote snapshots remain active,
including standalone expressions without helpers/constants. Production imports
no experiment files and does not execute user modules/packages.

## Actual native failure and validated repair

The [first A100 payload](candidate-a100-20261124T115739Z.json) runs all 2,400 cases
but has **54 actual predicate_while mismatches** at mined/sched/opt. Safe and ptxas
pass. Output is three loop trips where the independent reference requires two;
PTX/normal/poisoned emulators agree with the reference because they model values,
not native predicate timing. The source, expected bits, inputs and all failures
remain retained. No reference, tolerance or emulator is changed.

The native linear predicate-readiness pass checks the definition before a guard
in instruction order. A backward branch can bring a newer predicate definition
to that guard before it becomes ready. The repaired scheduler retains the existing
linear pass, then walks real direct-branch/fallthrough predecessor paths to enforce
the existing measured **13-cycle predicate-to-guard requirement**. It only increases
stall/yield bits; it does not alter instructions, register allocation or ordering.
Predicated definitions keep both write and skipped-write paths. A bounded backward
search stops once the required latency is met and terminates across cycles.

The [first CFG repair](candidate-a100-20261124T120206Z.json) retained the same 54
mismatches because its experiment omitted the assembler's quoted label syntax.
`first-cfg-backend.py` and `first-cfg-label-failed.log` retain that error. The final
parser handles quoted direct labels and passes explicit backward/alternate/
predicated-definition path regressions before the complete GPU rerun.

The final [A100 gate](candidate-a100-20261124T120418Z.json) passes **2,400 launches**.
The [failure-transition audit](failure-transition-audit.json) checks identical
original source/PTX/input/counter/guard identities, all repaired outputs, and all
240 native outputs. Only **12 outputs** change stall/yield bits; their instruction/
operand/order/register identity remains exact. This repairs all 54 observed bad
launches while preserving the independent references. It adds no new ISA template,
resource limit, fallback or performance claim.

## Independent integration gates

- [Frozen baseline](baseline.json): sixty original same-file/import/package value
  rejections, replayed with the pre-promotion frontend/source modules.
- [Offline matrix](candidate-offline.json): **4,320 complete executions / 240 rows**,
  three original trees/twenty kernels/four levels/two seeds/three lengths through
  PTX/normal/poisoned SASS. All output/input/counter/guard/tail bits match independent
  scalar truth, raw operand and stateful atomic references.
- [Literal contexts](context-probe.json): sixty exact leading-literal executions
  across all five scalar kinds, including FP32 special values and original ownership.
- [Generated values](control-properties.json): **128 numeric/Boolean lazy value
  trees / 1,536 executions**, independent recursive operand/effect evaluation and
  per-thread atomic/read states, seed 97531, retained sources/inputs/references.
- [Negative controls](negative-controls.json): sixteen kind/literal/ABI/array/
  arity/keyword/helper-memory/initialization/convergence/arithmetic/cast/shadow limits.
- [Toolkit](oracle-probe.json): actual ptxas accepts three original trees/sixty
  kernels, with 22 source/cubin/listing/log identities.
- [Production audit](promotion-audit.json): **16,957,440 independent complete bytes /
  240 exact native word/register/frame outputs**, zero local frames, original
  Boolean/source/helper/native ownership and corrupted-output rejection.
- **159 new frontend tests plus eight native predicate-path tests** pass before
  promotion; all **1,246 focused tests** pass afterward. One existing fixture has
  [three exact contract updates](test-contract-updates.json): mixed Boolean/numeric
  comparisons and unconverted predicate stores keep their genuine rejection;
  newly admitted and/or is replaced by an unsupported runtime-kind union control.
  Positive independent numerical/effect/conversion tests cover the new behavior.
- The [measured-policy audit](../../foundations_20261121/audit-20261124T121203Z.json)
  keeps all 49 measured policy binaries/controls/resources and 98 original sessions
  exact before compiler identity rebinding. `first-policy-scope-failed.log` retains
  the initial source whitelist rejection; the observer now admits only the exact
  A100-validated backend snapshot and still requires every frozen executable check.

The [fresh complete native replay](application-continuity.json) passes **3,254
direct comparisons, all exact**, including every word/control/register identity.
There are zero changed prior stall/yield outputs and no inherited-only shortcut.
The full supported Python 3.12.10/Torch 2.7.1 suite passes **4,438 tests**. The
separate Python 3.14.5/no-Torch suite passes **3,953 tests with fifteen skips**.
The [wheel audit](wheel-audit.json) checks 132 installed identities and 240 original
source/native regenerations. The [public Modal build](normal-remote-audit.json)
preserves original value/source/native ownership. Twenty source files retain
62 identical accepted kernels and unchanged rejections. The
[integration record](validation.json) binds exact production/test/experiment/
proof/log identities and live/exported documentation. Boolean ABI/type unions and
the complete original 51-gap program remain incomplete. No speed claim is made.
