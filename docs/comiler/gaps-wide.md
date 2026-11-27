# Wide scalar arithmetic, shifts and comparisons

Status: implemented

## Gap and proposed lowering

G11 included six admitted forms that selected but did not encode:
`sub.s64/u64`, `shl.b64`, and `shr.b64/s64/u64`. Subtraction required an
unlearned complemented-register template; wide literal shifts required
unrecovered immediate SHF forms. Wider typed Python also exposed conversion
values that aliased mutable source registers instead of preserving snapshots.

The repair uses existing recovered instructions exclusively. Subtraction
explicitly inverts the high word with LOP3 before ordinary IADD3.X. Wide shifts
materialize their count for register SHF forms. Arithmetic right shift across
word 32 combines logical low-word shifting with arithmetic high-word shifting
and selects the appropriate low word. No encoder fields or tables changed.

The maintained scalar contract now includes:

- Signed and unsigned 64-bit subtraction modulo `2^64`, including borrows,
  aliased destinations and predicated old values. Existing addition is retained.
- `shl.b32/b64` and `shr.b32/u32/s32/b64/u64/s64` with literal or register counts.
  Counts are unsigned 32-bit values. Counts at least the width produce zero,
  except signed right shifts, which sign-fill. Negative literals still reject.
- `setp.eq/ne/lt/le/gt/ge.s64/u64` and unsigned `lo/ls/hi/hs.u64`. Ordered
  comparisons use signed or unsigned high words and an unsigned low-word tie
  break. Predicate homes preserve guarded assignments and source snapshots.
- Existing integer conversions across signed/unsigned 32/64-bit types now
  preserve values when a source register changes after a conversion.

These are typed device integer semantics, not unbounded Python arithmetic.
FP64 arithmetic, wide/high integer products, 64-bit divide/remainder, saturation/carry chains and
arbitrary PTX modifier combinations remain separate work.

## Experiments and independent validation

The [isolated candidate](../../compiler_exp/gap_closure/scalar_wide/candidate.py)
and its generated modules were validated before production promotion. The
[reference corpus](../../compiler_exp/gap_closure/scalar_wide/cases.py) contains
signed/unsigned boundaries, low-word borrow cases, fixed-seed random words,
counts 0 through 65 and large counts through `0xffffffff`. Every source
updates an aliased destination under a predicate, retains old values on false
lanes and leaves inactive tail lanes untouched. Conversion controls mutate
their original source after taking a snapshot.

The [baseline](../../compiler_exp/gap_closure/scalar_wide/baseline-v2-20261121.json)
retains 76 passing cases, 16 wrong-result conversion cases and 416 rejections
across four optimization levels. The first candidate's eight missing NE
predicate-template rejections remain in its earlier record; the final
implementation composes existing AND comparisons and PLOP instead. The
[final candidate](../../compiler_exp/gap_closure/scalar_wide/candidate-v2-20261121.json)
passes all 508 encoded cases under both ordinary and poisoned-register
execution. All 127 fixtures independently match the
[PTX interpreter](../../compiler_exp/gap_closure/scalar_wide/ptx-reference-20261121.json).

The [A100 report](../../compiler_exp/gap_closure/scalar_wide/a100-20261121T131238Z.json)
passes **1,016 launches**: 127 fixtures × four OA optimization levels × OA and
same-PTX ptxas controls. It identifies Python 3.12.1, CUDA toolkit 12.8.1 and
NVIDIA A100-SXM4-40GB. Each launch uses 576 threads for 527 valid lanes. Every
output byte, guard, preserved lane and input immutability check passes.
Independent references use integer bit arithmetic without SASS emulation or
floating tolerance. The archive retains exact PTX, complete input/expected/output
bytes, manifests, cubins and SASS. These are correctness results, with no timing
or generalized architecture claim.

The [promotion audit](../../compiler_exp/gap_closure/scalar_wide/promotion-audit-20261121.json)
checks all 9,883,648 retained output bytes and exact promoted backend identities.
The toolkit logs retain a cosmetic disassembly spelling difference (`0x0f`
versus `0xf`) for LOP3's inversion LUT; the words and all outputs pass. Production
imports no experiment modules.

## Reproduction and remaining coverage

```sh
.venv/bin/python compiler_exp/gap_closure/scalar_wide/offline.py /tmp/oa-wide.json
.venv/bin/python compiler_exp/gap_closure/scalar_wide/ptx_gate.py
.venv/bin/python -m pytest -q tests/test_backend_wide.py
modal run compiler_exp/gap_closure/scalar_wide/run_modal.py
.venv/bin/python compiler_exp/gap_closure/scalar_wide/validate.py compiler_exp/gap_closure/scalar_wide/a100-20261121T131238Z.json /tmp/oa-wide-audit.json
.venv/bin/python compiler_exp/gap_closure/scalar_wide/ledger_snapshot.py
```

The hardware command stages the retained candidate bytes matching promoted
source hashes. The source generator reads retained baseline modules. The
[regression fixtures](../../tests/data/scalar_wide_fixtures.json) remain independent
of experiment imports; tests compute expected values directly from input bits.
Additional regression controls encode and execute every literal shift from 0
through 63, and keep negative counts and unsupported FP64/saturating forms
rejected.

The [new ledger](../../compiler_exp/gap_closure/scalar_wide/ledger-wide-20261121.json)
keeps all older snapshots intact. All 512 declarations now encode in its
register-backed representative domain. Parameter-backed fixtures encode 504;
eight existing constant-bank selection/comparison restrictions remain. Each
generated ledger fixture still records execution/toolkit/hardware/performance
as `not-run`: the hardware corpus above has separate exact source identities.
Closing the six recorded failures does not establish all operand combinations,
complete ISA conformance, general wide arithmetic or a new GPU target.

## Typed verification through the Python build route

Wide Python entries publish `openassembler-pyfront-types-v1` JSON comments with
schema version, exact entry name and ordered parameter names, scalar/array
roles and canonical PTX element types. The
[generic verifier](../../openassembler/verify.py) validates that declaration
against the parsed parameter ABI before allocating data. It distinguishes a
`u64` scalar from a pointer, allocates `f32/s32/u32/s64/u64` arrays at their real
element width and preserves complete 64-bit scalar bits in both execution
engines. Untagged PTX retains the existing pointer heuristic. Invalid, duplicate,
unknown-entry or mismatched metadata cannot silently select that legacy path.

The typed route runs ordinary and poisoned-register SASS against the PTX
interpreter and compares every element, including upper words. Both engines'
adjacent 64-byte guards are checked against their original values, so identical
out-of-bounds writes in both engines still fail. Arrays use deterministic edge
values plus seeded data; common dimension names remain bounded, while other
integer scalar probes include full-width boundary bits. This does not infer
arbitrary shape, stride or runtime scalar validity constraints, and it is not
general memory-safety proof.

The [typed-verifier experiment](../../compiler_exp/gap_closure/python_wide/verification/typed-verifier-20261121.json)
retains the old checker rejecting an explicit contract and treating an unsigned
64-bit data scalar as a pointer. The new route passes 20 type/seed cases and
rejects deliberate upper-word corruption and matching guard writes. The
[regressions](../../tests/test_verify_typed.py) additionally reject malformed
metadata; the typed and existing generic-verifier tests pass 33 cases. These
are CPU checker results; the Python frontend's hardware gates are separate.

```sh
.venv/bin/python compiler_exp/gap_closure/python_wide/verification/experiment.py
.venv/bin/python -m pytest -q tests/test_verify_typed.py tests/test_verify.py
```

## Decimal and raw floating literal contexts

The pressure-scheduling FP32 control exposed another G11 operand boundary.
The [baseline](../../compiler_exp/gap_closure/float_literals/baseline.json)
retains ptxas accepting decimal/exponent literals that OA parsed as variable
names. The [PTX floating-constant specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-constants)
defines binary64 decimal representation followed by use-type conversion.
The independent [rounding counterexample](../../compiler_exp/gap_closure/float_literals/rounding-probe.json)
shows why that matters: direct rational-to-f32 rounding of
`1.0000000596046448` yields `0x3f800001`; binary64-then-f32 and ptxas yield
`0x3f800000`.

Production [parser normalization](../../openassembler/ptx/parse.py) now admits
signed decimal/exponent forms, including `.5`, `2.`, `1E+3` and signed zero.
It stores their binary64 bits first, then converts for a scalar f32 use.
Binary64 decimal underflow/subnormals and overflow reject to match the retained
CUDA 12.8 oracle; f32 overflow/subnormals produced from normal binary64 literals
remain supported. Raw64-to-f32 conversion preserves sign/high NaN payload and
quiets converted NaNs. Raw32 literals retain their exact bits. A raw32 literal
in `mov.f64` or `st.f64` zero-extends those bits, as separately observed in the
toolkit and A100; it is not a numerical f32-to-f64 conversion. Existing f64
literal moves/stores preserve all 64 bits. No f64 arithmetic is added.

Literal type/width checks reject floating constants in signed/unsigned integer
contexts, wrong-width bit contexts, destinations, predicate selectors and
packed moves. Integer literals in `mov.f32/f64` also reject. Vector lane,
address, modifier and broader operand contracts retain their separate checks.
Normalization changes no encoding fields or arithmetic instructions, and the
production parser imports no experiment code.

Six toolkit matrices retain **110 context/spelling controls**, accepted and
rejected, with complete tool diagnostics and binary/disassembly identities.
The [oracle custody audit](../../compiler_exp/gap_closure/float_literals/oracle-evidence-audit.json)
checks 368 retained artifacts and current parser acceptance. The local oracle
identifies ptxas V12.8.61; the independent GPU worker uses CUDA 12.8.1.
The [independent offline gate](../../compiler_exp/gap_closure/float_literals/candidate-offline.json)
checks 5,016 exact-rational decimal parses/use conversions, 5,008 raw64
conversions/payloads, seven raw32 preservation/extension controls and all retained
context matrices. These compare a rational integer IEEE reference with the
parser's host binary64 conversion and use normalization; the reference is not
the production implementation.

The [execution gate](../../compiler_exp/gap_closure/float_literals/execution-offline.json)
passes **360 complete PTX/normal/poisoned-SASS executions** across five source
families and all four compiler levels. It checks independent output bits,
immutable inputs, guards and inactive tails for decimal/raw32/raw64 literal
moves, direct stores, exact FP32 arithmetic/comparison/selection positions and
predicated definitions/loops. The first execution record mistakenly reported
54 rather than 18 executions per configuration; its source/report/log remain
retained, and the final record uses the actual 360-execution inventory.

The [A100 gate](../../compiler_exp/gap_closure/float_literals/candidate-a100-20261123T121453Z.json)
passes **320 launches** on A100-SXM4-40GB, driver 13050, Python 3.12.1 and CUDA
12.8.1 toolkit. Five fixtures use all four OA levels and same-PTX ptxas controls,
three 128-thread CTAs, active lengths 0/1/259/384 and two changed input seeds.
Complete byte checks retain signed zero, f32 overflow/subnormals, double-rounding
ties, raw NaN signs/payloads and preserved upper words, plus guarded loop state.
The arithmetic corpus uses exactly representable power-of-two/half-integer
trajectories and does not claim arbitrary IEEE operation conformance.

The [production audit](../../compiler_exp/gap_closure/float_literals/promotion-audit.json)
rechecks **28,737,536 retained bytes**, every source/binary/ownership identity,
20 current native word/register comparisons, immutable inputs, output guards,
all tail slots and explicit corruption rejection. All OA frames are zero;
actual driver registers are retained independently from ptxas resources.
The [regressions](../../tests/test_ptx_float_literals.py) exercise golden literal
bits, invalid contexts, complete PTX/SASS outputs and ordinary build/ownership.
The 655 focused parser/arithmetic/wide checks pass before promotion.
Initial full suites each retained one obsolete rejection for
`st.global.f64` with a raw32 literal. The [failure record](../../compiler_exp/gap_closure/float_literals/initial-suite-failure.json)
keeps both complete logs and the old test source. That case is now a positive
hardware/CPU fixture; the width-rejection test covers unsupported vector64,
and a separate raw32/b64 mismatch still rejects. The updated literal/memory
regressions pass 184 checks before the final full-suite rerun.

The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T121857Z.json)
preserves all 49 measured policies across 98 retained sessions. The
[application continuity audit](../../compiler_exp/gap_closure/float_literals/application-continuity.json)
compares 1,478 earlier native occurrences, including the full pressure-scheduling
corpus. Prior numerical and timing records remain unchanged. Broad G11 remains
partial: floating expressions, broader types/modifiers and full operand/encoder
conformance require their original independent gates. No speed claim is made.

The [next expression baseline](../../compiler_exp/gap_closure/float_literals/expression-baseline.json)
retains fourteen toolkit/OA comparisons, including seven accepted ptxas sources
that OA still rejects. Examples include parenthesized decimal arithmetic and
binary64 raw constants inside arithmetic. Mixed integer/float expressions,
raw32 arithmetic, division by zero and the tested conditional/comparison
contexts reject in this toolkit. These are pending-expression observations,
not an implementation or a GPU numerical pass.
The [expression rounding probe](../../compiler_exp/gap_closure/float_literals/expression-semantics.json)
passes fourteen normal/poisoned CPU executions against independent rational
references. It exposes binary64 intermediate cancellation before final f32
conversion. Expression implementation still requires its separate validation.

The canonical [experiment custody](../../compiler_exp/gap_closure/float_literals/README.md)
and [integration record](../../compiler_exp/gap_closure/float_literals/validation.json)
bind the staged parser, raw failures, final sources, suites and documentation.
Final suites pass **2,974 supported tests** and **2,489 compiler-only tests with
15 explicit skips**. The single-literal repair is bounded progress within G11;
it does not close the original broad numerical and operand-conformance gate.

## Typed scalar operand constant expressions

The next [baseline](../../compiler_exp/gap_closure/constant_expressions/baseline.json)
retains six valid source families rejected by the preceding parser. The
[PTX constant-expression rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#constant-expressions)
define typed s64/u64/f64 evaluation, precedence and per-operation floating
rounding. The production [parser](../../openassembler/ptx/parse.py) now tokenizes
and evaluates a bounded scalar operand grammar without host-language evaluation.
Integer expressions support parentheses, unary signs/logical/complement,
arithmetic, shifts, comparisons, bitwise/logical operations, signed/unsigned
casts and same-type integer conditionals. Arithmetic wraps to 64 bits, signed
division truncates toward zero, and constant shifts mask counts to six bits.
Floating arithmetic rounds each operation in binary64 before existing use
conversion. No arithmetic expression is reassociated or fused.

Integer expression results in floating data operands reject rather than becoming
raw float bits implicitly. Raw32 literals preserve their established standalone
path but cannot enter expressions. General address expressions, declaration/
initializer expressions, symbol relocation, functions and full PTX grammar
remain separate gates. More than 4,096 tokens or 128 nested operands reject
with explicit diagnostics.

The [depth counterexample](../../compiler_exp/gap_closure/constant_expressions/depth-baseline.json)
found a long conditional chain reaching host RecursionError after an earlier
passing candidate bounded only parentheses/unary nesting. The final parser
also counts recursive expression levels; its separate source-exact A100 gate
and final suites cover the revised bytes. Earlier passing records remain
retained as pre-repair evidence, rather than being rebound implicitly.

Four [oracle corpora](../../compiler_exp/gap_closure/constant_expressions/oracle-evidence-audit.json)
retain **191 controls** and **680 source/binary/listing/diagnostic artifacts**.
Local ptxas V12.8.61 observations and CPU execution expose folded constant bits.
The independent [typed-tree/rational gate](../../compiler_exp/gap_closure/constant_expressions/candidate-offline.json)
compares **4,000 generated expressions** against integer semantics and rational
binary64 rounding independent of the production parser. The complete
[execution gate](../../compiler_exp/gap_closure/constant_expressions/execution-offline.json)
passes **432 PTX/normal/poisoned-SASS executions** across six fixtures and four
compiler levels. Full outputs, inputs, inactive tails, upper words and guards
cover integer/floating constants, f32 narrowing, packed source components,
predicate guards, loop bounds and parameter/address arithmetic using constants.

The [first A100 report](../../compiler_exp/gap_closure/constant_expressions/candidate-a100-20261123T130725Z.json)
remains failed. Its [failure audit](../../compiler_exp/gap_closure/constant_expressions/first-hardware-failure-audit.json)
rechecks 40,304,640 complete bytes with frozen source identities: all OA launches
match the independent reference, while 48 ptxas control launches differ. Local
V12.8.61 and worker V12.8.93 disagree about remainder-result signedness and the
sign of a NaN generated by invalid infinity arithmetic. The public type summary
also differs from observed mixed signed/unsigned conditional behavior.

Those are explicit unresolved contracts: mixed signed/unsigned conditional
branches, remainder by zero, and invalid NaN-producing arithmetic reject.
A nonzero-divisor remainder may be used directly, or explicitly cast to s64/u64
before further arithmetic; the cast removes the ambiguous intermediate type.
Floating conditional branches reject under this toolkit. NaN constants and
observed operand-payload propagation have their own passing controls. These
diagnostics preserve the discrepancies for research instead of treating the
first failed report as a semantic pass.

The [final A100 report](../../compiler_exp/gap_closure/constant_expressions/candidate-a100-20261123T131948Z.json)
passes **384 launches** on A100-SXM4-40GB, driver 13030, Python 3.12.1 and ptxas
V12.8.93. Six sources use all four OA levels and same-PTX controls, three
128-thread CTAs, lengths 0/1/259/384 and two seeds. It keeps full source/binary/
manifest/disassembly/input/output artifacts. The
[production audit](../../compiler_exp/gap_closure/constant_expressions/promotion-audit.json)
rechecks **40,304,640 retained bytes** and **24 current native comparisons**,
ownership, frame/register resources and deliberate output corruption. No
numerical tolerance or original failed gate is changed. The
[regressions](../../tests/test_ptx_constant_expressions.py) retain golden bits,
invalid types, unresolved diagnostics, complete outputs and ordinary build/
ownership; the focused expression/literal/backend selection passes 204 tests.

The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T135952Z.json)
retains all 49 measured policies across 98 sessions. The
[application continuity audit](../../compiler_exp/gap_closure/constant_expressions/application-continuity.json)
compares 1,498 earlier native occurrences, including all literal fixtures.
The [experiment](../../compiler_exp/gap_closure/constant_expressions/README.md)
and [integration](../../compiler_exp/gap_closure/constant_expressions/validation.json)
bind final sources, suites, failures and documentation. Broad G11 remains
partial under the original numerical/operand/encoder gate; no speed claim is made.
The final integration passes **3,042 supported tests** and **2,557 compiler-only
tests with 15 explicit skips**, binding the hardened parser source and current
documentation separately from the retained pre-depth suites.

## Scalar 64-bit conditional selection

The actual [CUDA noinline source](../../compiler_exp/gap_closure/wide_select/noinline.cu)
exposed a concrete G11 rejection: nvcc emits `selp.b64` inside a device helper.
Static function expansion worked, but scalar admission rejected the selection.
The [isolated candidate](../../compiler_exp/gap_closure/wide_select/candidate.py)
was validated before promoting its exact selector and admission sources.

Production now accepts **`selp.b64/u64/s64/f64`**. Two existing recovered scalar
SEL instructions select low and high words using the same predicate. Fresh
result registers preserve source snapshots before binding an aliased destination;
the existing guarded-home path preserves old values when an outer guard is false.
Register operands, constant-bank parameters, both immediate positions, negated
predicates and both aliased destinations are covered. FP64 selection copies all
bits, including NaN payloads and signed zero; it adds no FP64 arithmetic or new
instruction encoding field. Vector selection, 16-bit selection and unvalidated
modifiers remain explicit rejections.

The [PTX reference observation](../../compiler_exp/gap_closure/wide_select/ptx-reference.json)
records NVIDIA's documented type contract. The first sixteen-case experiment
incorrectly used integer registers for FP64 data. Actual toolkit rejection and
all [first sources](../../compiler_exp/gap_closure/wide_select/first-oracle/)
remain retained. A [typed investigation](../../compiler_exp/gap_closure/wide_select/typed-investigation/)
accepts all four corrected FP64 sources. The expanded
[toolkit audit](../../compiler_exp/gap_closure/wide_select/oracle-audit.json)
checks **36 accepted typed sources and 144 raw source/log/cubin/listing identities**.
This corrects the fixtures; it does not reinterpret their original rejections.

The [complete offline gate](../../compiler_exp/gap_closure/wide_select/candidate-offline.json)
passes **2,592 executions**: 36 sources × four levels × two seeds × three lengths
× PTX/normal/poisoned SASS. Independent references compare complete guarded
outputs, inactive tails and unchanged inputs using integer bit operations.
The [baseline](../../compiler_exp/gap_closure/wide_select/baseline.json) retains
all 36 production rejections before promotion.

The [A100 gate](../../compiler_exp/gap_closure/wide_select/candidate-a100-20261123T205126Z.json)
passes **1,480 launches** on A100-SXM4-40GB with Python 3.12.1 and CUDA 12.8.1.
It covers the 36 legal PTX sources plus the original CUDA noinline workload,
original-PTX ptxas controls and all four OA levels, three 128-thread CTAs,
lengths 0/1/259/384 and two seeds. Its archive retains original CUDA/PTX, emitted
PTX, manifests, cubins, disassembly, full original/returned inputs and outputs.
The [production audit](../../compiler_exp/gap_closure/wide_select/promotion-audit.json)
checks **15,155,200 complete bytes**, **148 exact current native word/register
comparisons**, zero local frames, ownership contracts and an intentionally
corrupted-output rejection control. No tolerance is used or relaxed.

The [production regressions](../../tests/test_backend_wide_select.py) pass
**87 focused tests**, including original CUDA compilation, complete bit snapshots
and explicit type/modifier/operand rejection. The
[first focused test log](../../compiler_exp/gap_closure/wide_select/first-targeted-diagnostic-failed.log)
retains an expected-diagnostic mismatch: the parser correctly rejects a 32-bit
raw floating literal before backend admission. Its dedicated parser assertion
now checks that exact diagnostic; production needed no change.
The [experiment record](../../compiler_exp/gap_closure/wide_select/README.md)
keeps full integration and continuity evidence separate from new GPU execution.
Full suites pass **3,361 supported tests** and **2,876 compiler-only tests with
fifteen skips**. The [isolated wheel audit](../../compiler_exp/gap_closure/wide_select/wheel-audit.json)
checks 129 source/data identities and all 148 native outputs. The
[policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T205620Z.json)
preserves all 49 measured policies across 98 sessions. The
[prior-native replay](../../compiler_exp/gap_closure/wide_select/application-continuity.json)
checks all 1,786 earlier kernel occurrences. The
[integration record](../../compiler_exp/gap_closure/wide_select/validation.json)
binds final sources, suites, raw failures, current native code and live/exported
documentation. G11 remains partial,
and no speed claim is made.

```sh
python compiler_exp/gap_closure/wide_select/probe.py --candidate
python compiler_exp/gap_closure/wide_select/oracle_probe.py
python compiler_exp/gap_closure/wide_select/audit_oracle.py
modal run compiler_exp/gap_closure/wide_select/run_modal.py
python compiler_exp/gap_closure/wide_select/audit.py compiler_exp/gap_closure/wide_select/candidate-a100-20261123T205126Z.json /tmp/wide-selection-audit.json
python -m pytest -q tests/test_backend_wide_select.py
```

## Low 64-bit integer multiplication

The [original unsigned CUDA workload](../../compiler_exp/gap_closure/wide_multiply/promotion/cuda.cu)
exposed a concrete G11 blocker: nvcc emits `mul.lo.s64`, which production rejected.
The [first backend experiment](../../compiler_exp/gap_closure/wide_multiply/README.md)
validated the arithmetic, then the [promotion cycle](../../compiler_exp/gap_closure/wide_multiply/promotion/README.md)
found and addressed the separate typed Python frontend rejection. Backend and
frontend candidate snapshots were validated together before exact promotion.

Production now supports **`mul.lo.u64/s64`** and Python **`i64/u64` `*` and `*=`**.
Results keep the low 64 bits, with signed values interpreted in two's complement.
Four existing recovered IMAD/IMAD.HI instructions compose the product: low32 of
the low-word product, its unsigned high32 carry, and the two low32 cross terms.
Fresh low/high results preserve aliases, later source mutations, and outer or
negated guards. Register operands, scalar parameters, both immediate positions,
Python arrays/literals/augmented assignment/pure helpers/loops/casts and register/
parameter squares are covered. Mixed typed Python values still require explicit
casts; unbounded arithmetic, high/wide products, FP64 arithmetic and division/
remainder remain separate unsupported contracts. No new encoding field or table
change is introduced.

The [baseline](../../compiler_exp/gap_closure/wide_multiply/promotion/baseline.json)
retains 35 PTX/CUDA/Python source rejections, and the
[square baseline](../../compiler_exp/gap_closure/wide_multiply/promotion/baseline-squares.json)
keeps four shared-operand rejections. The independent
[base offline gate](../../compiler_exp/gap_closure/wide_multiply/promotion/candidate-offline.json)
and [square gate](../../compiler_exp/gap_closure/wide_multiply/promotion/candidate-squares-offline.json)
pass **2,808 complete executions** through PTX, ordinary SASS and poisoned-register
SASS. References use arbitrary-precision integer multiplication modulo `2^64`,
including overflow/cross-word carry edges, complete guards, inactive tails and
input immutability. Frozen baseline replay remains reproducible after promotion.

Actual toolkit controls accept all 39 legal original sources. The
[base toolkit audit](../../compiler_exp/gap_closure/wide_multiply/promotion/oracle-audit.json)
retains 157 raw artifacts, and the
[square toolkit audit](../../compiler_exp/gap_closure/wide_multiply/promotion/oracle-squares-audit.json)
retains twenty more. Python PTX is regenerated exactly; CUDA uses the actual
nvcc frontend. Observed `IMAD.SHL` aliases stay in raw disassembly and do not
change the exact word/byte gate.

The [base A100 report](../../compiler_exp/gap_closure/wide_multiply/promotion/candidate-a100-20261123T212929Z.json)
passes 1,400 launches, and the
[square A100 report](../../compiler_exp/gap_closure/wide_multiply/promotion/candidate-squares-a100-20261123T213402Z.json)
passes 160 more. Both use A100-SXM4-40GB, Python 3.12.1 and CUDA 12.8.1, original
or same-emitted-PTX ptxas controls and all four OA levels, three 128-thread CTAs,
lengths 0/1/259/384 and two seeds. Reports retain exact original/emitted sources,
manifests, cubins, disassembly and all before/after input/output bytes.
The [combined production audit](../../compiler_exp/gap_closure/wide_multiply/promotion/promotion-audit.json)
keeps the two disjoint gates explicit: **1,560 launches, 15,974,400 independent
bytes, 156 native word/register comparisons**, zero local frames, source custody,
frontend regeneration, ownership and corrupted-output rejection. No tolerance
or timing is pooled.

The [production regressions](../../tests/test_backend_wide_multiply.py) use an
independent standalone fixture corpus and keep invalid operand/type/modifier
and Python mixed-width/float/division/power rejection controls. Together with
[existing typed wide tests](../../tests/test_pyfront_wide.py), **140 focused tests
pass**. Their stale `x*x` rejection becomes a genuine unsupported-power assertion;
accepted squares have independent full-bit and A100 controls. The
[frontend continuity audit](../../compiler_exp/gap_closure/wide_multiply/promotion/frontend-continuity.json)
checks fifteen prior files, 62 byte-identical PTX kernels and unchanged rejection
diagnostics. The exact validated candidate is production code; broader G03/G11
coverage remains incomplete, and no speed claim is made. Full suites pass
**3,465 supported tests** and **2,980 compiler-only tests with fifteen skips**.
The [installed wheel](../../compiler_exp/gap_closure/wide_multiply/promotion/wheel-audit.json)
checks 129 source/data identities, 156 native outputs and eighty exact Python
PTX regenerations. The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T213840Z.json)
retains all 49 policies and 98 original sessions. The
[prior-native replay](../../compiler_exp/gap_closure/wide_multiply/promotion/application-continuity.json)
checks **1,934 earlier kernel occurrences** with exact current words/registers.
The [integration record](../../compiler_exp/gap_closure/wide_multiply/promotion/validation.json)
binds final sources, suites, failures, full numerical/ownership archives and
live/exported documentation.

```sh
python compiler_exp/gap_closure/wide_multiply/promotion/offline.py --candidate
python compiler_exp/gap_closure/wide_multiply/promotion/offline_squares.py --candidate
python compiler_exp/gap_closure/wide_multiply/promotion/oracle_probe.py
python compiler_exp/gap_closure/wide_multiply/promotion/audit_oracle.py
modal run compiler_exp/gap_closure/wide_multiply/promotion/run_modal.py
modal run compiler_exp/gap_closure/wide_multiply/promotion/run_squares_modal.py
python -m pytest -q tests/test_backend_wide_multiply.py tests/test_pyfront_wide.py
```
