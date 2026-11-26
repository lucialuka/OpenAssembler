# Logical Boolean helper parameters and returns

Status: implemented and integration validated for pure statically inlined helpers.

## Gap and contract

The [Boolean-value frontend](gaps-boolean-values.md) already computes logical
predicates, but an original `@device def invert(value: bool) -> bool` rejects at
annotation resolution. This G03/G04 extension carries those logical values
through pure helper signatures. It does not require a physical device-call or
host launch ABI: the existing static expansion gives each call its own registers,
argument copies, locals and return join.

An unshadowed built-in `bool`, including a quoted annotation, denotes a predicate
only in scalar helper parameter/return positions. Arguments evaluate once, left
to right, before the body and must have the exact declared kind. Numeric truth
conversion is explicit with `bool(x)`; integer zero/one does not implicitly become
a Boolean argument or return. Existing every-path return typing, initialization,
purity, recursion/expansion limits, caller isolation and defining-module name
resolution remain required. Predicate copies use existing `mov.pred` lowering;
native operation coverage, resource bounds and scheduling are unchanged.

Kernel Boolean parameters, Array/Buffer/shared Boolean storage, arbitrary unions,
Python objects and a dynamic device-call ABI remain separate work. Recursive
Array/Buffer element resolution does not inherit the helper predicate opt-in.
The [physical Boolean ABI study](gaps-boolean-abi.md) preserves the independently
observed byte-parameter/storage boundary and its original rejected sources.

## Solution and original source ownership

[Annotation resolution](../../openassembler/pyfront/compile.py) opts into logical
predicates only when [helper signatures](../../openassembler/pyfront/helpers.py)
request it. The existing argument/return machinery then copies and checks the
exact predicate kind without executing user modules, packages or functions.

The [source graph](../../openassembler/pyfront/modules.py) records `pred` in the
existing helper input/output signature. A same-file forwarding helper now keeps
this contract even when it contains no Boolean expression, static value or import
that would otherwise trigger a graph. Signature validation rejects invalid
input/output kinds before source/native ownership acceptance. Original paths,
lines, body digests, dependency hashes and per-kernel call counts remain intact.

## Experiments and independent gates

The [experiment](../../compiler_exp/gap_closure/python_boolean_abi/helpers/README.md)
retains the frozen baseline, candidate generator, sources and independent raw-byte
references. Production imports no experiment files.

- [Baseline](../../compiler_exp/gap_closure/python_boolean_abi/helpers/baseline.json):
  sixty original annotation rejections across same-file, imported and remote package
  trees, with original file/line diagnostics and no user execution.
- [Offline gate](../../compiler_exp/gap_closure/python_boolean_abi/helpers/candidate-offline.json):
  4,320 executions / 240 kernel-level rows through PTX and normal/poisoned SASS,
  exact complete outputs, inputs, per-thread atomic state, guards and inactive tails.
  Twenty kernels cover forwarding, negation, Boolean arguments, early/loop returns,
  caller reassignment, nested calls, predicate loops, typed conversions, raw FP32
  NaN/infinity/signed-zero selection and eager versus lazy argument effects.
- [Generated controls](../../compiler_exp/gap_closure/python_boolean_abi/helpers/control-properties.json):
  128 predicate helper trees / 1,536 executions, seed 86173, independent recursive
  truth evaluation and once-only ordered atomic argument snapshots.
- [Negative controls](../../compiler_exp/gap_closure/python_boolean_abi/helpers/negative-controls.json):
  22 genuine type, physical ABI, storage, purity, initialization, capture, recursion,
  annotation shadowing and return-completeness rejection boundaries.
- [Toolkit control](../../compiler_exp/gap_closure/python_boolean_abi/helpers/oracle-probe.json):
  actual ptxas accepts all sixty original kernels; complete PTX/cubin/disassembly/
  source/tool-log identities remain retained.
- [Original A100 gate](../../compiler_exp/gap_closure/python_boolean_abi/helpers/candidate-a100-20261124T123953Z.json)
  passes 2,400 launches. The [independent audit](../../compiler_exp/gap_closure/python_boolean_abi/helpers/promotion-audit.json)
  checks **16,220,160 complete bytes** and **240 exact current native word/register/
  frame outputs**. Safe/mined/sched/opt and ptxas agree with the original independent
  numeric/effect references; every OA local frame is zero. This is correctness
  evidence, without timing or a new performance claim.
- All **155 new regressions** pass before promotion, including a standalone
  forwarding helper without an expression marker and malformed signature/native
  custody controls. Existing tests and expected values are unchanged.

The [continuity audit](../../compiler_exp/gap_closure/python_boolean_abi/helpers/application-continuity.json)
freshly regenerates all 240 preceding Boolean-value outputs with identical PTX,
words, controls, registers and frames. It inherits 3,254 earlier exact comparisons
only after checking retained proof/archive hashes and unchanged executable/backend
inputs. The three changed frontend snapshots are explicitly isolated; the first
observer's overly broad input check remains retained. Twenty existing source files
retain 62 accepted PTX kernels and unchanged rejected diagnostics.

The initial isolated test harness omitted its fixture directory; its failure and
corrected isolated copy remain retained. The first independent byte-count assertion
also used the wrong aggregate count after the fixture mix changed; all individual
byte/native comparisons had passed. Its exact observer/log remain retained, and
the corrected count sums the actual full buffers. Neither correction changes
production, a reference value, a tolerance or hardware output.

The full supported Python 3.12.10/Torch 2.7.1 suite passes **4,593 tests** with
zero skips. Python 3.14.5 without Torch passes **4,108 tests with fifteen skips**.
The installed wheel preserves 132 source/data identities and 240 original
source/native regenerations; the public Modal CLI preserves predicate signature
and native/source custody. Live and isolated-export documentation checks pass.
Final source/evidence/log identities are bound in the [integration record](../../compiler_exp/gap_closure/python_boolean_abi/helpers/validation.json).
The full 51-gap program remains incomplete.
