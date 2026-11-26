# Pure scalar helper return paths

Status: implemented and integration validated for the bounded original-source contract.

## Gap and selected solution

G04/G06 still rejected useful pure helpers with piecewise, nested or loop early
returns. The [initial normalization hypothesis](../../compiler_exp/gap_closure/python_helper_returns/README.md)
used a separate flag/value source and omitted branch-complete returns and ordinary
break/continue/exhaustion control. It remains historical experimental evidence.

Production now analyzes the original helper body and emits a private scalar
return home and exit label for each expanded call. Every return evaluates and
checks its value, copies it into that home and branches to the helper exit.
Nested/repeated helpers have separate homes and labels. Return from a helper
resumes its caller; it does not emit a kernel exit. The existing single terminal
return keeps its original emission path. No new native call ABI, instruction
encoding, vendor fallback or user module execution is introduced.

```python
from openassembler.pyfront import device, i64

@device
def classify(x: i64) -> i64:
    if x < i64(0):
        return -x
    if x == i64(0):
        return i64(17)
    return x + i64(3)
```

## Path, type, effect and source contract

Helpers retain their explicit i32/u32/i64/u64/f32 scalar annotations and existing
literal/conversion rules. Each emitted return must produce that exact scalar
kind. Missing, bare and wrong-type returns reject with original file/line
information. Integer operations retain their fixed-width wrapping semantics;
FP32 operations retain existing rounding. These are device scalar semantics.

Definite initialization joins only paths that continue to the next statement.
Returning paths leave the join, so a local defined on every remaining path is
safe even when another branch returns. Break and continue states stay separate
and bind to their nearest loop. Runtime range/while loops may execute zero times;
assignments in their bodies do not establish post-loop initialization. Finite
nonempty tuples establish their scalar target and can prove a return when every
first-iteration path returns. Empty tuples neither bind their target nor prove
that their body returns. Body-only assignments remain conservatively uninitialized
after general tuple loops too. Tuple exhaustion else and paths that bypass it
through break retain separate states. The analyzer does not prove termination.

Early returns work inside existing if/while/range/finite-tuple nesting, including
tuple exhaustion else and original break/continue. Runtime range/while else,
dynamic range steps, arbitrary iterables, exceptions, context managers and general
control-flow graphs remain outside this extension. All existing scalar/effect
checks remain active: no helper memory/atomic/barrier/thread-coordinate effects,
mutable captures, array/object arguments, recursion, runtime imports or native
aggregate ABI. The 32-level nesting and 1,024 expanded-call limits remain intact.
Kernel convergence checks continue to see thread-dependent helper arguments.

Arguments are evaluated and copied before entering each helper scope. Mutation
of a parameter or local cannot modify the caller's registers or loop context.
Original entry/helper/package hashes, signatures, complete body hashes and
per-kernel call counts remain in the source graph and native ownership. A single
same-file early-return entry now retains its source graph even without a static
constant or tuple. Changed-dependency rejection and fresh kernel-reference
snapshots remain active through normal local/remote and installed-wheel builds.

## Experiments and validation

The [frozen baseline](../../compiler_exp/gap_closure/python_helper_returns/promotion/baseline.json)
retains sixty original source/entry rejections. The final
[offline gate](../../compiler_exp/gap_closure/python_helper_returns/promotion/candidate-offline.json)
passes **4,320 complete executions / 240 rows** across same-file, imported and
remote-package source trees, twenty kernels, four levels, two seeds and three
lengths. PTX, normal and poisoned SASS compare full independent modulo64/FP32
outputs, guards, inactive tails and unchanged input bytes. Nested-call references
explicitly wrap intermediate signed values before later comparisons.

The [generated control gate](../../compiler_exp/gap_closure/python_helper_returns/promotion/control-properties.json)
adds **64 deterministic original return/skip/break/empty/exhaustion/caller
combinations** and **768 whole-byte executions**. Sources, inputs, independent
expected bits and seed 68193 remain retained. The
[path probe](../../compiler_exp/gap_closure/python_helper_returns/promotion/paths-probe.json)
checks five accepted and seven rejected return/initialization boundaries. The
[24 negative controls](../../compiler_exp/gap_closure/python_helper_returns/promotion/negative-controls.json)
retain missing/type/empty/zero-trip/break/continue/uninitialized/effect/recursion
rejections with original diagnostics. Actual ptxas accepts all three original
trees / sixty kernels with 22 retained source/cubin/listing/log identities.

The final [A100 gate](../../compiler_exp/gap_closure/python_helper_returns/promotion/candidate-a100-20261124T055307Z.json)
passes **2,400 launches** on A100-SXM4-40GB, Python 3.12.1/CUDA 12.8.1. Each of
three original trees has twenty kernels, emitted PTX controls and four OA levels,
two seeds, lengths 0/1/259/384 and three 128-thread CTAs. Original source/PTX/
graphs/manifests/cubins/disassembly and complete before/after input/output bytes
remain archived. The [production audit](../../compiler_exp/gap_closure/python_helper_returns/promotion/promotion-audit.json)
compares **16,465,920 complete independent bytes**, **240 exact native
word/register/frame outputs**, zero local frames, original source/helper ownership
and corrupted-output rejection. This is correctness evidence, with no speed claim.

The [regressions](../../tests/test_pyfront_helper_returns.py) add **156 tests**
covering original paths, numeric types, caller snapshots, nested calls, empty and
exhausted loops, changed return-body snapshots and native ownership tampering.
Four old early-return rejection assertions now use genuine missing-path or
bare-value cases; positive tests exercise the newly admitted forms. Their exact
[contract updates](../../compiler_exp/gap_closure/python_helper_returns/promotion/test-contract-updates.json)
and the first 610-pass/four-stale-assertion run remain retained. No numerical
reference, tolerance, rejection class or resource bound is weakened.

A focused test found missing source ownership on a single same-file entry; that
candidate was repaired and the complete final source/hardware gate rerun. The
first passed A100 archive remains distinct. Other retained failures include a
missing test import, a resolved macOS diagnostic-path alias, an unsupported f32
cast used by a fixture, and a first matrix run excluded because candidate edits
overlapped it. The final matrix freezes all candidate source identities before
execution and rejects mid-run changes. Full suites, installed-wheel regeneration,
public remote publication and prior-code continuity have separate integration
gates in the [experiment record](../../compiler_exp/gap_closure/python_helper_returns/promotion/README.md).

The [integration record](../../compiler_exp/gap_closure/python_helper_returns/promotion/validation.json)
passes **3,962 supported tests** and **3,477 compiler-only tests with fifteen skips**.
The [wheel audit](../../compiler_exp/gap_closure/python_helper_returns/promotion/wheel-audit.json)
checks 132 installed identities and 240 native regenerations. Normal
[public CLI/Modal publication](../../compiler_exp/gap_closure/python_helper_returns/promotion/normal-remote-audit.json)
passes original helper ownership; the
[frontend audit](../../compiler_exp/gap_closure/python_helper_returns/promotion/frontend-continuity.json)
keeps eighteen earlier source files and 62 identical PTX kernels/rejections.
The [continuity audit](../../compiler_exp/gap_closure/python_helper_returns/promotion/application-continuity.json)
newly regenerates 240 preceding tuple originals and separately inherits 2,294
earlier direct replay results after exact proof/archive and backend-input checks.
The aggregate is 2,534 prior occurrences; the direct/inherited counts remain
explicit. All 49 measured policies and 98 original sessions retain their existing
compiler inputs. No previous binary, numerical report or policy is rewritten.

```sh
python compiler_exp/gap_closure/python_helper_returns/promotion/baseline.py
python compiler_exp/gap_closure/python_helper_returns/promotion/offline.py
python compiler_exp/gap_closure/python_helper_returns/promotion/negative_controls.py
python compiler_exp/gap_closure/python_helper_returns/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_helper_returns/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_helper_returns.py
```

Production paths: [return analysis](../../openassembler/pyfront/return_flow.py),
[helper specialization](../../openassembler/pyfront/helpers.py),
[return lowering](../../openassembler/pyfront/compile.py) and
[source ownership](../../openassembler/pyfront/modules.py). Broad G04/G06 and the
complete 51-gap program remain incomplete.

The subsequent [runtime loop-control cycle](gaps-range-control.md#original-source-production-extension)
fixes finite range-counter wrap and adds range/while else under its separate
original-source/control/convergence validation. Dynamic steps and general control
graphs remain open.
