# Finite static tuple iteration

Status: implemented for the original-source contract below; G05/G06 remain partial.

## Gap and selected solution

The [original tuple-loop experiment](../../compiler_exp/gap_closure/python_static_loops/README.md)
reproduced a finite polynomial helper rejected by the range-only frontend. Its
separate normalized-source unrolling hypothesis passed a small numerical gate,
but omitted empty loops, target values and break/continue/else/nested semantics.
The [control matrix](../../compiler_exp/gap_closure/python_static_loops/control-matrix.json)
retains eight original requirements and independent expected values. That
prototype is historical evidence, not the production implementation.

Production now lowers a finite static tuple to a runtime counter, a scalar-value
selection chain, the original loop body, and explicit continue/exhaustion/break
labels. The body is emitted once. This preserves thread-dependent exits and
observed target state without replacing a general loop with unconditional source
copies. No backend/encoding template changes or vendor fallback are introduced.

```python
from typing import Final
from openassembler.pyfront import Array, i32, i64, device, kernel, tid_x

COEFFICIENTS: Final[tuple[int, ...]] = (3, -5, 7)

@device
def polynomial(x: i64) -> i64:
    result = i64(0)
    for coefficient in COEFFICIENTS:
        result = result * x + i64(coefficient)
    return result

@kernel
def apply(a: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = polynomial(a[i])
```

## Source, control-flow and type contract

The iterable must resolve to a finite immutable tuple from a literal or an
[explicit static value](gaps-static-values.md), including imported/re-exported
Final values and compile-time slices. Runtime values, mutable lists/dicts,
dynamic generators/iterators, tuple destructuring, nested tuple elements and
Boolean iteration values reject. Tuple length is bounded by the existing
256-element static-value limit. Numeric scalar elements must fit one device type.
A kernel may select at most 1,024 tuple elements across its lowered tuple-loop
sites, including expanded helper sites; the compiler does not unroll their
runtime iteration product or promise bounded total runtime for arbitrary nesting.

A previously defined target retains its declared device type, and each tuple
element must fit that type under the existing literal rules. A new target uses
i32 when all integers fit, then i64, or u64 for a nonnegative wider domain. A
mixed-sign tuple that cannot fit one type rejects. A tuple containing floating
values selects f32; integer elements must convert exactly under the existing
literal policy. Existing integer targets do not silently accept floating values.
FP32 elements retain the frontend's device conversion/rounding semantics; this
adds no FP64 arithmetic or implicit mixed-width runtime promotion.

Each iteration binds the target from the tuple before executing the body.
Reassigning that target inside the body does not alter the internal iteration
counter or the next selected tuple element. The last observable target value
survives normal completion or break. An empty tuple performs no target binding
or body effects; an existing target stays unchanged and an undefined helper
local remains uninitialized. The helper's conservative flow analysis does not
infer arbitrary body assignments as definitely initialized after a loop.

`continue` reaches the next selection; `break` exits the nearest active loop.
Loop `else` runs on exhaustion, including empty tuples, and is skipped by break.
Nested tuple loops, tuple loops inside range loops, range loops inside tuple
loops and nearest-loop `else` exits retain their control targets. Tuple-loop
`else` itself uses the enclosing loop context, matching Python's control binding.
Kernel early returns retain the existing kernel exit semantics. Subsequent
[pure helper return paths](gaps-helper-returns.md) add separately validated scalar
early returns. Subsequent [runtime loop control](gaps-range-control.md#original-source-production-extension)
adds separately validated range/while else and finite counter exhaustion.

The existing barrier checker propagates thread-dependent breaks/continues into
remaining loop bodies and tuple exhaustion/else paths. A potentially divergent
barrier in those paths rejects before native execution. This is a conservative
convergence contract, not general proof of synchronization or memory safety.

Original tuple declarations, helper modules and entry/package files retain their
source hashes. Each static iteration records its original path/function/line/
column, exact tagged tuple values and per-kernel identity in the versioned PTX
source graph and native ownership. Same-file/literal-tuple kernels also retain
source identity. Fresh source snapshots and changed-dependency checks remain
active through ordinary local, CLI, remote worker and installed-wheel builds.
No user module/package top-level code executes during compilation.

## Experiments and validation

The [frozen baseline](../../compiler_exp/gap_closure/python_static_loops/promotion/baseline.json)
retains **60 original source/entry rejections**. The
[offline candidate](../../compiler_exp/gap_closure/python_static_loops/promotion/candidate-offline.json)
passes **4,320 complete executions**: three original source trees, twenty kernels,
four levels, two seeds, lengths 0/91/128 and PTX/normal/poisoned SASS. Independent
references cover polynomial arithmetic, dynamic signed comparisons, break/continue,
empty tuples, retained/mutated targets, nested/range/tuple control, exhaustion else,
wide signed/unsigned values, slices and exactly representable FP32 arithmetic.
All output/input/guard/tail bytes are compared.

The [generated control gate](../../compiler_exp/gap_closure/python_static_loops/promotion/control-properties.json)
adds **128 deterministic original tuple/skip/break/target-mutation/exhaustion
combinations** and **1,536 whole-byte executions**. Each has a separately computed
Python loop reference with explicit modulo64 device arithmetic. Full source,
input and expected values are retained; these checks are finite CPU/emulator
evidence and add no GPU/timing claim.

The [negative controls](../../compiler_exp/gap_closure/python_static_loops/promotion/negative-controls.json)
retain twenty type/iterable/binding/resource/helper-return/convergence failures.
The [first target-type failure](../../compiler_exp/gap_closure/python_static_loops/promotion/first-target-type-failed.log)
exposed an existing literal helper's permissive float path when reused for an
integer target. The tuple lowering now explicitly rejects a floating element for
an integer target. The failed candidate emitter remains retained; no numerical
tolerance or control is weakened. A malformed fixture initially used Python
keywords as function names; its rejection log is retained separately.

The final [A100 gate](../../compiler_exp/gap_closure/python_static_loops/promotion/candidate-a100-20261124T051600Z.json)
passes **2,400 launches** on A100-SXM4-40GB with Python 3.12.1/CUDA 12.8.1:
three original source trees, twenty kernels, original emitted PTX controls and
four OA levels, two seeds, lengths 0/1/259/384, and three 128-thread CTAs. It
retains original files, PTX/source graphs, manifests, cubins/disassembly and
complete before/after input/output bytes. The
[production audit](../../compiler_exp/gap_closure/python_static_loops/promotion/promotion-audit.json)
checks **16,465,920 complete bytes**, **240 exact native word/register/resource
outputs**, zero local frames, source/loop/helper ownership and corrupted-output
rejection. Actual ptxas accepts all three source trees / sixty kernels, with
22 retained raw artifact identities. These are correctness checks, with no timing
or performance claim.

The [production regressions](../../tests/test_pyfront_static_loops.py) pass
**150 focused tests**, including kernel returns, empty-else target retention,
nearest-loop else/break binding, changed original constants and iteration metadata
or manifest tampering. All 464 preceding frontend tests pass. The experiment
[integration record](../../compiler_exp/gap_closure/python_static_loops/promotion/validation.json)
binds **3,806 supported tests**, **3,321 compiler-only tests with fifteen skips**,
and **2,294 prior native occurrences** with exact words/registers. These gates
retain earlier numerical/source/ownership proof chains without rewriting them.
The [wheel audit](../../compiler_exp/gap_closure/python_static_loops/promotion/wheel-audit.json)
checks 131 source/data identities and all 240 original-source PTX/native outputs.
The [public Modal publication](../../compiler_exp/gap_closure/python_static_loops/promotion/normal-remote-audit.json)
keeps original source/tuple graphs and native ownership through ordinary CLI/worker
builds. The [frontend audit](../../compiler_exp/gap_closure/python_static_loops/promotion/frontend-continuity.json)
retains eighteen earlier files / 62 identical PTX kernels and rejections. All 49
measured policies and 98 sessions retain exact code in the
[policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261124T052003Z.json).
General iteration, containers, runtime Python semantics, unbounded resources and
host/metaprogramming effects remain outside this bounded G05/G06 extension.

```sh
python compiler_exp/gap_closure/python_static_loops/promotion/baseline.py
python compiler_exp/gap_closure/python_static_loops/promotion/offline.py
python compiler_exp/gap_closure/python_static_loops/promotion/negative_controls.py
python compiler_exp/gap_closure/python_static_loops/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_static_loops/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_static_loops.py
```
