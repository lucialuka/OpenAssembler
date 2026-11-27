# Typed scalar device helpers

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

G04 blocked reusable numeric kernel code: the static Python frontend rejected
all user helper calls. The initial solution was same-file `@device` specialization
by inlining. It introduces no separate call ABI and executes no user module code
during compilation. This narrows G04; general Python/package calls and device
linking remain open.

```python
from openassembler.pyfront import Array, device, f32, kernel, tid_x

@device
def affine(x: f32, gain: f32, bias: f32) -> f32:
    x *= gain
    return x + bias

@kernel
def transform(x: Array(f32), out: Array(f32)):
    i = tid_x()
    out[i] = affine(x[i], 2.0, 0.5)
```

Helpers require explicit `i32`, `u32`, `i64`, `u64` or `f32` parameter and return annotations,
plain positional arguments, scalar assignments, `if`/`elif`/`else`, `for` over
`range` with a constant nonzero step, `while`, `break`/`continue`, finite static
tuples and [typed scalar return paths](gaps-helper-returns.md). They may call pure
scalar math intrinsics and other admitted helpers.
Each call copies arguments into fresh registers and allocates separate local
bindings, including repeated/nested calls. Integer literals adopt the declared
parameter type only under the existing exact-literal rules; other conversions
must be explicit. Arithmetic retains the frontend's wrapping integer and FP32
rounding contracts. This is not arbitrary Python integer/float semantics.

The initial same-file contract rejects memory operations, arrays as helper values,
barriers/thread coordinates inside helpers, unmarked global captures, early returns,
recursive calls, defaults, keyword arguments and imported helpers. The local-package
extension below now admits its declared imported-helper subset with source diagnostics. Module
helper rebinding and lexical shadowing of call targets also reject. Definitions
may follow their callers. Import aliases of the `device` marker are recognized.
Specialization is capped at 32 nested helper calls and 1024 expanded call sites
per kernel, preventing recursive or exponential compile-time expansion. The
decorator raises if called as ordinary host Python.

The retained [candidate](../../compiler_exp/gap_closure/python/candidate.py) was
validated before promotion. Its [original result](../../compiler_exp/gap_closure/python/candidate-result.json)
records the old production rejection and passing independent Python/PTX/encoded
SASS emulator comparisons. The [production replay](../../compiler_exp/gap_closure/python/production-result.json)
records source hashes, all inputs/expected outputs, and the same emitted PTX
hash. Six active threads plus two inactive tails exercise nested FP32 helpers,
negative integer quotient/remainder, u32 wrap, argument isolation, caller-local
isolation and input/tail immutability. Each emulated leg compares 48 array
elements. No GPU execution or performance measurement was performed for that initial
straight-line candidate.

Reproduce the production gate and regression suite from the repository root:

```sh
python3 compiler_exp/gap_closure/python/probe.py --production
python3 -m pytest -q tests/test_pyfront_helpers.py tests/test_pyfront.py tests/test_pyfront_static.py tests/test_pyfront_semantics.py
```

The recorded environment is Python 3.12.10 on macOS 15.6.1 arm64. The four test
files passed 72 tests. Rejection controls cover signatures, type mismatches,
missing return values, memory access, recursion/cycles, host calls, effects,
module rebinding and local/attribute-call shadowing.

Production implementation: [helper specialization](../../openassembler/pyfront/helpers.py),
[frontend](../../openassembler/pyfront/compile.py) and
[regressions](../../tests/test_pyfront_helpers.py). The experiment imports
compiler internals; production never imports experiment code.

The initial G03–G18 review found scalar types limited to i32/u32/f32. The
[wide integer extension](gaps-python-types.md) adds bounded i64/u64 support;
[checked buffers](gaps-buffers.md) add explicit shapes/strides beside flat
unchecked arrays, and [atomic primitives](gaps-python-primitives.md) add bounded
global effects. Later [scalar device/source linking](gaps-device-functions.md), [bounded spilling](gaps-spilling.md)
and [atomic ordering](gaps-atomics.md) address their named foundation contracts.
The backend's modifier admission stays distinct from its operand/resource/encoding
domain; these bounded additions do not imply a general ISA or runtime. This helper change adds no ISA forms or target
architectures. At that initial stage, multi-module helpers and early returns were still open.
The local-package extension below and the [return-path dossier](gaps-helper-returns.md)
now address their bounded source/control subsets. General effects, richer types
and remaining GPU primitives retain separate gates.


## Structured helper control flow

G04/G06 still excluded useful numeric helpers containing loops or branches. A
second [candidate](../../compiler_exp/gap_closure/python_control/candidate.py)
adds the existing kernel control-flow vocabulary inside pure scalar helpers.
The [baseline/candidate record](../../compiler_exp/gap_closure/python_control/candidate-result.json)
retains the old rejection and the independent Python/PTX/SASS gate before
promotion. The [production replay](../../compiler_exp/gap_closure/python_control/production-result.json)
retains matching PTX/native hashes and all inputs/outputs across safe, mined,
sched and opt compiler levels.

A conservative definite-initialization pass intersects the definitions from
both branches and assumes every loop can execute zero times. Parameters are
initialized on entry; branch-only locals and new loop locals cannot be read
after the construct. An existing initialized loop target retains its value for
an empty range. Nested early returns, loop `else`, memory effects and calls
shadowed by scalar locals reject with their original source lines. This is
bounded dataflow analysis, not a proof of loop termination: integer loop
arithmetic follows the existing wrapping i32 contract, and callers must supply
terminating loops. No new instructions, runtime ABI or synchronization rules
were introduced. The existing barrier checker still rejects caller barriers
controlled by thread-dependent helper arguments.

The control-flow fixture covers negative/zero/positive branch values, empty and
negative-step ranges, loop-carried values, break/continue, nested repeated
helpers, copied arguments/caller locals, dyadic FP32 arithmetic, input immutability
and inactive tails. The offline gate compares independent Python bodies with
PTX and all four encoded SASS variants. The five frontend regression files
passed 88 tests, including 16 control-flow/initialization checks. The source
contains a top-level exception to prove compilation never executes it.

Reproduce:

```sh
python3 compiler_exp/gap_closure/python_control/probe.py --production
python3 -m pytest -q tests/test_pyfront_helper_control.py tests/test_pyfront_helpers.py tests/test_pyfront.py tests/test_pyfront_static.py tests/test_pyfront_semantics.py
modal run compiler_exp/gap_closure/python_control/run_modal.py
```

The hardware gate compiles the Python fixture through the ordinary compiler at
all four optimization levels, validates OA native ownership, packages and
nvdisassembles each cubin, and compares its outputs with independent Python
bodies and a separate same-PTX ptxas control. Each executable runs with changed
inputs and active lengths 0, 14 and 16, using 32 launched lanes, guarded input
and output allocations and exact equality of all 64 array elements. This tests
15 launches and 960 element comparisons without a speed claim. Per-launch full
outputs/expected inputs, device/toolkit versions, resources, compiler source
hashes, native manifests, cubins and listings remain in the hardware records and
archives under the experiment directory.

The [final hardware report](../../compiler_exp/gap_closure/python_control/hardware-20261121T122924Z.json)
and [archive](../../compiler_exp/gap_closure/python_control/hardware-20261121T122924Z.zip)
pass all gates on NVIDIA A100-SXM4-40GB, Python 3.12.1 and CUDA toolkit 12.8.1.
The final source hashes match the uploaded payload. The
[initial hardware run](../../compiler_exp/gap_closure/python_control/hardware-20261121T122804Z.json)
also passed; the final replay adds a local-call shadowing rejection and source
contract documentation. The generated fixture PTX is identical in both runs.
`nvdisasm` prints signed immediate aliases such as `-0x1` for the encoded
`0xffffffff`; those textual warnings remain in the logs and do not indicate a
native-word, ownership or numerical mismatch.

## Static local-package device helpers

The original [source-graph baseline](../../compiler_exp/gap_closure/python_module_helpers/README.md)
reproduced three G04 call-target failures: direct import aliases, module aliases
and transitive relative imports. The initial normalization experiment passed
functional checks but lacked package initializer identity and original diagnostic
paths. The [promotion cycle](../../compiler_exp/gap_closure/python_module_helpers/promotion/README.md)
instead compiles each helper under its defining module's namespace, preserving
original annotations, local bindings and source lines. This validated result is
now production code.

```python
# arithmetic/helpers.py
from openassembler.pyfront import device, i64

@device
def product(x: i64, y: i64) -> i64:
    return x * y

# kernels.py
from openassembler.pyfront import Array, i32, i64, kernel, tid_x
from arithmetic.helpers import product as multiply

@kernel
def apply(a: Array(i64), b: Array(i64), out: Array(i64), n: i32):
    i = tid_x()
    if i < n:
        out[i] = multiply(a[i], b[i])
```

The root defaults to the directory above the enclosing regular package, or the
entry's directory when it is outside a package. `--python-source-root DIR` (or
`compile_file(..., source_root=DIR)` and `find_kernel(..., source_root=DIR)`)
selects an explicit local root. The resolver reads only local `.py` source
within that root. It supports direct/module aliases, transitive relative imports
and static helper re-exports. Definitions retain their own annotation/intrinsic
bindings and copied scalar parameters. Entry/helper/package top-level statements
are never executed. Installed/runtime imports, `sys.path` lookup, dynamic import
hooks, nonlocal sources, symlinked dependencies, ambiguous file/package names,
conditional/star helper imports, rebinding/attribute mutation, recursion, mutable
captures, untyped calls and host/device effects outside the existing pure helper
contract reject. Pure helper bodies retain existing control-flow and type limits.
No arbitrary package or host application compatibility is implied.

The source graph is bounded by **64 files/modules**, **8 MiB** of source,
**32 nested helper calls/re-export steps**, and the existing **1,024 expanded call
sites per kernel**. Package initializers on the resolved paths participate in
identity even though their host code is not executed. Every source is checked
against the compilation snapshot. A changed helper invalidates its dependency
identity; changes during frontend or backend compilation prevent publication.
Repeated `compile_kernel(find_kernel(...))` calls reread source snapshots.
Imported signature/body/syntax diagnostics retain the actual helper file and
line. Module import discovery excludes function-local and conditional imports;
local call-target shadowing and module rebinding remain failures.

Generated PTX and each owned kernel record the versioned source graph: exact
entry and dependency paths/hashes, helper AST body hashes, scalar signatures,
original lines and per-kernel expanded call-site counts. Ownership snapshots the
complete graph. Normal local/Modal/SSH builds use the same frontend. Remote
bundling includes reachable ignored helpers, preserves the package root and
explicit source root, validates the received graph, and rebases returned entry
paths before normal local publication. No experiment import or vendor fallback
is introduced in production.

The [offline gate](../../compiler_exp/gap_closure/python_module_helpers/promotion/candidate-offline.json)
passes **1,080 complete executions**: five original source-tree variants, three
kernels, four levels, two seeds, lengths 0/91/128, and PTX/normal/poisoned SASS.
Independent references use complete modulo64 products and exactly representable
FP32 values, with all input/output guards and inactive tails. The
[negative controls](../../compiler_exp/gap_closure/python_module_helpers/promotion/negative-controls.json)
pass 29 import/type/effect/recursion/rebinding/source-snapshot checks. The
[focused production tests](../../tests/test_pyfront_module_helpers.py) pass 73
cases, including original syntax diagnostics, changed-during-build rejection,
manifest/graph tampering and an actual serialized worker with an ignored helper.
All 273 earlier frontend tests and 100 earlier build/ownership/database tests
also pass. A first recursion diagnostic changed its established spelling; the
candidate restored that spelling while keeping module-specific cycle identity.

The original-source A100 gates retain exact source-tree, emitted PTX, manifest,
cubin, disassembly and complete before/after input/output archives. They cover
normal CLI builds, package re-exports, entry packages, explicit roots and the
serialized worker. The independent audit checks **600 launches**, **5,529,600
complete bytes**, **60 exact native word/register outputs**, zero local frames,
full source/helper/ownership contracts and a deliberate corrupted-output control.
These are correctness results; there is no timing or speed claim. The experiment
record keeps final source identities, earlier successful gates, diagnostic/harness
failures, installed-wheel/public CLI evidence and full integration separate.
Both rerun suites pass **3,538 supported tests** and **3,053 compiler-only tests
with fifteen skips**. The [installed wheel](../../compiler_exp/gap_closure/python_module_helpers/promotion/wheel-audit.json)
checks 130 source/data identities and sixty exact original-source PTX/native
regenerations. The [policy audit](../../compiler_exp/gap_closure/foundations_20261121/audit-20261123T224235Z.json)
retains all 49 policies and 98 original sessions. The
[complete replay](../../compiler_exp/gap_closure/python_module_helpers/promotion/application-continuity.json)
retains all 2,090 earlier native occurrences; its supplemental adapter binding
requires unchanged generator AST and the final source-exact hardware admission.
The [integration record](../../compiler_exp/gap_closure/python_module_helpers/promotion/validation.json)
binds final sources, failed first suites, corrected frontend routing, independent
byte/native/ownership proof and documentation. General G04/G42, mutable captures,
arbitrary packages and application host effects remain open.

```sh
compile kernels.py --python-source-root . --out-dir output
compile kernels.py --python-source-root . --build-backend modal --out-dir output_remote
python compiler_exp/gap_closure/python_module_helpers/promotion/offline.py
python compiler_exp/gap_closure/python_module_helpers/promotion/negative_controls.py
python compiler_exp/gap_closure/python_module_helpers/promotion/oracle_probe.py
modal run compiler_exp/gap_closure/python_module_helpers/promotion/run_modal.py
python -m pytest -q tests/test_pyfront_module_helpers.py
```

## Explicit static values in kernels and helpers

The [static-value extension](gaps-static-values.md) now admits original explicit
Final numeric/tuple declarations under bounded compile-time evaluation. Constants
retain defining-module types, original source hashes/diagnostics, exact binary64/
integer/tuple identity and per-kernel use records through local-package imports,
normal builds and native ownership. All 1,440 original-source A100 launches and
144 native outputs pass complete bit/guard/input gates. The separate normalized
prototype and its annotated-rebinding counterexample remain historical evidence.
Unmarked mutable captures, runtime containers, arbitrary metaprogramming and host
effects remain open G05 work.

## Finite tuple iteration in helpers

The [static tuple-loop contract](gaps-static-loops.md) extends the pure scalar
helper vocabulary with finite numeric tuple iteration, including empty/nested
loops, break/continue, exhausted else and observable target mutation/retention.
Original type/source/convergence and ownership contracts remain explicit. All
2,400 original-source A100 launches pass, with 240 exact native outputs.
Subsequent [pure helper returns](gaps-helper-returns.md) have separate original-source
validation; arbitrary runtime iterables and general control/effects remain open.

## Pure helper early returns

The [return-path contract](gaps-helper-returns.md) now admits original piecewise,
branch-complete, range/while/nested tuple early returns with typed return homes,
caller isolation, path-sensitive initialization and source/native ownership.
The final original-source gate passes 2,400 A100 launches, 16,465,920 complete
independent bytes and 240 exact native outputs. The normalized prototype remains
historical; missing/bare/type/effect/initialization rejection controls stay active.
The subsequent [runtime loop-control cycle](gaps-range-control.md#original-source-production-extension)
adds range/while else and prevents finite range-counter wrap. General Python
effects, arbitrary iterables and runtime steps remain open. Full integration
evidence is retained in the linked dossiers.

## Typed condition comparison chains

The [condition-chain contract](gaps-comparison-chains.md) adds original adjacent
relations with single-evaluated middle values, short-circuit effects, exact
pairwise literal contexts and source/expression/native ownership. The final
original-source gate passes 2,400 A100 launches; first-class Boolean values and
general Python effects remain open.

## Boolean values and operand selection

The [Boolean-value contract](gaps-boolean-values.md) now preserves lazy numeric
operand selection, explicit Boolean locals/truth/conversion, original value/source
ownership and native predicate readiness across control-flow edges. Final original
source gates pass 2,400 A100 launches after retaining the 54 actual first failures.
Boolean array/parameter ABI and arbitrary Python type unions remain open.

## Logical Boolean helper signatures

The [Boolean helper extension](gaps-boolean-helpers.md) admits exact predicate
arguments/returns in pure static helper expansion, complete return paths and caller
isolation, with original signatures and source/native custody. Its original-source
A100 gate passes 2,400 launches. Boolean launch parameters and byte-backed arrays
remain separate ABI/storage research.
