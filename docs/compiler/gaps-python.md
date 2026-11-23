# Typed scalar device helpers

Status: implemented

G04 blocked reusable numeric kernel code: the static Python frontend rejected
all user helper calls. The selected solution is same-file `@device` specialization
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
`range` with a constant nonzero step, `while`, `break`/`continue`, and one
terminal returned expression. They may call pure scalar math intrinsics and other admitted helpers.
Each call copies arguments into fresh registers and allocates separate local
bindings, including repeated/nested calls. Integer literals adopt the declared
parameter type only under the existing exact-literal rules; other conversions
must be explicit. Arithmetic retains the frontend's wrapping integer and FP32
rounding contracts. This is not arbitrary Python integer/float semantics.

Memory operations, arrays as helper values, barriers/thread coordinates inside
helpers, global captures, early helper returns, recursive calls, defaults,
keyword arguments and imported helpers reject with source diagnostics. Module
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
[wide integer extension](gaps-python-types.md) adds bounded i64/u64 support; arrays remain
flat unchecked pointers, device linking/spilling and general memory ordering
remain unsupported, and the backend's modifier admission is distinct from its
operand/resource/encoding domain. This helper change adds no ISA forms or target
architectures. Multi-module helpers, early/multiple helper returns, buffer and
effect contracts, richer device types and the remaining GPU primitives require
separate research and validation.


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
