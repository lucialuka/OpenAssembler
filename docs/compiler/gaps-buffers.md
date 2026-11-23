# Checked shaped and strided kernel buffers

Status: implemented

G07 exposed only flat unbounded pointers, and G09 had no runtime global-buffer
bounds checks. The selected foundation adds `Buffer` annotations with literal
shape, element strides, base alignment and read/readwrite access. `Array` keeps
its original unchecked flat-pointer behavior. This closes a bounded safe-index
route; general tensor slicing, negative strides, alias analysis, race freedom and
shared-memory initialization remain open.

```python
from openassembler.pyfront import Buffer, f32, kernel, tid_x

@kernel
def transform(a: Buffer(f32, (3, 4), strides=(6, 1), access="read"),
              out: Buffer(f32, (4, 3), strides=(1, 6))):
    lane = tid_x()
    row = lane // 4
    column = lane % 4
    out[column, row] = a[row, column] + 0.5
```

Shapes contain one to four positive int32 dimensions. Default strides are dense
row-major element strides. Explicit strides are nonnegative integers; padded and
transposed views are admitted. Read-only views may broadcast with zero or
overlapping strides. Writable views must pass a conservative nonoverlap check:
strides ordered from smallest to largest must each cover the lower-dimensional
span. Some legal layouts may therefore reject. Negative strides, zero-size
shapes, runtime shapes and general slice objects are outside the contract.

Every Buffer load/store evaluates each scalar index once, checks `0 <= index <
dimension`, and branches around address calculation and memory access when any
coordinate is invalid. An invalid load returns typed zero; an invalid store has
no memory effect. Literal and runtime negative indices follow this policy and
do not count from the end. Extreme i32 values are checked before multiplication.
The declared physical byte span must fit positive int32, ensuring every admitted
coordinate/stride sum and byte offset fits that domain. Index expressions retain
the language's wrapping i32 arithmetic before the bounds check; the checker does
not recover an unbounded Python index from an already-overflowed expression.

Supported element types are f32/i32/u32/i64/u64. Index rank must match exactly and
every coordinate must be i32. Loads/stores use the exact element width. Stores
require matching element types and reject writes to a read-only annotation.
Augmented Buffer stores currently reject; spell a checked load followed by an
assignment explicitly. Buffer values cannot be passed into scalar device
helpers. Flat Array accesses elsewhere in the same kernel remain unchecked.

Base alignment defaults to element size and may be a power of two through 256
bytes. This promises the view's base alignment, not that every strided element
has that larger alignment. Callers must bind storage spanning
`1 + sum((shape[d]-1)*strides[d])` elements. Device checks know logical coordinates,
not the true size of an arbitrary raw CUDA allocation; an undersized raw-driver
allocation violates the caller contract.

## Explicit launch and metadata validation

Ordinary compilation saves each Buffer contract in the kernel manifest and the
native ownership inventory, with the generated PTX hash. Explicit `--spec` launch
planning verifies the saved PTX hash and its contract against the manifest,
checks buffer element dtype, remaining storage extent and base offset alignment,
and rejects raw integer addresses for Buffer arguments. Overlapping bindings to
the same storage buffer reject whenever either view may write; read-only aliases
are allowed. The interval check is conservative for views containing holes.
Distinct buffer names in the spec allocate distinct CUDA buffers. Direct driver
users must enforce equivalent allocation and alias obligations themselves.

Automatic guessed runtime launches skip Buffer kernels and request an explicit
spec. Generic `--verify` currently reports `not-verified` with an explicit shaped
allocation diagnostic and return code 2. It does not substitute a guessed
flat allocation for a shaped/strided contract. The independent gates below
supply exact physical buffers and compare all bytes/elements. No general memory
safety, synchronization or convergence proof is implied.

## Research and validation

The [candidate](../../compiler_exp/gap_closure/buffers/checked.py),
[source overlay](../../compiler_exp/gap_closure/buffers/make_candidate.py) and
[pre-feature snapshots](../../compiler_exp/gap_closure/buffers/before/)
retain the experiment before promotion. The
[candidate result](../../compiler_exp/gap_closure/buffers/candidate-result.json)
and [production replay](../../compiler_exp/gap_closure/buffers/production-result.json)
compare independent shape/stride integer indexing with PTX and all four native
optimization levels, including poisoned-register execution. The fixture tests
padded transpose views, readonly broadcast views, i64 values, negative and
extreme i32 indices, zero results from invalid loads, ignored invalid writes,
input immutability, padding holes and allocation guards.

The [final hardware report](../../compiler_exp/gap_closure/buffers/hardware-20261121T141341Z.json)
and [archive](../../compiler_exp/gap_closure/buffers/hardware-20261121T141341Z.zip)
pass on **NVIDIA A100 80GB PCIe**, Python 3.12.1, CUDA toolkit 12.8.1. Fifteen
ordinary OA and same-PTX ptxas launches across safe/mined/sched/opt variants pass
640 exact element comparisons and complete allocation guards. Three additional
saved native explicit-spec executions pass full outputs. Undersized, misaligned,
aliased-writable and incorrectly typed bindings reject before GPU launch. This
device cohort differs from the earlier A100-SXM4-40GB language experiments;
there are no timings to pool or speed claims.

The [initial hardware attempt](../../compiler_exp/gap_closure/buffers/hardware-20261121T141152Z.json)
stopped before execution because the harness incorrectly expected generic
`--verify` return code 0. The actual return code 2 correctly represented explicit
`not-verified` outcomes. Its full failure record is retained; the final harness
separately checks that outcome and uses exact application buffers for hardware
validation. No numerical failure is hidden. nvdisasm mnemonic-alias warnings
remain in the logs; native ownership and complete outputs passed.

The frontend/runner/typed-verifier selection passes 220 tests, including 34 new
Buffer checks. [PTX continuity](../../compiler_exp/gap_closure/buffers/32bit-continuity.json)
retains identical output for 39 existing scalar/wide kernels. The
[runner continuity proof](../../compiler_exp/gap_closure/buffers/runner-continuity.json)
records unchanged explicit-spec function ASTs after separate pointer-annotation
auto-harness changes; it does not claim the whole runner file is unchanged.

```sh
python3 compiler_exp/gap_closure/buffers/probe.py --production
python3 -m pytest -q tests/test_pyfront_buffers.py tests/test_pyfront_wide.py tests/test_runner_wide.py tests/test_pyfront.py tests/test_pyfront_static.py tests/test_pyfront_semantics.py tests/test_pyfront_helpers.py tests/test_pyfront_helper_control.py tests/test_runner.py tests/test_verify_typed.py
modal run compiler_exp/gap_closure/buffers/run_modal.py
```

Production: [Buffer contract and lowering](../../openassembler/pyfront/buffers.py),
[frontend](../../openassembler/pyfront/compile.py),
[explicit binding checks](../../openassembler/runner.py),
[native ownership](../../openassembler/native_ownership.py) and
[regressions](../../tests/test_pyfront_buffers.py). Production never imports the
experiment overlay. G07/G09 remain partial beyond these named contracts.
