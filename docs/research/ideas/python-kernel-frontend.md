# Typed Python kernel frontend

Status: implemented

A defined numeric subset of Python compiles to PTX
and then native OpenAssembler SM80 instructions. This is not general Python,
NumPy or PyTorch compilation.

## Idea and implementation

Write a GPU kernel with familiar arithmetic, conditions and loops while keeping
types and thread ownership explicit. OpenAssembler parses a source file without
importing it or executing its top-level code. Functions marked `@kernel` become
PTX kernels; that PTX follows the same native compiler route as hand-written PTX.

The subset includes typed `Array(f32/i32/u32)` parameters, scalar types,
array indexing, integer/float arithmetic, comparisons, boolean `and`/`or`/`not`,
supported `if`/`for`/`while` control flow, `break`/`continue`, early returns,
shared arrays and synchronization.
Thread/block index intrinsics state which thread handles each element.
Supported math functions distinguish exact and approximate operations.
Unsupported syntax produces a source-line diagnostic instead of running Python
on the GPU or silently changing the implementation.

Python-like signed floor division/remainder and GPU numeric behavior need
explicit handling; the semantic tests cover negative operands and boundary
cases. The module documents differences from unrestricted Python integers
and floating arithmetic. Syntax support is not a claim that arbitrary loops
or a naive kernel mapping will be fast.

The important numeric/structural rules are:

- `/` produces FP32, even from integers. Integer `//` and `%` implement Python's
  floor/sign convention; floating `//` and `%` are rejected.
- Integers have fixed GPU widths and floats round to FP32, unlike Python's
  arbitrary-size integers/usual 64-bit floating point. Mixed integer/float arithmetic needs
  explicit casts. Integer zero division/overflow follow the emitted PTX/device
  behavior, not Python exceptions. Float zero division yields IEEE Inf/NaN;
  `int(f32)` truncates/saturates to i32, mapping NaN to zero.
- `sqrt` is correctly rounded; `exp`, `log`, `tanh` and `rsqrt` use approximate
  GPU operations. Supported `math` aliases are recognized statically.
- `range` steps must be constant and nonzero, with Python's visible loop-variable
  behavior retained. Arbitrary calls/classes, recursion, dynamic allocation,
  atomics and scalar f16/f64 arithmetic are unsupported.
- `shared_array(TYPE,CONSTANT_SIZE)` and `syncthreads()` support block-local
  cooperation. Thread-dependent barriers are rejected conservatively; valid
  cross-warp `bar.sync 0` is modeled by the interpreters, while unmodeled or
  deadlocking synchronization fails clearly.

## Methodology and findings

Regression suites compare emitted PTX, static errors, and executable semantics,
including a three-way Python/PTX/SASS check for supported cases. Compiling a
source file does not execute its imports. Native neural examples establish
that the route works for activations and simple reductions, while the
[serial reduction results](cooperative-softmax.md) show that source mapping can
dominate performance even when compilation is correct.

## Reproduce

Run from the repository root with Python 3.12+ and CUDA 12.8 `ptxas`/`nvdisasm`.
Compilation and `--verify` do not require a GPU; actual execution does.

```sh
openassembler compile examples/vector_add.py --verify --out-dir output_python
openassembler run output_python --backend modal --budget 300 --job-timeout 60
.venv/bin/python -m pytest -q tests/test_pyfront.py tests/test_pyfront_semantics.py \
  tests/test_pyfront_static.py tests/test_interp_barrier.py tests/test_cli_compile.py
```

The default run harness suits simple one-dimensional elementwise kernels.
Different buffers, grids or shared-memory launches require `run --spec`.
Toolkit-dependent tests can explicitly skip when NVIDIA tools are unavailable.
See [COMMANDS.md](../../COMMANDS.md) for complete options.

## Source and evidence

- [Compiler](../../../openassembler/pyfront/compile.py),
  [public API and supported subset](../../../openassembler/pyfront/__init__.py),
  [types](../../../openassembler/pyfront/types.py),
  [intrinsics](../../../openassembler/pyfront/intrinsics.py).
- [Semantic tests](../../../tests/test_pyfront_semantics.py),
  [static tests](../../../tests/test_pyfront_static.py),
  [three-way checker](../../../tests/pyfront_threeway.py).
- [Recorded CLI integration](../../evidence/integration-20261104.json).

The [native Python model graph route](native-model-bridge.md) is separate.
The older [Inductor application experiment](pytorch-application-capture.md) is
retired and unavailable through production commands.
More Python syntax, automatic parallelization and automatic substitution into
model graphs remain future work, not implied frontend features.
