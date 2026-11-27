# Typed Python kernel frontend

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

The [typed helper contract](../../compiler/gaps-python.md) adds same-file
`@device` scalar inlining with explicit signatures, separate local bindings and
effect/recursion limits. Independent Python/PTX/encoded-SASS gates and 72
frontend regressions pass; this adds no cross-module call ABI or GPU performance
claim.

The [source diagnostics contract](../../compiler/semantic-acceptance.md) now
rejects shared-memory reads with no possible producer, alongside existing
literal bounds/zero-division and divergent-barrier checks. This does not provide
general runtime bounds, temporal initialization or race analysis.

A defined numeric subset of Python compiles to PTX
and then native OpenAssembler SM80 instructions. This is not general Python,
NumPy or PyTorch compilation.

## Idea and implementation

Write a GPU kernel with familiar arithmetic, conditions and loops while keeping
types and thread ownership explicit. OpenAssembler parses a source file without
importing it or executing its top-level code. Functions marked `@kernel` become
PTX kernels; that PTX follows the same native compiler route as hand-written PTX.

The subset includes typed `Array(f32/i32/u32/i64/u64)` parameters, scalar types,
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
- `range` steps remain constant nonzero i32 numeric integers. Finite ranges
  exhaust before counter wrap, preserving visible target/bound snapshots;
  range/while else has separately validated nearest-loop exit semantics.
  Arbitrary calls/classes, recursion, dynamic allocation and scalar f16/f64
  arithmetic remain unsupported. Typed pure helpers and the declared
  [atomic primitives](../../compiler/gaps-python-primitives.md) have their own contracts.
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

The [typed Python shared-array extension](../../compiler/gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
adds i64/u64 storage, mixed padding and eight-byte index scaling. A hardware
conditional-writer failure led to a separate bounded CTA convergence pass.
The expanded gate passes 1,280 A100 launches; cyclic control flow and general
shared-memory safety remain open.

## Reproduce

Run from the repository root with Python 3.12+ and CUDA 12.8 `ptxas`/`nvdisasm`.
Compilation and `--verify` do not require a GPU; actual execution does.

```sh
compile examples/vector_add.py --verify --out-dir output_python
run output_python --backend modal --budget 300 --job-timeout 60
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

The [local-package helper extension](../../compiler/gaps-python.md#static-local-package-device-helpers)
now specializes imported pure scalar helpers under their defining module's type
bindings. Direct/module aliases, transitive relative imports and static re-exports
retain exact entry/helper/package source identity, original diagnostics and
CLI/local/remote ownership. The final original-source A100 gate passes 600
launches, 5,529,600 audited bytes and sixty exact native outputs. Arbitrary runtime
imports, mutable captures and host application effects remain open. The separate
[static-value hypothesis](../../../compiler_exp/gap_closure/python_static_values/README.md)
retains its original-source and integration limitations.

[Explicit static source values](../../compiler/gaps-static-values.md) add bounded
Final numeric/tuple evaluation in original source graphs, including imported
aliases, tuple indexing/slicing/len and per-kernel helper specialization. Exact
source/value/diagnostic/native identity is preserved. The final A100 gate passes
1,440 launches and 144 native outputs with independent complete-bit gates. This
extends G05 under explicit source semantics; runtime containers, arbitrary host
state and general metaprogramming remain open.

[Finite static tuple loops](../../compiler/gaps-static-loops.md) now preserve
runtime break/continue, exhausted else, nested tuple/range loops and target state
using a counter/value-selection lowering. Original source/type/loop-value and
convergence/ownership contracts remain explicit. The original-source gate passes
2,400 A100 launches and 240 native outputs. This adds no runtime container,
arbitrary iterable, early-helper-return or general Python control-flow claim.

The [pure helper return-path cycle](../../compiler/gaps-helper-returns.md) now preserves
original branch/loop returns, caller snapshots, live-path initialization and exact
source/native ownership. Its final gate passes 2,400 A100 launches; broader
Python effects remain open; the subsequent runtime loop-control cycle below
now admits range/while else under its separate validated contract.

The [finite runtime loop-control cycle](../../compiler/gaps-range-control.md#original-source-production-extension)
now preserves range/while else and exhausts finite i32 ranges before counter wrap.
Its original-source gate passes 2,400 A100 launches; dynamic steps, wider bounds
and general control/effects retain separate open contracts.

The [typed condition-chain cycle](../../compiler/gaps-comparison-chains.md) preserves
single-evaluation/short-circuit memory and atomic effects, pairwise exact literal
types, special-value truth and original expression/source/native ownership.
Its final gate passes 2,400 A100 launches; general Boolean values and arbitrary
Python effects remain open.

The [Boolean-value cycle](../../compiler/gaps-boolean-values.md) preserves lazy operand
selection, contextual literal types, explicit truth/conversion and original value
ownership. A100 loop failures led to a native predicate-path readiness repair; the
final gate passes 2,400 launches. Boolean ABI and arbitrary type unions remain open.

The [logical Boolean helper extension](../../compiler/gaps-boolean-helpers.md) now admits exact predicate
parameters/returns in pure static expansion, with original source/native signatures
and a 2,400-launch independent A100 gate. Byte-backed launch/storage semantics
remain a separate physical ABI experiment.

The [narrow scalar entry ABI](../../compiler/gaps-narrow-parameters.md) adds exact byte/halfword
slots and extension with coordinated parser/selector/interpreter/verifier and
public explicit-launch bindings. Independent original-PTX/A100 and explicit
run-spec gates pass; logical Python Boolean launch/storage is a separate contract.

The [logical Boolean kernel parameter cycle](../../compiler/gaps-boolean-parameters.md) preserves predicate
body semantics, canonical byte host values and exact source/manifest/native roles.
Its original-source A100 gate passes 9,600 launches plus 960 public Boolean
run-spec launches; Array/Buffer/shared Boolean storage remains separate research.
