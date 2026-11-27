# Native PTX input and compilation boundary

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Status: implemented

The November 19 [semantic acceptance repair](../../compiler/semantic-acceptance.md)
fixes signed saturation and predicate FTZ, adds explicit scalar modifier/type/
operand admission and verifies outputs against independent math and ptxas on
A100. The remaining unsupported forms below stay outside that contract.

Supported SM80 PTX reaches OpenAssembler's own
instruction selection, register allocation, scheduling and SASS encoder.
Unsupported forms reject; the production CLI disables vendor GPU fallback.
The [entry completeness/ABI contract](../../compiler/source-completeness.md)
records bare/weak entries, launch-bound headers, supported scalar parameters,
inventory checks and runtime block constraints, including its A100 evidence.

## Idea in plain language

PTX is a virtual GPU instruction language, not the instruction stream A100
executes. Compiling it directly makes the replacement boundary precise:
OpenAssembler generates the native SASS, while NVIDIA tools provide an ELF
container, a same-source control and independent disassembly. The GPU executes
the resulting cubin; Python orchestrates compilation/loading, not each GPU op.

Scalar kernels use the scalar backend. PTX containing matrix-load, MMA or
asynchronous-copy instructions selects the restricted SM80 tensor backend.
Those operations have additional layout and synchronization contracts; the
presence of an MMA instruction is not enough to make arbitrary tensor PTX safe.

The scalar subset handles representative nvcc pointer-increment/grid-stride,
nested-loop and dot-product patterns, loop-carried 64-bit pointers, common
casts/bit operations, FP32 math/conversions and integer division/remainder.
Precise recovered arithmetic forms live in
[`ptxas_fragments.json`](../../../openassembler/backend/ptxas_fragments.json).
Scalar f16/f64 arithmetic and general vector/memory-order forms remain
unsupported. Later dossiers add bounded [atomic ordering](../../compiler/gaps-atomics.md),
[scalar spilling](../../compiler/gaps-spilling.md) and
[full-warp shuffle modes](../../compiler/gaps-collectives.md); their exact
type/scope/participation/resource boundaries remain explicit. Arbitrary
shuffle/vote forms are unsupported. The emulator's coverage is broader than
what the compiler can generate.

The [floating literal repair](../../compiler/gaps-wide.md#decimal-and-raw-floating-literal-contexts)
adds decimal/exponent representation in binary64 followed by use conversion,
with separately observed raw NaN/width behavior and invalid-context rejection.
Independent rational references, toolkit contexts and A100 complete-byte gates
precede promotion. The later [typed-expression extension](../../compiler/gaps-wide.md#typed-scalar-operand-constant-expressions)
adds bounded scalar integer/floating operands with independent intermediate
rounding and explicit unresolved type/exception diagnostics. General constant
expressions across declarations/addresses and wider arithmetic remain open.

The [shared scalar memory extension](../../compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory)
adds byte/halfword/64-bit loads/stores with exact sign/neighbor/guard semantics.
Four narrow-store templates come from actual oracle words and 12,096 varied
disassemblies. A failed A100 exchange revealed a barrier issue-gap dependency;
the final repaired 1,216-launch gate precedes production admission. Dynamic
shared alignment, initialization and race freedom remain caller contracts.

The [typed Python shared-array extension](../../compiler/gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
adds i64/u64 storage, mixed padding and eight-byte index scaling. A hardware
conditional-writer failure led to a separate bounded CTA convergence pass.
The expanded gate passes 1,280 A100 launches; cyclic control flow and general
shared-memory safety remain open.

The [direct scalar device-function route](../../compiler/gaps-device-functions.md)
resolves module-local calls through static specialization, preserving lexical
slots, copied arguments, returns and local resources. The final original-source
A100 gate passes 960 launches, including ordinary CUDA noinline input. Native
call stacks, aggregate/register ABI and cross-module linking remain open.

The [explicit source-library extension](../../compiler/gaps-device-functions.md#explicit-ptx-and-cuda-source-libraries)
adds deterministic scalar PTX/CUDA symbol resolution and per-unit dependency
identity through normal CLI and remote builds. Its 576-launch A100 gate keeps
original separately linked controls distinct from merged-source controls.
Native objects/archives, dynamic stacks and complete host linking remain open.

## Validation and attribution

`compile --verify` compares a functional SASS emulator with a PTX reference
for the modeled subset. It does not run hardware or prove every memory race,
floating behavior or instruction timing. Some tensor forms are unmodeled and
must not be reported as fully verified by this flag.
For standalone compilation, exit 0 means compilation/requested checks succeeded,
exit 1 means error/mismatch, and exit 2 means the checker could not model the
kernel. An unmodeled result is not a correctness pass.

`run` executes OA and retained same-PTX ptxas binaries on the same inputs.
It normally compares bytes. Special generated contracts provide correct launch
information and explicitly labelled smoke inputs. Separate numerical tests
use independent formulas, full outputs, guards and hardware timing. That
distinction prevents an equal wrong result or a bad launch guess from becoming
a performance claim.

## Reproduce

Requirements: Python 3.12+, CUDA 12.8 `ptxas`/`nvdisasm`; compilation and
functional verification need no GPU. Run from the repository root:

```sh
compile examples/div_by_const.ptx --verify --out-dir output_ptx
run output_ptx --backend modal --budget 300 --job-timeout 60
gemm --m 128 --n 128 --k 128 --dtype f16 --out-dir output_gemm
compile output_gemm/.build/generated/gemm.ptx --out-dir output_recompiled
.venv/bin/python -m pytest -q tests/test_backend.py tests/test_backend_ops.py \
  tests/test_backend_loops.py tests/test_backend_tensor.py tests/test_cli_compile.py
```

Generated GEMM/rowwise annotations retain inferred launches only when the source
body exactly matches the built-in generator. Edited/unannotated tensor PTX
requires `run --spec`. Never assume the generic one-dimensional pointer-buffer
harness supplies matrix shapes, alignment or dynamic shared memory.

## Source and evidence

- [PTX parser](../../../openassembler/ptx/parse.py),
  [build/backend selection](../../../openassembler/build.py),
  [scalar backend](../../../openassembler/backend/ra.py),
  [tensor backend](../../../openassembler/backend/tensor.py).
- [Functional verification](../../../openassembler/verify.py),
  [GPU runner](../../../openassembler/runner.py),
  [integration record](../../evidence/integration-20261104.json).
- [Current direct-pointer policy](direct-pointer-homes.md) and
  [cooperative softmax contract](cooperative-softmax.md).

The user-facing architecture target remains SM80. Other architectures,
unsupported operand forms and general device linkage need their own validation;
successful disassembly alone does not prove correct execution.

The [scalar 64-bit selection contract](../../compiler/gaps-wide.md#scalar-64-bit-conditional-selection)
now handles `selp.b64/u64/s64/f64` with existing scalar SEL words. The original
CUDA noinline workload and 36 legal typed PTX sources pass 1,480 A100 launches,
complete bit/guard/input gates and 148 native comparisons. FP64 arithmetic,
vector/16-bit selection and broader G11 remain open.

The [low64 integer multiplication contract](../../compiler/gaps-wide.md#low-64-bit-integer-multiplication)
now supports PTX/CUDA `mul.lo.s64/u64` and Python `i64/u64` products using four
recovered IMAD/IMAD.HI words. Both source routes, helpers/loops/casts/guards and
register/parameter squares pass 1,560 A100 launches and 156 exact native matches.
General high/wide products, division and FP64 arithmetic remain open.
