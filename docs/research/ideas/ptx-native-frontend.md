# Native PTX input and compilation boundary

Status: implemented

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
Scalar f16/f64 arithmetic, atomics, spilling and some vector/memory-order forms
remain unsupported. Selected full-mask butterfly shuffles are now supported;
arbitrary shuffle/vote forms are not. The emulator's coverage is broader than
what the compiler can generate.

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
openassembler compile examples/div_by_const.ptx --verify --out-dir output_ptx
openassembler run output_ptx --backend modal --budget 300 --job-timeout 60
openassembler gemm --m 128 --n 128 --k 128 --dtype f16 --out-dir output_gemm
openassembler compile output_gemm/.build/generated/gemm.ptx --out-dir output_recompiled
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
