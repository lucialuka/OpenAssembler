# CUDA C/C++ frontend through nvcc PTX

Status: implemented

CUDA source is parsed and optimized by NVIDIA's nvcc;
its emitted PTX is compiled to SASS by OpenAssembler. OpenAssembler does not
claim to have reverse-engineered or replaced nvcc's C++ parser/NVVM optimizer.

## Idea and implementation

Reuse the established CUDA language frontend while replacing the final GPU
code generator. This makes existing `__global__` kernels accessible without
requiring a new C++ implementation. The source is passed to `nvcc --ptx -x cu`;
the explicit CUDA language flag also supports recognized kernel files ending
in `.cpp`, `.cc`, `.cxx` or `.cuh`. Headers and include directories remain real
compiler dependencies.

The file's own directory is placed on the include path. Extra include paths
and nvcc flags are explicit CLI options. Use `extern "C"` for predictable
entry names, or select the exact mangled entry reported in PTX. The frontend
does not guess C++ demangling or instantiate unused templates for the caller.

After PTX generation, OpenAssembler selects and encodes native SASS. NVIDIA
ptxas still supplies the ELF container/reference, and nvdisasm reads the result.
This downstream split is the same as the [native PTX route](ptx-native-frontend.md).

## Methodology and limits

Tests cover invocation, include/flag handling, entry selection, diagnostics
and artifact routing. The supported language surface is bounded downstream:
nvcc accepting source does not mean the native backend supports every PTX
operation or device-linking requirement. Unsupported backend operations fail
with diagnostics. The retired ptxas fallback flag rejects; production executes
only OA-owned instructions.

On macOS, the Linux toolkit runs through the repository's Colima wrapper.
Files outside the mounted home tree may be staged, and their includes must
remain reachable. This is a tool-access constraint, not a CUDA semantic change.

## Reproduce

Requirements: Python 3.12+, CUDA 12.8 nvcc/ptxas/nvdisasm and nvcc's compatible
host compiler. No GPU is required merely to compile.

```sh
openassembler compile examples/vector_add.cu --verify --out-dir output_cuda
openassembler compile examples/vector_add.cu -I examples \
  --nvcc-flag=-std=c++17 --out-dir output_cuda_flags
openassembler run output_cuda --backend modal --budget 300 --job-timeout 60
.venv/bin/python -m pytest -q tests/test_cppfront.py tests/test_cli_compile.py
```

Actual execution requires a CUDA driver/GPU or authenticated remote backend.
Use `run --spec` when a kernel's actual buffers/launch differ from the simple
auto-harness. Full command syntax is in [COMMANDS.md](../../COMMANDS.md).

## Source and evidence

- [CUDA frontend and exact invocation](../../../openassembler/cppfront.py),
  [language discovery](../../../openassembler/frontend.py),
  [native build pipeline](../../../openassembler/build.py).
- [Frontend tests](../../../tests/test_cppfront.py),
  [CLI tests](../../../tests/test_cli_compile.py),
  [recorded integration](../../evidence/integration-20261104.json).

Arbitrary host C++ programs are not transformed into standalone GPU executables.
Repository packaging is described separately in [repository builds](repository-builds.md).
