# Supported inputs and boundaries

OpenAssembler is an SM80/A100 compiler with distinct frontends. Recognition by
discovery does not imply that every operation compiles. Unsupported contracts
should fail explicitly; production execution has no vendor GPU fallback.

| Input | Route | Main boundary |
|---|---|---|
| PTX | Parse and link admitted PTX; compile with the OA native backend | Instruction, modifier, operand, entry ABI and launch contracts are bounded. See [compiler contracts](../compiler/README.md) and [source and launch completeness](../compiler/source-completeness.md). |
| Typed Python @kernel | Parse supported Python syntax into PTX, then use the native backend | A GPU-kernel subset, not general Python, NumPy or PyTorch. See [frontend guide](../FRONTENDS.md) and [Python contract](../research/ideas/python-kernel-frontend.md). |
| CUDA C/C++ device source | nvcc emits PTX; OA compiles the admitted PTX | Host code, arbitrary build graphs and host launch rewriting are not translated. See [CUDA frontend](../research/ideas/cuda-cpp-frontend.md). |
| Python model repository | Static discovery or explicit [model] configuration; CPU FX capture and OA-native graph lowering | Exact positive shapes, supported graph nodes, dtypes and execution modes are required. Host training scripts and custom extensions are not compiled. See [model operation reference](../reference/supported-operators.md). |
| Generated GEMM/rowwise kernels | Dedicated OA generators with explicit shape and policy contracts | Generator policies do not automatically replace operations in arbitrary model graphs. See [commands](../COMMANDS.md) and [status](../STATUS.md). |

## Important distinctions

- The production native target is SM80. Changing the arch option does not add
  encoder or hardware support for another architecture.
- Discovery, compilation, emulator verification and hardware execution are
  separate evidence states.
- A discovered model does not imply host launch rewriting, CMake linking,
  arbitrary Python effects, or custom operator integration.
- Triton, Inductor and ptxas GPU fallback are not production routes. NVIDIA
  tools may parse CUDA or supply cubin metadata, but executable code is OA-owned.

Use the [command reference](../COMMANDS.md) for syntax, defaults, exit behavior
and outputs.
