# Compilation and execution pipelines

The routes for typed Python kernels, PTX, CUDA C/C++, model graphs and generated
kernels are distinct. A stage shown for one route may not appear in another.

## Standalone source

    PTX source --------------------------+
    Typed Python -> Python subset -> PTX +-> parse/link -> OA selection,allocation, 
    CUDA C/C++ -> nvcc emits PTX -------+           |       scheduling and encoding
                                                    |
                                                    v
    A100 execution <- CUDA driver <- cubin with OA executable text

The pyfront frontend statically parses supported Python syntax and emits PTX;
it does not run the kernel function as Python. CUDA C/C++ uses nvcc only to
produce PTX. OpenAssembler owns native instruction selection, register
allocation, scheduling, and SM80 word encoding.

The build uses ptxas to create an ELF/cubin container with required metadata and
a same-source reference artifact. OA replaces executable sections, updates
metadata and validates the result. The donor's executable instructions are not
the production kernel. nvdisasm can list the final cubin. See
[build.py](../../openassembler/build.py), [cubin.py](../../openassembler/container/cubin.py)
and [native_ownership.py](../../openassembler/native_ownership.py).

## Python model graph

    Repository/config -> model discovery -> CPU PyTorch FX capture
       -> supported graph and backward lowering -> OA kernel build
       -> saved plan, weights, cubins and ownership ledger
       -> runtime binds saved modules -> optional CPU-reference verification

The model frontend captures a CPU module at declared example shapes and lowers
admitted nodes to generated OA kernels. It is not arbitrary Python compilation:
unsupported FX nodes, dtypes, attributes or shapes reject. Replay loads the
saved bundle and does not generate GPU code. See [graph capture](../../openassembler/modelfront/graph.py),
[model frontend](../../openassembler/modelfront/frontend.py), and the
[native model contract](../research/ideas/native-model-bridge.md).

## Generated kernels and transport

gemm and rowwise commands generate kernels from bounded contracts. Exact
installed policy tables select admitted cases; a generator policy is not
automatic substitution into an arbitrary model graph. See
[TENSOR_GEMM.md](../TENSOR_GEMM.md), [commands](../COMMANDS.md) and
[current status](../STATUS.md).

Local, SSH and Modal choose where workers run. Transport does not change the
admitted frontend, backend or numerical policy. Standalone emulator
verification is distinct from GPU execution. Native model verification can
compare outputs, gradients and state to a CPU reference.

## Stage ownership

| Stage | Owner | Boundary |
|---|---|---|
| CLI and repository discovery | cli.py, repository.py | Finds candidates; does not prove full compilation support |
| Python kernel frontend | pyfront/ | Parses the declared subset and emits PTX |
| CUDA C/C++ frontend | nvcc via cppfront.py | Emits PTX; does not rewrite host applications |
| PTX parsing and linking | ptx/ | Admits supported PTX; interpreter is a validation reference |
| Native selection and scheduling | backend/, opt/ | OA lowering to supported SM80 forms |
| Instruction encoding | isa/ and checked-in specs | Encodes words for the supported target |
| Cubin and ownership | container/, build.py, native_ownership.py | Packages and audits OA executable text and metadata |
| Runtime | runner.py, gpu/, modelfront/runtime.py | Loads saved modules and validates launches |
| Transport | transport.py and remote/model workers | Stages compiler and artifacts to selected workers |
