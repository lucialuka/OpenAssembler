# Native mixed repository example

```sh
openassembler cnr examples/native_mixed_repo --backend modal --verify
openassembler run examples/native_mixed_repo --backend user@a100 --verify-application
```

The Python model is captured on CPU and compiled through the OA graph backend.
The CUDA, C/C++ and typed-Python kernel translation units are separately compiled
through OA and saved under `output/repository-native`. nvcc supplies PTX for the
C/C++ units; OA supplies their executable instructions.

These independent kernels are not graph operators and are not invoked by this
MLP. Discovery does not rewrite C++ host launch stubs, infer custom autograd rules
or silently substitute a CUDA extension into the model. All called model GPU
operations come from its own complete OA launch plan. This example tests mixed
source packaging/compilation as well as full native model execution.
