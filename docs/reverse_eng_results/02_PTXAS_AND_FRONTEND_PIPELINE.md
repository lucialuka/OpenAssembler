# ptxas and frontend pipeline

## Native compilation boundary

The native path is:

```text
PTX / CUDA C++ / typed Python @kernel
        ↓
frontend to PTX
        ↓
PTX parser and contract checks
        ↓
scalar or SM80 tensor selector
        ↓
virtual values, DCE, copy coalescing and allocation
        ↓
SASS text plus control fields
        ↓
OpenAssembler encoding
        ↓
ptxas-built ELF container + selected text replacement
        ↓
cubin, SASS listing and manifest
```

ptxas remains necessary for the reference cubin/container and for CUDA C/C++
PTX generation. The reference is built from the same PTX that OpenAssembler
receives. This keeps comparisons focused on backend code generation rather than
different frontend semantics.

## Frontend selection

Repository inspection is static. The native frontend is the default even when
`openassembler.toml` exists. TorchInductor/Triton capture is selected only with
`--frontend torch`, `--frontend triton` or `--frontend inductor`. A configured
application is therefore not silently executed during native compilation.

This boundary matters for reverse engineering: a Triton or Inductor cubin is
evidence about that frontend/runtime, not evidence that OpenAssembler generated
the kernel.

## Fallback policy

Unsupported PTX forms fail with a source-level explanation unless the caller
explicitly requests `--fallback ptxas`. Fallback kernels are labeled in the
manifest. A successful ptxas fallback is not counted as an OpenAssembler
optimization result.

## Reference implementation

The main code paths are:

* `openassembler/frontend.py` and `openassembler/pyfront/` — source to PTX.
* `openassembler/ptx/parse.py` and `openassembler/ptx/interp.py` — parsing and
  reference semantics.
* `openassembler/backend/ra.py` — scalar/tensor selection, DCE, allocation and
  lowering.
* `openassembler/backend/compile.py` and `openassembler/backend/tensor.py` —
  dependency controls and tensor scheduling.
* `openassembler/build.py` — source compilation, ptxas reference build,
  splicing, disassembly and manifest publication.

The production build now compiles independent repository sources with bounded
parallelism while preserving candidate order and output determinism. That is a
host build optimization; it does not alter generated SASS.

