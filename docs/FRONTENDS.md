# Source and execution documentation index

This short entry point preserves paths used by compiler diagnostics and older
links. The canonical feature contracts and reproduction commands are in the
linked dossiers; complete flag syntax is in [COMMANDS.md](COMMANDS.md).

## Native Python model graphs

The [native model bridge](research/ideas/native-model-bridge.md) adds CPU graph
capture, symbolic backward and OA-only launch plans for supported exact-shape
models. `[model]` selects it automatically, or use `--frontend model`/`torch`.
The former Inductor/Triton application route is disabled. Its retained dossiers
below document historical measurements and infrastructure, not current GPU
backend selection. CUDA C/C++ continues to use nvcc only as a PTX frontend.

## Python

Read [Typed Python kernel frontend](research/ideas/python-kernel-frontend.md)
for supported syntax, fixed-width arithmetic and limitations. This is a GPU
kernel subset, not general Python or PyTorch compilation.

## CUDA C/C++

Read [CUDA C/C++ through nvcc](research/ideas/cuda-cpp-frontend.md). NVIDIA parses
CUDA and emits PTX; OpenAssembler's supported native backend handles that PTX.

## PTX

Read [Native PTX input](research/ideas/ptx-native-frontend.md). Unsupported forms
fail; the production CLI does not permit vendor GPU fallback.

## Repository builds

Start with [the repository workflow guide](REPOSITORY_WORKFLOW.md) for the shared
`compile REPO`, `run REPO`, `cnr REPO` interface and local/SSH/Modal setup.

Repository builds select `[model]` automatically for a configured CPU Python
factory; `--frontend native` requests static PTX, CUDA/C++ and typed `@kernel`
discovery. Host code and build systems remain context and are not rewritten into
launch stubs. Triton/Inductor compilation is disabled.

[Discovery, manifests and publication](research/ideas/repository-builds.md)
explains what directory compilation finds, output ownership and limits.

## PyTorch training applications

[Native Python model capture and replay](research/ideas/native-model-bridge.md)
describes the supported FX nodes, autodiff, numerical modes and OA artifact
ownership. The following whole-transformer and BF16 dossiers describe historical
Inductor comparisons, not the current execution backend.

## Cooperative FP32 row reductions

Read the separate [softmax](research/ideas/cooperative-softmax.md) and
[population LayerNorm](research/ideas/cooperative-layernorm.md) dossiers. These
are explicit native generators, not automatic graph replacements.
The [rowwise geometry selector](research/ideas/shape-aware-rowwise-dispatch.md)
automatically chooses only five measured exact contracts; manual 1/2/4/8-warp
plans remain available, with four warps as the unmeasured fallback.

## Native tensor GEMM

The [`gemm` generator and options](TENSOR_GEMM.md) cover full-tile FP16/BF16
inputs with FP32 accumulation/output. Exact contract dispatch selects measured
tile, address, pointer and traversal policies; optional A100 tuning checks
admitted alternatives. This selection applies to the generated native kernel,
not arbitrary matrix products discovered inside a PyTorch application graph.

## What --verify proves

Native emulation, probe verification, standalone execution smoke and native model
state verification have different scopes. See [Native PTX verification](research/ideas/ptx-native-frontend.md),
[GPU launches and comparisons](research/ideas/gpu-execution-and-launch-specs.md),
[state diagnostics](research/ideas/state-verification-diagnostics.md), and
[COMMANDS verification semantics](COMMANDS.md).

## Shared prerequisites

Use [DEVELOPMENT.md](DEVELOPMENT.md) for installation/toolkit requirements and
[STATUS.md](STATUS.md) for the current architecture, capability matrix and latest
admitted results. This page does not duplicate per-feature reproduction.
