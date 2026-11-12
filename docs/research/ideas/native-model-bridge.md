# Native Python model graphs, autodiff and launch plans

Status: implemented

The native model route captures a CPU `torch.nn.Module` factory with explicit
input shapes, lowers supported FX nodes and their derivatives to an OA launch
plan, and builds executable SM80 SASS/cubins through the native backend. CUDA
driver APIs launch the saved OA modules. PyTorch supplies CPU capture/reference
calculations, tensor storage, DMA and stream access. Inductor, Triton and vendor
compute libraries are not code generation or execution fallbacks.

## Contract and ownership

The graph subset covers FP32 Linear, ReLU, same-shape add/multiply, finite
scalar arithmetic, immutable reshapes, rank-two through rank-four permutations,
matmul/batched products, last-dimension softmax, tanh GELU, affine LayerNorm,
concatenation and slice derivatives. Batched products currently require FP32
math and exact equal batch shapes; broadcasting is not inferred. The
symbolic backward graph includes input/weight/bias gradients, branch accumulation
and aliased parameters. Complete native training uses MSE and plain SGD.
Unsupported nodes report the offending FX node and fail before native execution.
This is bounded application compilation, not arbitrary Python or all ML frameworks.

The default numerical contract uses FP32 operands, accumulation, output and
master state. It preserves explicit multiply/add ordering in its baseline
matrix kernels. The optional `f16_rne_f32_accum` contract rounds each matrix
operand to FP16 RNE, uses OA Tensor Core GEMMs with FP32 accumulation/output,
and prescribes straight-through cast derivatives. It requires the existing
full-tile GEMM shape contract for forward, dX and dW; it never silently falls
back to scalar FP32. CPU reference formulas independently apply the declared
rounding to every product. It differs from unmodified autocast.

Every model bundle records exact shapes, strides/layout assumptions, dtype,
numerical policy, generated source, module hashes, every executable text section,
register count and launch. The runtime validates the complete graph inventory,
binds functions from the admitted cubins, reuses allocations and rejects changes,
wrong devices or stale tapes. Runtime workspace guards and poisoned replay detect
overwrites and missing writes. Deployment archives exclude PTX and vendor
reference cubins. The same worker/runtime is used locally, on Modal and over SSH.

Capture tags contiguous reshape operations separately from explicit clones.
The storage-view pass removes eligible copy launches and allocations without
changing arithmetic or gradient order. Direct parameter-root views retain
copies so SGD cannot retroactively change an exposed forward result. The
attention CLI example retains 44 safe views and drops 44 launches per update.
Two independent A100-SXM4-40GB workers measure complete resident-input attention,
LayerNorm and GELU inference at 1.711–1.735× the speed of OA's materialized-copy
baseline across three paired rounds each. Forward launches fall from 42 to 20
and private allocations from 105 to 61. The 16-invocation CUDA graphs exclude
capture, compilation, allocation and transfers. This comparison is entirely OA
versus OA and is not a NVIDIA performance result. Raw paired records and the
driver live in [the model experiment](../../../exp/native_models/README.md).

Guarded buffer reuse is opt-in with `reuse_storage = true` in the model
configuration or `compile_model(..., reuse_storage=True)`. A deterministic
exact-size liveness pass spans forward and backward together. Every alias
extends its real storage root; exposed outputs and all returned gradients remain
live through the region's end. Inputs/parameters, Tensor Core packing workspaces
and MSE helpers retain separate ownership. Inputs and outputs of the same
launch cannot overlap. The loader re-derives the complete persisted schedule
and rejects a changed assignment or lifetime. Reuse is bound to one CUDA stream.

The attention example explicitly enables this policy. Its private graph storage
falls from 59 buffers to 30 exact-size slots; including the MSE helpers, guarded
allocations fall from 61 to 32 and FP32 payload from 26,756 to 12,308 bytes.
All supported arithmetic and launch order are unchanged. The memory reduction
is measured separately from the view-pass inference timings above. The
[storage experiment](../../../exp/native_models/README.md) retains full-state
paired comparisons, isolated profiler audits and failed combined-process traces.

`compile_model` also offers a conventional PyTorch custom-autograd adapter.
External GPU loss/gradient accumulation/optimizer code can launch PyTorch kernels;
that interoperability interface alone does not establish complete OA ownership.
Use `CompiledModel.native_forward` for reusable OA-only inference and
`CompiledModel.native_train_step` or the configured native training route for
the full OA loss/backward/update scope. Explicit external derivative seeds must
be contiguous CUDA FP32 tensors. Separate outstanding autograd invocations own
separate tapes; the lower-level reusable Runtime supports one outstanding tape.

## Repository configuration and reproduction

Adapt [the residual MLP example](../../../examples/native_model/openassembler.toml):

```toml
[model]
entrypoint = "model.py"
factory = "create_model"
input_shapes = [[17, 8]]
math = "fp32"
mode = "training"
loss = "mse"
optimizer = "sgd"
learning_rate = 0.01
steps = 3
build_backend = "modal"
```

The factory must return a CPU FP32 Module and can import project modules.
An explicit `[model]` selects the native model route in auto mode. Existing
CUDA/C/C++/PTX/typed-Python kernel files are separately compiled through OA;
unsupported native files reject the build. Host launch rewriting, general CMake
linking and arbitrary custom CUDA operators are not inferred from source discovery.

```sh
.venv/bin/python -m openassembler cnr examples/native_model --backend modal --verify
.venv/bin/python -m openassembler cnr examples/native_tensor_model --backend modal --verify
.venv/bin/python -m openassembler cnr examples/native_attention_model --backend modal --verify
.venv/bin/python -m openassembler cnr examples/native_mixed_repo --backend modal --verify
.venv/bin/python -m openassembler run examples/native_model --backend user@a100 --verify-application
.venv/bin/python -m pytest -q tests/test_model_frontend.py tests/test_model_kernels.py tests/test_model_runtime.py
python3.13 -m pytest -q tests/test_model_graph.py
python3 tools/check_docs.py
```

SSH requires a Linux A100, Python 3.12+, supported Torch and CUDA toolkit for
compilation; saved execution only needs the matching Torch storage runtime and
driver. Modal uses the pinned CUDA 12.8.1/PyTorch 2.7.1 image. This development
host has a separate Torch-enabled Python interpreter for optional CPU graph tests.

## Python use and deployment

```python
from openassembler.modelfront import compile_model, load_model
from openassembler.modelfront.frontend import export_model

compiled = compile_model(cpu_model, (cpu_example,), "model-output").cuda()
prediction = compiled.native_forward(cuda_input)
step = compiled.native_train_step(cuda_input, cuda_target, learning_rate=0.01)
compiled.close()
# Load without invoking a compiler or importing the original project:
restored = load_model("model-output").cuda()
export_model("model-output", "deployment.zip")
```

Deployment export enumerates only the plan, weights and admitted native modules;
it excludes references, PTX, caches and unrelated cubins. Including CPU source
for reference verification is explicit with `verification=True`. Configured
repository builds also bind source/configuration and all published artifact
bytes. Changed or missing required files reject before transport.

## Evidence and limitations

The initial production CLI compile and separate replay both pass a three-update
residual MLP full-output/loss/parameter-gradient/parameter gate on actual A100
workers. Subsequent replays additionally cover input gradients and poisoned
workspaces. Retained native graph evidence belongs under `exp/native_models/`;
runtime integer regression is independently recorded in
[the indexing dossier](runtime-integer-indexing.md). Tensor recipe controls are
under [the tensor experiment](../../../exp/native_model_tensor/README.md).

The API attention replay adds four complete updates with affine LayerNorm,
GELU and batched attention: 92 full checks pass against CPU Module/autograd.
A separate cached-decode proof compiles four exact past-length buckets and
feeds new-token cache outputs directly into the next bucket on A100. All 12
output/cache fields pass. Cache concatenation copies the existing prefix; this
is bounded cached execution, not an in-place KV allocator or production server.
Unknown cache shapes reject. Sources and all failures remain under
`exp/native_model_api`; the neural primitive cohort has 792 full checks against
independent formulas and explicit same-PTX controls.

These tests establish the supported native route and its correctness contract.
They do not establish faster-than-ptxas compilation, whole-model speed superiority,
general transformer training, distributed execution or production cached serving.
The FP32 baseline matrix kernels prioritize semantic coverage. Performance work
must retain complete precision-matched region costs, including packing,
reductions, updates and launch overhead, with independent paired measurements.

## Response to the independent whole-repository review

The [review response and frozen experiments](../../../exp/review_response/RESULTS.md)
reproduce the reported ownership-audit loophole and distinguish Python dispatch,
serial loss, matrix code generation and build diagnostics. Candidate experiments
were frozen under exp/ before production promotion.

Optional CUDA graph training replay preserves the OA kernel sequence and binds
input/target/parameter storage, learning rate and stream. Capture performs no
training update. The public Python method is:

    prepared = compiled.prepare_native_training(cuda_inputs, cuda_target, learning_rate=0.01)
    # Update input/target values in place on the same stream between iterations.
    step = prepared.run()
    prepared.close()

Alternatively load an exported native bundle with TrainingReplay.load from
openassembler.modelfront.replay. Returned result tensors borrow replay storage.
Pointer/layout changes, parameter replacement or external parameter mutation,
wrong streams and use after close reject. A configured training repository
selects this path with execution = "cuda_graph"; direct execution remains the
default. Both transports use the same implementation.

Two independent A100 campaigns each pass four complete changing-input updates,
bitwise against ordinary OA dispatch, with exact profiler activity counts.
Host-step gains are 6.89–8.98× for the small MLP, 11.19–15.66× for attention,
and 1.18–1.25× for the Tensor Core MLP, before changing loss kernels. These are
resident-input OA-versus-OA measurements including host submission and
synchronization; replay preparation is separate.

Ordered MSE now coalesces the squared-product loads across one warp while
retaining increasing-index FP32 additions. Exact element counts 112, 136, 513,
16384 and 65539 have repeated bitwise/canary/immutable-input and paired timing
evidence; other counts keep serial MSE. Measured kernel gains are 1.41–1.76×.
One-element and 33-element candidates do not pass speed admission and remain
serial. Optional sass_listings = false omits textual disassembly; paired
fresh-output builds retain identical native cubin bytes and mandatory ownership
validation. Listing-enabled builds remain the default.

Profiler audits classify raw Kineto activities by category; names containing
memcpy or memset receive no compute exemption. The conservative event-list API
rejects all unaccounted CUDA events. Model reports now retain worker-local
capture/planning/bundle phases, graph preparation, CUDA stream event intervals,
profiled iteration indexes and observed per-kernel duration samples. Samples
from a profiled invocation are not steady-state performance distributions.
