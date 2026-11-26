# FP32 model convolution

`nn.Conv2d`, `torch.conv2d` and `torch.nn.functional.conv2d` now compile through
CPU FX capture, native graph lowering, first-order autodiff and OA execution on
SM80/A100. The runnable CNN example combines two convolutions, ReLU, sigmoid,
spatial mean and a linear head. Its public CLI build and saved training replay
pass independent PyTorch checks. [CNN example](../../examples/native_cnn/model.py)

## Supported contract

Inputs use contiguous positive-shape FP32 NCHW storage. Weights use OIHW and an
optional FP32 bias has one element per output channel. Groups must equal one.
Stride and dilation accept positive static integers or integer pairs. Padding
accepts nonnegative static integers or pairs with zero padding mode. Shapes,
channel counts and output dimensions must agree exactly. String padding,
nonzero padding modes, grouped/depthwise and transposed convolutions remain
unsupported.

Forward computes cross-correlation. Backward gathers contributions separately
for each input or weight element. Bias gradients use the existing reduction
route. Each gradient element has one writer and deterministic serial FP32
accumulation. Parameter reuse and graph joins use existing gradient accumulation.
Convolution uses FP32 under both model math modes. The optional FP16 matrix
policy continues to apply to the separately admitted linear and matrix products.

Existing finite-input/intermediate semantics, 16,777,216-element buffer limit,
exact shapes and contiguous storage rules apply. Spatial coordinate arithmetic
must fit signed int32, including intermediate padded coordinates. This direct
implementation supplies a correctness baseline for compact CNN workloads.
Large-model throughput, Tensor Core convolution, pooling, BatchNorm, AMP and
dynamic shapes require further work.

## Use it

From the repository root, with a configured A100 backend, run

```sh
openassembler cnr examples/native_cnn --backend modal --verify
```

The example runs three SGD updates with full output, loss, input-gradient,
parameter-gradient and updated-parameter checks. Its static input shape is
`[2, 2, 8, 8]`. The model configuration also works with the existing local and
SSH backends. [Example configuration](../../examples/native_cnn/openassembler.toml)

## Research and validation

The original rejection and exact frontend snapshots establish the starting gap.
The candidate uses direct convolution with explicit bounds and separate
multiply/add operations, then receives independent CPU and hardware validation.
Experiments and reproduction scripts live in the convolution dossier.
[Research record](../../compiler_exp/gap_closure/conv2d/README.md)

The offline gate passes 144 complete tensor executions against PyTorch at all
four optimization levels through PTX, normal SASS emulation and poisoned SASS
emulation. Rectangular kernels, batch size two, stride, padding, dilation,
unchanged inputs and guards are covered. The model regression selection passes
645 tests, including 21 convolution tests.

A100-SXM4-40GB validation passes 80 full forward/gradient comparisons across nine
isolated convolution configurations and one composed CNN, with changed inputs,
poisoned workspaces and allocation guards. Ten profiler checks confirm expected
OA launches. Public CLI compilation and saved replay each pass 45 checks across
three SGD updates. Tolerances are `rtol=4e-5`, `atol=5e-6`. The maximum absolute
error in the 80 direct comparisons is `4.76837158203125e-7`.
[Hardware report](../../compiler_exp/gap_closure/conv2d/hardware-20261126T082735Z.json)

The independent audit rechecks retained full tensors, source identities, both CLI
reports and 142 exact native outputs across eleven bundles. The audit records
current compiler inputs and checks their stability throughout regeneration.
Concurrent work changed the allocator, policy proof references and the separate
FP16 tensor scheduling module. All eleven audited plans use FP32.
Current native regeneration verifies the exact executed convolution/CNN words,
while policy selections must match the retained hardware payload.
[Validation record](../../compiler_exp/gap_closure/conv2d/validation.json)

Implementation belongs to the graph capture/autodiff route, convolution kernel
emitter and CPU reference. The emitter validates saved-plan contracts before
native code generation. [Graph](../../openassembler/modelfront/graph.py),
[kernels](../../openassembler/modelfront/kernels_conv.py),
[regressions](../../tests/test_model_conv2d.py)
