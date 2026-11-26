# Native model configuration

Model repositories use openassembler.toml with a [model] table. The
[repository workflow](../REPOSITORY_WORKFLOW.md) describes discovery, CLI
overrides, transport selection and saved artifacts.

## Minimum explicit contract

    [model]
    entrypoint = "model.py"
    factory = "create_model"
    input_shapes = [[17, 8]]
    math = "fp32"
    mode = "inference"

entrypoint is a Python file inside the repository; factory names a top-level
model class or function. Each shape describes one exact, positive input tensor
shape. Input dtype defaults to FP32. The factory must produce a CPU
torch.nn.Module with FP32 parameters and buffers. Capture rejects unsupported
graph operations rather than replacing them with another GPU backend.

Static discovery can infer one unambiguous model and its exact source example
shapes for forward FP32 inference. Use CLI entrypoint and input-shapes options
to resolve ambiguity for one command, or save an explicit table. See
[automatic model discovery](../REPOSITORY_WORKFLOW.md#automatic-model-discovery).

## Fields and defaults

The defaults below come from the model configuration loader. Shapes are
required unless static discovery supplies them.

| Field | Default | Contract |
|---|---|---|
| entrypoint | Required | Python source file inside the repository |
| factory | create_model | Top-level class or factory function |
| factory_args / factory_kwargs | Empty list / empty table | Finite JSON-compatible constructor values |
| input_shapes | Required | One or more exact positive tensor shapes. Product of dimensions per input is at most 2^28. |
| input_dtypes | f32 for each input | Values are f32, i64 or bool. Count must match input_shapes. |
| input_ranges | [0, 1] per input | i64 example generation uses a nonnegative half-open integer interval with upper bound at most 2^31. |
| math | fp32 | Or f16_rne_f32_accum, with its separate operand and shape contract |
| mode | inference | Or training |
| loss | mse | Or cross_entropy under the bounded model semantics contract |
| optimizer | sgd | Or adam or adamw, with only the explicitly listed policy options |
| seed | 7 | Integer seed for model and verification inputs |
| steps | 3 | Integer from 1 through 1000 |
| learning_rate | 0.01 | Finite, nonnegative value |
| rtol / atol | 1e-4 / 1e-5 | Finite, nonnegative CPU-reference tolerances |
| build_backend | auto | Local, Modal or SSH selection as described in the workflow guide |
| host | Unset | SSH destination when the selected backend requires one |
| gpu | A100-40GB | Requested Modal GPU type. The run report records returned hardware. |
| budget | 900 seconds | Per-stage budget from 30 through 1800 seconds |
| execution | direct | cuda_graph is a bounded training replay option |
| tensor_packing | separate | dual requires f16_rne_f32_accum |
| storage_views | true | Boolean runtime/storage policy |
| reuse_storage | false | Boolean runtime/storage policy |
| optimizer_fusion | false | AMSGrad requires split optimizer updates |
| sass_listings | true | Set false to omit readable SASS listings |
| profile_iterations | 1 | Integer from 1 through steps. Selects final verified iterations for profiling. |

For f32 inputs, input_ranges does not clamp random verification tensors. For
i64 inputs, it defines the half-open range used to generate example indices.
Runtime index bounds follow the compiled graph's semantic contract.

The default optimizer is SGD with zero momentum, dampening and weight decay,
and false Nesterov and maximize flags. Adam defaults to beta1 0.9, beta2 0.999,
epsilon 1e-8 and zero weight decay. AdamW uses the same betas and epsilon with
weight decay 0.01. SGD accepts momentum, dampening, weight_decay, nesterov and
maximize. Both Adam policies accept beta1, beta2, epsilon, weight_decay,
AMSGrad and maximize. Read the
[optimizer contract](../compiler/gaps-optimizers.md) for update semantics.

CUDA Graph training requires FP32 inputs, MSE and the default plain-SGD policy.
Advanced model, optimizer, FP16, caching and profiling settings have additional
interactions. Read the [full model workflow](../REPOSITORY_WORKFLOW.md#explicit-native-model-configuration),
[model bridge](../research/ideas/native-model-bridge.md), and
[model semantics](../compiler/model-semantics.md). CLI overrides are documented
in the [command reference](../COMMANDS.md).
