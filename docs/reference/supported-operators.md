# Native Python model operation reference

Model capture admits specific FX nodes and combinations of dtype, shape,
layout, attributes and execution mode. The linked contracts define corner cases
and validation; this table is an orientation index.

| Operation family | Bounded contract | Contract details |
|---|---|---|
| Linear and elementwise | FP32 baseline; ReLU, sigmoid, residual add/multiply and broadcast arithmetic | [Model bridge](../research/ideas/native-model-bridge.md), [operator contracts](../compiler/gaps-operators.md) |
| Convolution | FP32 NCHW/OIHW Conv2d with groups=1, optional bias and static stride/padding/dilation. Forward, first-order gradients and SGD training/replay validated. | [Convolution contract](../compiler/gaps-convolution.md), [CNN example](../../examples/native_cnn/model.py) |
| Reductions and normalization | Static FP32 sum/mean axes; last-axis softmax; affine LayerNorm; tanh GELU | [Operator contracts](../compiler/gaps-operators.md), [LayerNorm](../research/ideas/cooperative-layernorm.md), [softmax](../research/ideas/cooperative-softmax.md) |
| Matrix products | Rank 2–4 matmul with compatible batch-axis broadcasting; bmm requires equal batches. FP16-RNE operands with FP32 accumulation have separate shape and packing bounds. | [Batch-broadcast contract](../compiler/gaps-model.md), [FP16 batch products](../../exp/attention_half/RESULTS.md) |
| Shape and layout | Reshape/view/flatten, rank 2–4 permutation and concatenation under captured shape/layout contracts | [Model bridge](../research/ideas/native-model-bridge.md), [model semantics](../compiler/model-semantics.md) |
| Embedding and indices | Dense FP32 embedding table with int64 indices and checked index bounds | [Model semantics](../compiler/model-semantics.md) |
| Attention | Bounded FP32 scaled dot-product attention with Boolean broadcast or causal masking, zero dropout and supported rank-3/4 shapes | [Model semantics](../compiler/model-semantics.md) |
| Loss and updates | MSE or bounded mean cross-entropy; SGD extensions and explicit Adam/AdamW policies with stated limits | [Optimizers](../compiler/gaps-optimizers.md), [model semantics](../compiler/model-semantics.md), [checkpoints](../compiler/gaps-checkpoints.md) |

Native execution means OA-owned code. It does not mean every node is fused,
uses a dedicated kernel, or outperforms a framework. Read each contract for
forward, backward, training and replay evidence.

## Other frontends

- **Typed Python kernels:** arrays of f32/i32/u32/i64/u64/bool, supported scalar
  parameters, recognized Python statements and explicit intrinsics. See the
  [Python frontend dossier](../research/ideas/python-kernel-frontend.md) and
  [compiler gap register](../compiler/gap-register.md) for primitive, buffer,
  Boolean and ABI contracts.
- **PTX:** admission is specific to instruction forms, operands and modifiers.
  See the [compiler contracts](../compiler/README.md).
- **CUDA C/C++:** nvcc parses device source and emits PTX; admission then follows
  the same PTX/native backend contracts. See the
  [CUDA frontend dossier](../research/ideas/cuda-cpp-frontend.md).

Use [STATUS.md](../STATUS.md) for the current implementation snapshot. Research
pages can describe candidates that were not integrated; check their integration
status before treating a result as a production feature.
