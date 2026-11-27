# Small OA-native causal language block

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

This example captures a CPU model factory and builds OA-only A100 forward,
backward, integer cross-entropy and AdamW kernels. It exercises embedding,
causal attention, residual paths, LayerNorm and tanh GELU with FP32 master state.

```sh
compile examples/native_language_model --backend modal --verify --no-sass
run examples/native_language_model --backend modal --verify
```

Eight changing-input/label updates are compared with independent CPU PyTorch,
including all parameter gradients, weights and optimizer moments. The explicit
[configuration](openassembler.toml) is deliberately small and exact shape;
verification example ranges do not weaken runtime index bounds. This example
does not claim arbitrary language-model repository support or competitive
whole-model throughput. See the [contracts and evidence](../../docs/compiler/model-semantics.md).
