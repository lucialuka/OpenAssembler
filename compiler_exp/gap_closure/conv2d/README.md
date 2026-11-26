# Native model convolution

## Gap and solution

G21's framework whitelist rejects ordinary Conv2d models. The original
[rejection record](baseline.json) uses the frozen frontend in `baseline/`.
The baseline observer initially missed the repository import path. Its corrected
observer loads that exact frozen frontend and records the original FX diagnostic.

The candidate adds FP32 contiguous NCHW/OIHW groups=1 convolution, optional bias,
static stride/padding/dilation, and gather-based input/weight gradients. Existing
sum-to reduction supplies bias gradients. Each output or gradient element has one
writer. Positive shapes and existing element limits remain, with explicit int32
spatial-coordinate validation. The canonical supported contract and example are
in the documentation. [Contract](../../../docs/compiler/gaps-convolution.md)

## Experiments and results

- `offline.py` passes 144 independent complete-tensor PyTorch comparisons across
  four geometries, three convolution operations, four optimization levels and
  PTX/normal/poisoned execution. All input and output guards pass.
- `candidate-tests.log` passes 21 focused capture, gradient, shape and rejection
  tests. `model-regressions.log` passes the 645-test model selection.
- `hardware-20261126T082735Z.json` passes 80 A100 forward/gradient comparisons,
  ten profiler ownership checks and both public CLI commands. Compile and saved
  replay each pass 45 complete training-state checks over three SGD updates.
- `audit.py` rechecks retained full tensors and original CLI results, then
  regenerates 142 exact native outputs across eleven bundles using current source.

The common numerical criterion is `rtol=4e-5`, `atol=5e-6`. Hardware inputs and
workspaces change between runs. External inputs and parameters remain immutable
through forward/backward. CNN training updates are checked independently against
eager CPU Torch. No throughput measurement was performed.

## Integration and reproduction

The validated implementation lives under `openassembler/modelfront/`.
`candidate-*.py` freezes its four source modules. Tests live under `tests/` and the
runnable example under `examples/native_cnn/`. The isolated wheel gate checks
packaged source identities and regenerates all eleven bundles.

```sh
python compiler_exp/gap_closure/conv2d/baseline.py
python compiler_exp/gap_closure/conv2d/offline.py
python -m pytest -q tests/test_model_conv2d.py
python -m pytest -q tests/test_model*.py
modal run compiler_exp/gap_closure/conv2d/run_modal.py
python compiler_exp/gap_closure/conv2d/audit.py
```

The Modal gate uses Python 3.12, Torch 2.7.1 and CUDA toolkit 12.8.1 on A100.
The retained report names the actual runtime. The initial candidate tests exposed
a list-versus-tuple shape check and a missing FX result binding. The corrected
implementation passes eager numerical checks. A later test counted only forward
steps when asserting gradient inventory. It now inspects the existing backward
step list as well. Numerical expectations and tolerances were preserved.

Allocator and tensor scheduling work can run concurrently in the shared checkout. All eleven bundles use FP32. The audit
requires exact source identities for the model modules, validates equal policy
selections and regenerates every current native word before accepting backend
source changes. It checks source stability over the whole audit. Arbitrary
framework coverage and the original 51-gap program remain incomplete.

The measured [installed-wheel audit](wheel-audit.json) checks package source identities and regenerates every audited FP32 model source and native word in an isolated process.
