# Compile a supported Python model

This walkthrough uses examples/native_model, with an explicit model
configuration and a small FP32 PyTorch module. Its current example contract is
three MSE/SGD training steps at input shape 17 by 8. OpenAssembler captures the
CPU graph at that shape and emits OA-owned GPU modules for the supported graph.

## Prerequisites

- Install OpenAssembler and authenticate Modal with modal setup, or configure
  an SSH A100 as described in [Modal and SSH setup](modal-setup.md).
- With Modal, the workstation does not need PyTorch, CUDA tools or a GPU. The
  configured remote environment supplies capture and execution.
- For local capture or execution, see [host requirements](../DEVELOPMENT.md#toolkit-and-host-requirements).

The documented route uses source and CLI contracts present at repository
revision 0ca89bd. A complete remote build and run require the external A100
service; this is not represented as a live hardware run from this workstation.

## Compile and verify

From the repository root:

    openassembler compile examples/native_model --backend modal --verify

The model verification option compares native results with a CPU reference for
the configured case. This checks that contract; it does not prove support for
other shapes or arbitrary models.

The output is written to examples/native_model/output/. Inspect its manifest,
plan, cubin/ and sass/ directories, and model build report. Together, these
record the selected graph, generated modules, launch metadata and verification
result.

## Replay the saved model

    openassembler run examples/native_model --backend modal --verify-application

Replay loads the saved bundle; it does not compile edited sources. Compile
again after changing the model. The verification option checks saved native
execution against the CPU reference and state/guard checks in the
[native model contract](../research/ideas/native-model-bridge.md).

To assess another repository, first run openassembler inspect PATH --json,
then compare graph, shapes, dtypes and execution mode with
[supported model operations](../reference/supported-operators.md). Discovery
can succeed even when a graph operation is unsupported.
