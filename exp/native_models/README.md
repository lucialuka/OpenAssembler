# Native model compiler validation

This workspace retains production native-model CLI capture and saved-replay
evidence. Source factories/configuration live in `examples/native_model` (FP32)
and `examples/native_tensor_model` (explicit FP16 RNE matrix operands with FP32
accumulation/master state). Both compile forward, backward, MSE and SGD through
OA; deployment never invokes Inductor, Triton or vendor compute references.

```sh
.venv/bin/python -m openassembler cnr examples/native_model --backend modal --verify
.venv/bin/python -m openassembler cnr examples/native_tensor_model --backend modal --verify
.venv/bin/python -m openassembler run examples/native_model --backend user@a100 --verify-application
```

`runs/` retains separate build verification and replay results with actual GPU
metadata, full field checks and native entry/text hashes. Each iteration changes
inputs and target. Later complete checks poison reusable scratch buffers and
read every workspace guard. `manifest.json` records copied evidence hashes.
The same worker module is transported to Linux/SSH; live SSH hardware is not
claimed unless a corresponding result exists. No SSH GPU destination was
supplied for these recorded runs.

The CUDA launch ledger records native functions, module/text hashes and launch
contracts at the driver call. This is runtime artifact custody; independent
profiler/CUPTI event matching remains a separate validation layer. PyTorch CPU
references are outside timed workload GPU arithmetic. CPU reference and runtime
precision modes are fixed by configuration; strict failures remain failures.

These records establish bounded end-to-end native compilation and numerical
correctness. Host step times include synchronization and startup; they are not
matched performance claims. The scalar FP32 baseline is deliberately simple.
Tensor recipe performance comparisons, including remaining ptxas gaps, live in
`exp/native_model_tensor`. General ML repository coverage, broader operator
families, whole-transformer training and faster-than-NVIDIA performance require
their own completed contracts and measurements.

## Final native CLI cohort

[The retained four-repository cohort](runs/cli-20261112-final/) includes both
build verification and independently transported saved replay for FP32 residual
MLP, explicit Tensor Core MLP, attention/LayerNorm/GELU and mixed-source discovery.
The two stages each pass 33, 33, 93 and 33 full fields respectively. Profiler
compute names and counts match each actual OA launch ledger; all guard checks
pass and replay reports zero GPU code generation and zero vendor compute.
Actual GPU variants remain recorded per build/replay; absolute timings are not
pooled across these workers. The mixed repository's four independent native
translation units compile through OA but are not inferred custom model operators.

## Paired storage-view inference measurement

[benchmark_views.py](benchmark_views.py) compares complete resident-input FP32
attention/LayerNorm/GELU inference with materialized copies versus immutable
storage views. Three changed-input proofs check full CPU outputs, bitwise
paired OA equality, immutable inputs/weights and poisoned guarded workspaces.
Independent profiler counts cover both implementations. Three paired rounds
each contain 15 shuffled samples of 16-invocation CUDA graphs, with event timing
divided by 16. Capture, compilation, allocation and DMA are outside this scope.

- [First independent A100 worker](runs/views-20261112T034440.402045Z.json):
  copy/view median time ratios 1.7342, 1.7299 and 1.7299.
- [Fresh worker repeat](runs/views-20261112T034618.432670Z.json):
  ratios 1.7121, 1.7110 and 1.7120.

Both workers are A100-SXM4-40GB. Forward launches fall from 42 to 20 and private
allocations from 105 to 61; 44 captured copies alias safely across forward and
backward. This is an OA-versus-OA inference speedup, not a ptxas/cuBLAS comparison
or a training-step timing result.

## Guarded allocation-aware forward/backward planning

[validate_storage.py](validate_storage.py) validates exact-size lifetime slots
with a separate model per worker. Four changed-input MSE/backward/SGD updates
compare every output, loss, input gradient, parameter gradient and updated
parameter to the CPU reference and compare the two OA implementations bitwise.
Workspaces are poisoned between steps; guards and immutable inputs/targets are
checked. Steady-state profiler names/counts must match the complete launch ledger.

| Contract and raw record | Full field checks | Private guarded allocations | FP32 workspace payload |
|---|---:|---:|---:|
| [FP32 MLP](runs/storage-20261112T045526.304203Z.json) | 132 | 15 → 13 | 8,740 → 7,108 bytes |
| [FP32 attention/LayerNorm/GELU](runs/storage-20261112T045422.869193Z.json) | 372 | 61 → 32 | 26,756 → 12,308 bytes |
| [Explicit Tensor Core MLP](runs/storage-20261112T044955.146120Z.json) | 132 | 27 → 24 | 1,181,188 → 853,508 bytes |

All three workers report A100-SXM4-40GB. Payload includes dedicated MSE helpers
but excludes guard bytes and external parameters/inputs; the Tensor Core row
reports FP32 payload only, with FP16 packing allocations retained separately.
The pass changes storage assignments, not arithmetic or launch order. It is
opt-in through the model configuration or Python API. The attention example
enables it, and [normal CLI build and separate replay](runs/cli-storage-20261112/)
each pass 93 full fields with 32 guarded allocations and exact profiler audits.
These records establish memory reduction and correctness, not a timing gain.

Reproduce each isolated case:

    PYTHONPATH=. modal run exp/native_models/validate_storage.py --case native_model
    PYTHONPATH=. modal run exp/native_models/validate_storage.py --case native_attention_model
    PYTHONPATH=. modal run exp/native_models/validate_storage.py --case native_tensor_model

[The admission review](runs/storage-admission.json) distinguishes admitted
records from all retained failures and superseded harness runs. The combined
process's 636 numeric fields pass, but two packing activities are absent from
its exact profiler audit; adding a CPU profiling region also introduced a GPU
annotation counted by the conservative audit. That combined gate remains failed.
Initial cold/warm logs remain under failures/. A source-collection loop then
overwrote the requested CLI case in three runs: those records are tensor repeats,
not separate MLP/attention evidence. They are not admitted. The corrected harness
asserts the returned model identity and freezes its own source before dispatch.

## Regression scope

[Retained local logs](runs/regression-20261112/) report 926 passed and two
optional Torch modules skipped in the full suite; 246 passed and one skipped
in the final focused compiler/runtime/dispatch/repository checks; and 49 passed
in the separate Torch-enabled CPU graph/numerics interpreter. The counts
overlap by scope and are not added together. Documentation and whitespace
checks also pass. No live SSH host was available for a hardware run; transport
tests check that both compile and replay ship native contract data, while the
deployment archive admits only OA executable modules.
