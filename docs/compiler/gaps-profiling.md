# Repeated kernel profiling and resource identities

Status: implemented

G48 identified that a single diagnostic invocation supplies no repeated
per-contract timing distribution. The new optional `[model] profile_iterations`
selects the final N existing verification iterations for profiling. It defaults
to one and must be an integer in `[1, steps]`. For example, `steps=4` and
`profile_iterations=3` profile iterations 1, 2 and 3 after iteration 0. No extra
forward/backward/optimizer updates are inserted. Unverified execution profiles
nothing.

## Evidence contract and implementation

The [worker](../../openassembler/modelfront/worker.py) independently audits every
profiled invocation against its actual OA launch ledger. The
[aggregator](../../openassembler/modelfront/profiling.py) groups samples by entry,
shape/operator signature, launch geometry and cubin/text hashes. Each group
retains every invocation's kernel durations and launch count, median, min/max,
interpolated p10/p90 and interquartile range. A single invocation is explicitly
marked as such even if it contains several launches. Missing durations, duplicate
invocation audits, count mismatches or changed executable identities reject.

The [runtime resource query](../../openassembler/modelfront/runtime.py) reads
loaded driver register/shared/local-memory attributes, dynamic shared memory,
module cubin bytes and per-entry executable text bytes. SASS listing size is
reported only when that optional saved listing exists; absence is `null` and
never triggers disassembly. Module bytes are shared by all entries in that module
and must not be summed as independent allocation costs. Resource inspection
executes no GPU arithmetic.

`kernel_profiles` contains these records. `profiled_iterations` names all sampled
steps; `profiler_free_iterations`, `profiler_free_host_step_ms` and
`profiler_free_cuda_stream_elapsed_ms` expose the other existing whole-step
samples separately. Inference now records stream event intervals as training
already did. The full legacy sample arrays remain labeled by their iteration
indices. Whole-step intervals include synchronization and resident dispatch;
input generation/host reference work and initial transfers stay outside them.

Kernel durations are Kineto observations with profiler overhead, under changing
inputs and training state. They are not profiler-free per-kernel performance
measurements. Preceding requested iterations are explicitly reported as warmup;
zero preceding iterations means no warmup. No stability or convergence claim is
inferred merely from a chosen repetition count. Transfers/storage activities
retain audit counts; their durations, fusion coverage, host validation breakdown,
per-pass packing attribution and separate profiler-free kernel measurements
remain open parts of G48/G49.

## Research cycle and validation

The pre-integration [candidate](../../compiler_exp/gap_closure/profiling/candidate.py)
uses independent synthetic observations to distinguish six launches from three
invocations, retain per-contract identities and verify the expected summary.
Its [record](../../compiler_exp/gap_closure/profiling/candidate-results.json) is
accounting evidence, not a GPU timing result. Production integration followed
this gate. [Regression tests](../../tests/test_model_profile_distributions.py)
cover exact accounting and rejection controls; 24 new tests and 52 related
frontend/runtime/reporting tests pass in Python 3.12.10/Torch 2.7.1.

The final A100 [worker gate](../../compiler_exp/gap_closure/profiling/hardware-20261121T124038Z.json)
passes five routes: inference, direct SGD, CUDA Graph SGD, the default single
profile, and unverified saved execution. Three repeated profiles each retain
3/16/16 kernel contracts for inference/direct/graph respectively. There are
108 independently audited launches across the verified routes and 96 complete
CPU-reference checks over four requested iterations per route. Every loaded
function reports SM80 resources and saved binary sizes; no-SASS builds report
null listing sizes. The full reports and saved artifacts live in the adjacent
[archive](../../compiler_exp/gap_closure/profiling/hardware-20261121T124038Z.zip).
The [current native-code audit](../../compiler_exp/gap_closure/integration_20261121/hardware-continuity-20261121T125602Z.json) binds these saved kernels to the final backend after its immediate-operand repair, with all executable words, controls, registers and exits unchanged. This reuses the named hardware evidence; it is not a new profiling measurement.
The actual environment is NVIDIA A100-SXM4-40GB, Python 3.12.1, Torch
2.7.1+cu126, CUDA runtime 12.6 and toolkit 12.8.1.

Reproduce from the repository root:

```sh
python -m pytest -q tests/test_model_profile_distributions.py tests/test_model_profiling.py
modal run compiler_exp/gap_closure/profiling/run_modal.py
```

For a configured model, set `steps=4` and `profile_iterations=3`, then use ordinary
`compile --verify`, `cnr --verify` or `run --verify-application`. Build and saved
run reports retain the same accounting fields. There is no named speed ratio or
whole-application performance claim in this correctness/accounting gate.
