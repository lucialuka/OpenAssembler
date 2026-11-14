# GPU execution, explicit launch specifications and fault isolation

Status: implemented

Compiled kernels can run locally, on Modal or through an SSH transport. Current
retained hardware evidence is strongest for A100 Modal execution; real SSH
execution remains outside the latest hardware-validation record. Implemented
transport support must not be mistaken for a newly measured backend result.

## Idea in plain language

A cubin is a GPU module, not a standalone executable. It needs buffers,
arguments, a grid of thread blocks and a driver context. The runner reads the
build manifest and launches OA-owned binaries. Explicit research comparison
plans can additionally launch same-PTX ptxas controls on identical data. A small JSON specification makes the real memory/launch contract
explicit instead of guessing matrix dimensions from pointer names.

`run` never recompiles. `cnr` compiles, then runs only products from that
invocation. `gpu run` is a separate probe-corpus harness, not the command for
arbitrary output manifests. Native model bundles have a separate saved replay
route and CPU state gate; see [native model capture](native-model-bridge.md).

## Implemented contract

The convenience auto-harness allocates FP32 pointer buffers and guesses common
scalar dimensions for a one-dimensional launch. It is suitable only for
matching simple kernels. Generated GEMM/rowwise annotations supply validated
geometry and precision; arbitrary tensor kernels require an explicit spec.

A spec names typed buffers, initialization, sequential steps, arguments and
compared outputs. Buffer types are `f32/i32/u32` (four bytes) and `f16/bf16`
(two bytes). Explicit numeric half values round to input precision, with BF16
ties to even; random half inputs are exact integers in[-3,3]. Initialization
can be random, zero or a numeric list, with remaining list elements zero.
Pointer offsets are bytes and must fit/alignment-match the buffer; vector/tensor
kernels can require stronger alignment than the basic element size.

Argument order follows the kernel ABI. Scalars may be i32/u32/u64/f32, grid and
block dimensions are positive one-to-three-dimensional values, and `smem` is
dynamic shared memory in bytes. Steps share buffers and execute sequentially.
`compare` defaults to all buffers or a nonempty explicit list. The runner opts
into required large dynamic shared memory for validated launches.

Production execution reports launch success plus analytic smoke checks where
specified; it does not claim a vendor comparison. Explicit research comparisons
check requested output spans byte for byte, which is not an independent
mathematical proof: both binaries can share a source bug. Generated rowwise
smoke adds a finite-value gate and NaN output poison; broader floating accuracy,
output guards and whole-state checks live in their research validators. Vendor
GPU fallback and replay of old Inductor bundles are disabled.

## Transport, watchdog and artifact behavior

The GPU worker is standard-library/ctypes based. Local execution needs Linux
and an NVIDIA driver. SSH stages the worker/jobs to a compatible Linux host;
Modal uses an optional authenticated client and a bounded ephemeral app. Jobs
run in worker subprocesses. A fault/hang is recorded, the worker can be killed
by its per-job watchdog, and a fresh worker resumes remaining jobs. A global
time budget reports unfinished jobs rather than claiming they passed.

The manifest and admitted native modules must accompany a production run;
SASS text alone is insufficient. Failed build entries are refused. Standalone
source changes generate stale-source warnings rather than recompiling behind
the user's back; use a new compile to test new source. Native model replay
validates its admitted graph, source and executable inventory before execution.

## Actual validation and findings

The November 4 integration record passes a four-case native batch covering
vector addition, a 64-thread cross-warp barrier, FP16 GEMM and BF16 GEMM. An
explicit BF16 spec verifies two-byte input buffers, two-dimensional launch and
64-KiB dynamic shared-memory opt-in. Installed-wheel tests confirm no experiment
imports are required. The retained historical CLI rowwise execution on A100-SXM4-40GB
passes 2/2 strict same-PTX and independent zero-input analytic smoke checks.
Current ordinary CLI execution uses OA-only launches and analytic smoke; the
[November 14 native attention replay](../../../exp/attention_half/RESULTS.md)
adds complete CPU state checks and profiler ownership.

These small timings are correctness smoke, not new performance estimates.
The detailed GEMM/reduction studies separately validate complete outputs,
guards, precision and meaningful graphs. Tests also exercise malformed specs,
argument counts/ranges, byte offsets, buffer sizes and fault/report handling.
No latest real-SSH hardware result is claimed.

## Reproduce

From the repository root, install the Python 3.12+ compiler and CUDA 12.8 tools;
local/SSH hardware needs a compatible driver. The optional Modal client may
live in a separate environment.

```sh
openassembler compile examples/vector_add.py --verify --out-dir output_vector
openassembler run output_vector --spec examples/run_spec.json --backend modal --dry-run
openassembler run output_vector --spec examples/run_spec.json --backend modal \
  --budget 60 --job-timeout 20
# On a Linux GPU machine:
openassembler run output_vector --spec examples/run_spec.json --backend local
# Replace this placeholder with a configured Linux GPU host:
openassembler run output_vector --spec examples/run_spec.json --backend user@gpu-host
.venv/bin/python -m pytest -q tests/test_runner.py tests/test_cli_gemm.py tests/test_cli_rowwise.py
```

The example spec supplies three 128-element FP32 buffers and the vector-add ABI;
it is not interchangeable with an unrelated kernel's launch. `--dry-run` does
not touch a GPU. Full syntax/schema, output selection and report options are
maintained in [COMMANDS.md](../../COMMANDS.md).

## Source and evidence

- [Manifest/spec planning and comparison](../../../openassembler/runner.py),
  [driver and typed initialization](../../../openassembler/gpu/cuda_driver.py),
  [batch isolation](../../../openassembler/gpu/batch.py),
  [Modal transport](../../../openassembler/gpu/modal_app.py).
- [Concrete vector spec](../../../examples/run_spec.json),
  [integration record](../../evidence/integration-20261104.json),
  [latest ordinary CLI smoke](../../../exp/weaknesses/cli_rowwise_smoke.json).

Arbitrary kernels still need correct user-supplied semantics and resource
contracts. Other architectures, real SSH validation and broader concurrency/
fault scenarios require their own evidence; a valid JSON file cannot prove a
kernel's memory accesses safe.
