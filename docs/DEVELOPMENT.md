# Development, evidence and documentation conventions

Use [STATUS.md](STATUS.md) for architecture and current limits,
[COMMANDS.md](COMMANDS.md) for complete CLI syntax, and the
[research catalogue](research/README.md) for per-feature methods and commands.
The cross-cutting black-box compiler and GPU reverse-engineering record is in
[reverse_eng_results/](reverse_eng_results/README.md).
This page contains shared setup and maintenance rules, not a second benchmark
reproduction guide.

## Toolkit and host requirements

The production package requires **Python 3.12+**. Native compilation uses CUDA
toolkit tools; the retained measurements use **CUDA 12.8**. A GPU is needed to
execute kernels, not to compile or disassemble them.

| Task | Additional requirement |
|---|---|
| Install/import the core package | No mandatory third-party Python runtime dependency |
| Native PTX/Python-subset compilation | `ptxas` and `nvdisasm` |
| CUDA C/C++ frontend | `nvcc`, toolkit headers and a compatible host C++ compiler |
| Local execution | Linux, NVIDIA driver and a GPU matching the compiled architecture |
| macOS compilation | Colima Linux VM; keep input/output paths under the mounted home directory |
| SSH or Modal execution | Configured remote access or an authenticated optional Modal client |
| Regression tests | `pytest`; toolkit-dependent cases need the toolkit/corpus |
| Numerical/application research | PyTorch and compatible GPU packages; capture supports the pinned 2.7.1 runtime |
| Optional proof/research tools | The dependencies documented by that dossier, such as Z3 |

Install the core and test runner:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

For an end-user installation, use the Homebrew formula documented in
[HOMEBREW.md](HOMEBREW.md). It installs the CLI, compiler data, Modal client
and repository-relative resources together. The editable Python setup above
remains the development path for running tests against a checkout.

On macOS, install Colima if needed and start it. The wrapper runs NVIDIA's
Linux tools inside the VM:

```sh
colima start
env/fetch_cuda.sh
env/cuda.sh ptxas --version
env/cuda.sh nvdisasm --version
```

On Linux the wrapper can use the downloaded tools, a toolkit on `PATH`, or the
directory named by `OPENASSEMBLER_CUDA_HOME`. The download script checks the
published toolkit hashes. Neither a Colima VM nor a local GPU is a substitute
for actually executing the hardware-validation cases.

An optional Modal client can live in another Python environment. The original
workspace uses a compiler-only `.venv`, a separate PyTorch interpreter for some
CPU tests, and a separately authenticated `modal` executable. Do not assume
every one of those environments contains every dependency. Windows is not a
supported host path in the current wrapper.

The [repository workflow guide](REPOSITORY_WORKFLOW.md) gives workstation SSH,
interactive SSH-terminal and Modal commands, including remote-only compilation
without a local CUDA toolkit.

## Common validation commands

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m openassembler --help
.venv/bin/python -m openassembler status
python3 tools/check_docs.py
```

`tools/check_docs.py` is a read-only documentation check. It checks catalogue
coverage, dossier status tags, local links/anchors and the command reference
against the parser. It never dispatches a compile or GPU workload. Per-feature
hardware/proof commands belong in that feature's dossier.

The probe corpus is generated build output. If a dossier requires it, run its
documented `openassembler corpus` preparation before the relevant checks.
Do not turn an unavailable toolkit, unmodeled instruction or skipped test into
a passing hardware result.

## Evidence states

Each dossier uses one primary integration/research status:

- **`implemented`**: used or explicitly available in the current code under its
  stated validated contract. An opt-in option is not necessarily a good default.
- **`validated-not-integrated`**: a bounded capability, method or experiment was
  validated, but it is not deployed as a production optimization. Validation can
  mean semantic correctness without any measured performance improvement.
- **`tested-not-selected`**: the tested candidate failed a correctness gate,
  lost performance, added no distinct result, or otherwise was not selected.
  The dossier states which reason applies. This does not prove the general idea
  mathematically invalid or impossible on all workloads.
- **`untested-or-hardware-unavailable`**: a proposal, incomplete implementation,
  source-only inference or an unavailable hardware feature. Missing evidence
  must not be mislabeled as a negative benchmark.

For every status, distinguish **implementation presence**, **correctness**,
**performance** and **scope**. Source retained as an example can still fail its
numeric gate. A ptxas-only sparse capability probe does not validate an OA sparse
compiler. A correct low-rank factorization does not establish a fast GPU kernel.

### What a result proves

1. Parsing/assembly/disassembly shows an accepted representation, not correct
   runtime behavior.
2. Functional emulation checks implemented semantics, not real scheduling or
   operand-collection timing.
3. Hardware correctness needs an identified device, stated precision/reference,
   complete outputs, boundary checks and documented tolerances.
4. A performance claim needs correctness before timing, matched work and
   precision, paired same-device comparisons, repeated samples and explicit
   measurement boundaries.
5. A whole-step claim also includes required conversions, packing, launches,
   saved activations and optimizer work. A kernel-only win is insufficient.

Same-PTX ptxas compares compiler backends. cuBLAS/GemmEx compares library
dispatches that can select different tiles/algorithms. An eager-vs-Inductor model
comparison includes graph fusion and library/runtime behavior; it is not a
direct native OA-vs-NVIDIA SASS comparison.

Speedup is `reference_time / candidate_time`. Time reduction is
`1 - candidate_time / reference_time`; 1.12× speedup is about 10.7% lower time,
not 12%. Do not pool absolute times from 40GB and 80GB devices or unrelated
precision/layout contracts. Actual worker metadata overrides the requested
cloud GPU tier. Counter-replay duration is not benchmark timing.

## How to maintain an idea dossier

The canonical list is `docs/research/catalogue.json`; its human entry point is
`docs/research/README.md`. Every idea has exactly one file under
`docs/research/ideas/`.

A dossier should contain:

1. Its title and literal `Status: <status>` tag.
2. A plain-language description of the problem and idea.
3. What was implemented or tested, with correctness and timing methods.
4. Actual findings, including negative/mixed results and narrow contracts.
5. Prerequisites, current runnable command sets and their expected outputs.
6. Links to implementation/tests, raw evidence and relevant primary references.
7. Reproduction limitations and the next unanswered question.

Use a separate dossier for a distinct feature or hypothesis, not one file for
an entire research round. Closely coupled correctness mechanisms may share one
explanation if they form one inseparable fix. When an original experimental
script was retired, say that exact rerun is unavailable; link the retained raw
record and the closest current control. Never invent a replacement command or
claim that a current CLI run recreates a historical timing.

Update the catalogue row, STATUS only if the current implementation/results
change, and COMMANDS only if CLI behavior changes. Run the documentation check.
Evidence snapshots and hardware claims require their own validation; editing
prose does not rerun them.

## Evidence custody and retired documents

Raw JSON, source/binary hashes, source snapshots and retained oracle inputs stay
under `docs/evidence/` and `exp/`. The reorganization does not delete those
records or alter measurement values.

The exact pre-catalogue narrative documents and documentation manifests are
preserved in one [compressed provenance snapshot](evidence/documentation-history/README.md).
It records each original path, byte count, SHA-256 and text. It is historical
evidence, not an alternative active user guide. The documentation migration
record maps retired pages to their replacements.

The broad `REPRODUCING.md` is now a small compatibility pointer. `FRONTENDS.md`,
`TENSOR_GEMM.md` and the old findings filenames remain short entry points where
runtime diagnostics or established links still name them. Actual contracts and
reproduction are maintained once, in their dossiers.

Generated cubins, payload caches and intermediate listings are not additional
canonical reports. Keep successful and failed research outcomes in the dossier
and durable raw evidence, rather than starting another phase report. The full
cross-cutting [whole-stack study](research/WHOLE_STACK_RESULTS.md) is retained
because its complete methodology and attribution span several features.

## Short glossary

| Term | Meaning here |
|---|---|
| PTX | NVIDIA's virtual GPU assembly; it still needs native instruction generation |
| SASS | The GPU's actual machine instructions and scheduling controls |
| cubin | CUDA's loadable binary/container, including instructions and metadata |
| warp / CTA | A warp has 32 lanes; a CTA is a CUDA thread block containing one or more warps |
| MMA / HMMA | Matrix multiply-accumulate; HMMA is the relevant tensor instruction family |
| LDSM | Matrix-shaped load from shared memory into register fragments |
| `cp.async` / LDGSTS | Asynchronous global-to-shared transfer in PTX/native SASS |
| Register fragment | The lane-owned pieces of an operand/accumulator matrix |
| Scoreboard | Hardware completion/dependency tracking used by the scheduler |
| RAW / WAR / WAW | Read-after-write, write-after-read and write-after-write dependencies |
| Stage | A buffer slot in a software data-movement/compute pipeline |
| Canary | Guard data around a buffer used to detect out-of-bounds writes |
| Bitwise / tolerance | Exact stored-bit equality versus an explicit numeric error bound |
| Model graph / CUDA graph | Captured tensor computation versus a replayable GPU launch graph |
| RCR GEMM | Here, contiguous `A[M,K]`, `B[N,K]`, FP32 `C=A@B.T` |
| Fused epilogue | Work performed while storing GEMM output instead of separate kernels |
| Admission gate | The correctness/resource/provenance requirements that must pass before a result counts |
