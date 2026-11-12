# OpenAssembler command reference

This is the single reference for the public `openassembler` CLI in the current
checkout. It is checked against `openassembler/cli.py`, the interactive shell,
and the commands' runtime dispatch. Feature dossiers in the
[research catalogue](research/README.md) explain experiments and their concrete
reproduction procedures; [STATUS.md](STATUS.md) describes the current architecture
and measured support.

See [DEVELOPMENT.md](DEVELOPMENT.md) for environment/toolkit setup and local
test prerequisites. This reference documents commands, not benchmark claims.

Examples run from the repository root. After installation, `openassembler ...`
and `python -m openassembler ...` select the same CLI. The production compiler
requires Python 3.12+. Compiling native kernels needs NVIDIA's CUDA tools
(`ptxas`, `nvdisasm`, and `nvcc` for CUDA C/C++); it does not require a GPU.
Running compiled kernels needs a Linux NVIDIA driver locally or on a remote
machine. Modal is optional and can be installed in a separate authenticated
Python environment: the CLI invokes its `modal` executable.

## Native model route and current ownership policy

A repository with `[model]` in `openassembler.toml` automatically uses the native
model frontend; `--frontend model` or `--frontend torch` explicitly selects it.
See [the native model contract](research/ideas/native-model-bridge.md) for config,
operator/numerical/shape scope and Python API. All workload GPU kernels are OA
artifacts on local, Modal and SSH workers. `--frontend triton`, `--frontend inductor`
and `--fallback ptxas` reject. Old `[application]` Inductor bundles cannot replay
through the production CLI. The legacy application sections below describe the
retained historical experiment APIs, not a supported vendor execution route.

`run` and `cnr` now launch only OA modules; ptxas donor/reference cubins remain
build diagnostics. Standalone execution reports execution success and available
analytic smoke checks, rather than claiming bitwise vendor comparison. Paired
same-PTX research tools retain explicit comparison plans. Native model `--verify`
and `run --verify-application` compare CPU references, complete gradients/state,
changed inputs and workspace guards. `--budget` bounds each worker stage;
`--job-timeout` and standalone launch selectors do not apply to model plans.

```sh
openassembler cnr examples/native_model --backend modal --verify
openassembler cnr examples/native_tensor_model --backend modal --verify
openassembler run examples/native_model --backend user@a100 --verify-application
```

## Find a command

| Command | Purpose |
|---|---|
| `openassembler` | Open the interactive command shell. |
| `openassembler --help` | List the top-level commands. |
| `openassembler --version` | Print the installed OpenAssembler version. |
| `openassembler compile SRC` | Compile discovered OA kernels or a configured native Python model graph. |
| `openassembler gemm ...` | Generate and compile a checked native A100 FP16/BF16 GEMM. |
| `openassembler rowwise ...` | Generate and compile cooperative native FP32 softmax or LayerNorm. |
| `openassembler run [PATH]` | Execute existing kernel cubins or replay a bundled OA-native model. |
| `openassembler cnr SRC` | Compile, then run the outputs built by this invocation. |
| `openassembler verify CUBIN PTX --kernel NAME` | Compare a probe-ABI cubin's emulated SASS with the PTX interpreter. |
| `openassembler corpus` | Build the instruction-probe corpus with NVIDIA's tools. |
| `openassembler check` | Compile/check the existing probe corpus with an OpenAssembler backend. |
| `openassembler bench` | Build the scalar optimization-level microbenchmark cubins. |
| `openassembler fastmm build` | Build research kernels for direct/Strassen/AlphaTensor plans. |
| `openassembler fastmm run` | Run the fast-matrix-multiplication research plans on Modal. |
| `openassembler gpu run` | Execute the probe corpus on local/SSH/Modal hardware. |
| `openassembler benchmark` | Describe or execute the CLI integration matrix. |
| `openassembler status` | Print the first summary table from `docs/STATUS.md`, or its opening text when no table exists. |

Every parser level accepts `-h` / `--help`, including
`openassembler fastmm --help`, `openassembler fastmm run --help` and
`openassembler gpu run --help`. `fastmm` requires `build` or `run`; `gpu`
requires `run`. Flags belong after their command, for example
`openassembler compile kernel.ptx --verify`, not `openassembler --verify compile ...`.

## Interactive shell

Run `openassembler` with no arguments. Enter the same commands without the
leading executable name:

```text
compile examples/div_by_const.ptx -o division --verify
rowwise --op softmax --rows 1024 --width 512 --out-dir output_softmax
run output_softmax --backend modal --dry-run
status
exit
```

| Shell input | Behavior |
|---|---|
| `help` or `?` | Show CLI command help and the shell built-ins. |
| `clear` | Clear the terminal display. |
| `exit`, `quit` or `q` | Leave the shell. Ctrl-D also leaves it. |
| `<command> --help` | Show that command's flags; for example `compile --help`. |
| Ctrl-C at the prompt | Cancel the current input and keep the shell open. |

The shell uses ordinary quoted argument parsing; it is not an operating-system
shell and does not execute pipes, redirection or arbitrary Python. When
`readline` is available it remembers up to 1,000 history entries in
`~/.openassembler_history` and completes top-level command names. The built-in
`help` accepts no command argument: use `compile --help`, not `help compile`.

## `compile`: kernel files, source trees and native model graphs

```sh
openassembler compile examples/div_by_const.ptx -o division --verify
openassembler compile examples/vector_add.py --out-dir output_python
openassembler compile examples/vector_add.cu -I include --nvcc-flag=-DTILE=32 --out-dir output_cuda
openassembler compile my_kernel_repo/ --out-dir output_repo
openassembler compile examples/native_model --frontend model --verify --backend modal
```

`SRC` is required. A standalone `.ptx` file goes to the native backend; a
supported Python kernel uses OpenAssembler's typed `@kernel` subset; CUDA
C/C++ goes through `nvcc --ptx`, then the native backend. Host-only Python/C++
code is not translated into a GPU program. Directory discovery respects Git's
tracked/non-ignored files and recognizes kernel sources; output paths mirror
the source tree. A directory build does not accept a single output name.

A `[model]` configuration selects native Python graph capture automatically.
`--frontend model` or `--frontend torch` explicitly chooses it. Supported model
factories run on CPU; OA emits executable GPU computation. `--frontend native`
selects static source discovery. Triton/Inductor compilation is disabled.

### Compilation options

These options are also accepted by `cnr`.

| Option | Default | Meaning |
|---|---|---|
| `-o NAME`, `--out NAME` | Source stem | Name one file's kernel output. Use a name, not a path; use `--out-dir` for its directory. |
| `--out-dir DIR` | `REPO/output` for directory builds; `./output` for files | Build root; use the same override with `run REPO`. |
| `--kernel NAME` | All entries | Compile only the named entry in a standalone source. |
| `--arch ARCH` | `sm_80` | Target architecture. The validated native product target is SM80/A100; accepting an architecture string is not a support guarantee. |
| `--opt-level LEVEL` | `opt` | One of `safe`, `mined`, `sched`, `opt`; see below. |
| `--verify` | Off | Check standalone semantics with the emulator/PTX checker; for a native model, compare CPU references, gradients/state and GPU ownership. |
| `--trials N` | `6` | Random standalone semantic-check trials. |
| `--fallback ptxas` | Disabled | Retained parser spelling; rejected by the production CLI. Unsupported forms need OA implementation. |
| `-I DIR`, `--include DIR` | None | Additional CUDA include directory; repeatable. The source file's directory is included; directory builds also add the discovered repository root. |
| `--nvcc-flag FLAG` | None | Additional CUDA compiler flag; repeatable. Use `--nvcc-flag=-DNAME=VALUE` when a value begins with `-`. |
| `--dump-ptx` | Off | Print generated/input PTX. |
| `--no-sass` | Off | Omit the readable `nvdisasm` listing; still produce cubins and metadata. |
| `--backend MODE` | `auto` | Shared build/run destination: `local`, `modal`, `user@host`, or `ssh` with `--host`. Native auto builds locally; models use configuration. |
| `--host USER@HOST` | None | SSH target or SSH config alias. Ports/keys/ProxyJump come from SSH configuration. |
| `--gpu TYPE` | Model config / `A100-40GB` | Modal GPU for native model capture/execution. Native remote compilation is CPU-only. |
| `--budget SECONDS` | Model config / native build `900` | Per-stage budget. Remote native compilation accepts 30–1800 seconds. |
| `--job-timeout SECONDS` | Standalone kernel `60` | Native model workers use `--budget` for their stage watchdog. |
| `--build-backend MODE` | Shared `--backend` / model config | Override only compilation, e.g. build on Modal and execute via SSH. Also accepts legacy `ssh`. |
| `--build-host USER@HOST` | Shared `--host` / model config | Override the SSH compilation target. The model bundle records its build host. |
| `--build-gpu GPU` | Model configuration | Requested Modal GPU for native model capture. Actual returned hardware is recorded separately. |
| `--build-budget SECONDS` | Shared `--budget` / configuration | Override only compilation time budget. |
| `--dry-run` | Off | Plan native compilation or model capture, without writing outputs, uploading sources or running workloads. `cnr` plans both stages. |
| `--frontend MODE` | `auto` | `auto` selects `[model]` capture when configured, otherwise static kernels. `native` forces static discovery. `model`/`torch` select OA model compilation. `triton`/`inductor` reject. |

For scalar kernels, `safe` uses conservative in-order scheduling; `mined`
uses dependency-distance stall selection; `sched` adds list scheduling; `opt`
also applies the supported bounded loop unrolling. They are compiler policies,
not promises that increasing the level always improves a workload. Native
tensor kernels use their separate validated tensor policy at every level.

Native models require SM80, a CPU FP32 factory and explicit exact input shapes.
Model compilation rejects standalone frontend/selection flags, output renaming,
`--no-sass`, nondefault optimization/trial flags and `--job-timeout`; use `--budget`
for its stage watchdog. The model configuration controls precision, seed, steps,
verification tolerances and default transport. See the native model dossier.

Native remote builds ship this compiler/data and publish artifacts locally.
`--backend` applies to both `cnr` stages; `--build-backend` overrides compilation.
Standalone native local compilation has no build watchdog and rejects explicit
build budgets; remote build and model workers have bounded subprocesses.

### Output artifacts

For a standalone source named `NAME`, the output root contains
`cubin/NAME.cubin`, `sass/NAME.sass` unless suppressed, `manifest.json`,
`.build/ptx/NAME.ptx` and `.build/reference/NAME.cubin`. Native SASS is encoded
by OpenAssembler and spliced into a ptxas-built ELF container; the reference
cubin is a separate ptxas compilation. A cubin is launched through the driver,
not executed as a host program.

Model builds add `model.json`, `plan.json`, `weights.json`, raw FP32 weight files,
`native-ledger.json`, and scalar/tensor cubin trees. `model-artifacts.json` binds
publication bytes. Model replay transports only the enumerated native modules,
plan and weights; CPU source accompanies explicit verification. Neither PTX nor
ptxas references are deployment inputs. Output publication obeys writer locks.

## `gemm`: checked native A100 tensor GEMM

```sh
openassembler gemm --m 128 --n 512 --k 512 --dtype f16 --out-dir output_auto
openassembler gemm --m 128 --n 512 --k 512 --tile 64x64 --out-dir output_tile64
openassembler gemm --m 1024 --n 4096 --k 1024 --dispatch manual --out-dir output_control
openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 --pointer-mode direct --out-dir output_gemm
openassembler gemm --m 16384 --n 4096 --k 4096 --cta-order morton8 --out-dir output_morton
openassembler gemm --m 16384 --n 4096 --k 4096 --cta-order grouped_n8 --out-dir output_grouped_n
openassembler gemm --m 1024 --n 4096 --k 1024 --fragment-schedule load_next --out-dir output_load_next
openassembler gemm --m 128 --n 512 --k 512 --tile 64x64 --copy-address-init reuse --pointer-mode direct --pointer-advance early --out-dir output_copy_reuse
openassembler gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --epilogue bias_residual_relu --out-dir output_fused
```

The operation is `C = A @ B.T`: contiguous `A[M,K]` and `B[N,K]` FP16/BF16
inputs; FP32 accumulation and contiguous `C[M,N]` output. M/N must be positive
multiples of 64 with the 64×64 tile, or of 128 with the 128×128 tile; K must
be a positive multiple of 128. Inputs need 16-byte alignment; output needs
eight-byte alignment. Arbitrary strides, tails, alpha/beta and alternate
transposed storage are not part of this command.
Each input/output element count must fit the generator's unsigned 32-bit index
contract, and the N tile count must fit CUDA's grid-y limit.

| Option | Default | Meaning |
|---|---|---|
| `--m M`, `--n N`, `--k K` | Required | Matrix dimensions. |
| `--dtype TYPE` | `f16` | Input type: `f16` or `bf16`. |
| `--arch ARCH` | `sm_80` | This checked generator accepts only `sm_80`. |
| `--dispatch MODE` | `auto` | `auto` selects an exact measured native policy when its compiler/source identity and contract match; otherwise it retains the baseline. `manual` selects the baseline plus any explicit implementation knobs. |
| `--tile SHAPE` | Automatic / `128x128` reference | Explicit `128x128` or `64x64` CTA tile. A shape divisible by 64 but not 128 requires the smaller tile. Explicit selection disables measured dispatch. |
| `--copy-address-init MODE` | Selected policy / `original` | Explicit `original` per-chunk address calculation or `reuse` derived 64-bit offsets for the 64×64 tile; candidate is included in the production A100 tuning pool. |
| `--schedule MODE` | Selected policy / `stagepair` reference | Explicit `stagepair` alternating-stage plan or `combined` comparison plan; selects manual dispatch. |
| `--fragment-schedule MODE` | Selected policy / `source` reference | Explicit `source` or `load_next`; the latter is admitted only for FP16/BF16 at 1024×4096×1024 and 4096×1024×1024 with the 128×128 tile, stagepair, legacy pointers, late advancement and no epilogue. |
| `--cta-order ORDER` | Selected policy / `m_fast` reference | Explicit `m_fast`, `morton8` or `grouped_n8`; Morton requires complete 8×8 output-tile groups, while grouped-N requires N tile count divisible by 8 and changes the launch grid to `(m_tiles×8, n_tiles/8)`. |
| `--epilogue MODE` | `none` | `none`, `bias_residual`, `bias_residual_relu`. Nonempty modes require `stagepair` and append FP32 bias/residual pointers. |
| `--pointer-mode MODE` | Selected policy / `legacy` reference | Explicit `legacy` or `direct` pointer homes in either tile family. Direct requires `stagepair`, `m_fast`, and `--epilogue none`. |
| `--pointer-advance MODE` | Selected policy / `late` reference | Explicit `late`, `early`, or bounded shape-specific `auto` placement in either tile family; requires stagepair/m-fast/unfused. The placement value `auto` uses early updates only at 1024×4096×1024; it is distinct from `--dispatch auto`, which selects a complete measured plan. |
| `--autotune BACKEND` | Off | Compile, fully check and time the admitted native policies on an A100 using `local` or `modal`. Requires automatic dispatch, no explicit policy knobs and a real build. Excludes fused epilogues. |
| `--tuning-record JSON` | None | With `--autotune`, retain the tuning result and companion binary/source/SASS archive; without it, explicitly reuse a current shape/compiler-matched record. |
| `--tune-rounds N` | `3` | Paired GPU timing rounds, bounded to 3–10. |
| `--tune-samples N` | `12` | Samples per candidate per round, bounded to 8–100. |
| `--out-dir DIR` | `output` | Output root. |
| `-o NAME`, `--out NAME` | `gemm` | Artifact basename; entry name remains `gemm`. |
| `--no-sass` | Off | Skip the readable listing. |
| `--emit-only` | Off | Write annotated generated PTX without invoking the CUDA tools. |

Both tile families use 128 threads. The 128×128 tile uses 65,536 dynamic
shared-memory bytes; the 64×64 tile uses 32,768. The smaller tile uses
stagepair, m-fast, source-order fragments and no fused epilogue. It has
selectable legacy/direct pointer homes, late/early pointer placement, and
original/reused copy-address initialization; every combination remains a
separate candidate and is compared under its exact contract. The generated
source annotation supplies the checked launch to `run`.
Custom callers must use the recorded launch and opt into the shared-memory
size. Fusion computes `(A @ B.T + Bias) + Residual`, optionally followed by
ReLU, with contiguous FP32 `Bias[N]` and `Residual[M,N]`.

Morton and grouped-N orders are logical output-tile traversals, not new MMA
instructions or promises about which SM runs each block. Every policy has
shape-specific wins and regressions. Automatic dispatch uses exact recorded
contracts; it does not extrapolate a winner to neighboring shapes or silently
switch to ptxas/cuBLAS. There is no `morton6` CLI option.

The current table has 42 exact FP16/BF16 contracts: 34 tuned policies and eight
measured baseline retentions. Twenty-two select tile64 variants, including 20
with copy-address reuse; two select Morton8. The remaining tuned entries use
128-tile pointer policies. These counts describe the current evidence, not a
promise that neighboring shapes or a future toolkit have the same fastest
policy. See the [dispatch dossier](research/ideas/shape-aware-native-gemm-dispatch.md)
and [complete measured matrix](../exp/dispatch_tuning/RESULTS.md).

To compare the current admitted native policies on a particular A100 and keep
the result for a later compile:

```sh
openassembler gemm --m 128 --n 512 --k 512 --autotune modal \
  --tuning-record output/tuning/128-512-512-f16.json --out-dir output/tuned
openassembler gemm --m 128 --n 512 --k 512 \
  --tuning-record output/tuning/128-512-512-f16.json --out-dir output/reused
```

GPU tuning includes compilation and validation overhead. Its sixteen-operation
graph timings cover resident-input isolated kernels, excluding packing, model integration
and whole training. A candidate must pass full zero/integer/normal outputs,
canaries, input immutability, poisoned graph, bitwise ptxas/baseline,
disassembly and resource gates before timing. Selection requires at least a
2% improvement in every paired round; uncertain gains keep the reference.
The selected policy is the fastest admitted repeatable candidate in that
bounded test, not a guarantee over every compiler, library, GPU or future run.

`gemm-dispatch.json` records the selected plan, reason and scope beside the
build. The source annotation and regular manifest carry its actual launch and
native code-generation contract. Imported tuning records are explicit evidence;
the CLI does not run a hidden GPU search during an ordinary compilation.

The annotated PTX is saved under `.build/generated/NAME.ptx`. Recompiling that
exact source with `compile` retains its plan/pointer policy. Edited or
unannotated PTX loses inferred launch/buffer metadata and uses the default
tensor policy; describe an actual launch with `run --spec`.

## `rowwise`: cooperative FP32 softmax and LayerNorm

```sh
openassembler rowwise --op softmax --rows 1024 --width 512 --out-dir output_softmax
openassembler rowwise --op layernorm --rows 129 --width 127 --epsilon 1e-5 --out-dir output_norm
openassembler rowwise --op softmax --rows 7 --width 33 --emit-only --out-dir generated
openassembler rowwise --op layernorm --rows 4096 --width 32 --dispatch auto --emit-only --out-dir generated_norm
openassembler rowwise --op layernorm --rows 4096 --width 32 --dispatch manual --warps-per-cta 2 --emit-only --out-dir generated_norm2
```

Both operators read/write contiguous FP32 `[rows,width]`, with four-byte-aligned
buffers. One warp cooperates on each row. The default four-warp plan uses 128
threads and `ceil(rows/4)` blocks. Five exact measured A100 contracts select
two or eight warps per CTA; explicit manual choices additionally allow one
warp. A chosen plan uses `32×warps` threads and `ceil(rows/warps)` blocks, with
no shared memory. Row and column
tails are supported. Input values and the relevant differences/sums/squares must
remain finite in FP32; NaN/Inf and overflowing intermediate arithmetic are not
covered. The entry ABI is `(u64 X, u64 Y)`; use separate input/output buffers.

| Option | Default | Meaning |
|---|---|---|
| `--op NAME` | Required | `softmax` or `layernorm`; also selects the entry name. |
| `--rows R` | Required | Positive row count below 2³²; total elements must not exceed 2³². |
| `--width W` | Required | Integer width 1–1024. |
| `--epsilon VALUE` | `1e-5` | LayerNorm epsilon. It must round to a positive, normal, finite FP32 value. The plan validates it for both operations. |
| `--arch ARCH` | `sm_80` | Only `sm_80` is supported by this checked generator. |
| `--dispatch auto|manual` | `auto` | Auto selects only an exact promoted `(op, rows, width, epsilon, dtype, layout, alignment, SM80)` table entry. Unknown, stale or body-mismatched entries fall back to four warps. `manual` keeps the four-warp baseline unless `--warps-per-cta` is supplied. |
| `--warps-per-cta 1|2|4|8` | Auto table; otherwise `4` | Explicitly choose whole warps per CTA; overrides automatic dispatch. The runner carries and validates the resulting exact launch. |
| `--out-dir DIR` | `output` | Output root. |
| `-o NAME`, `--out NAME` | Operation name | Artifact basename; entry name remains the operation name. |
| `--no-sass` | Off | Skip the readable listing. |
| `--emit-only` | Off | Write annotated PTX without the CUDA tools. |

LayerNorm uses population variance and produces normalized values only: no
affine weight/bias, saved mean/inverse deviation or backward. These explicit
generators do not automatically replace arbitrary Python loops or operators
inside captured PyTorch graphs. Tiny cases can regress; read the dossier's
timing scope before treating the command as a dispatch recommendation.

The [rowwise dispatch dossier](research/ideas/shape-aware-rowwise-dispatch.md)
lists the five entries, independent replay gates and rejected near-threshold
candidates. The CLI has no rowwise GPU-autotuning flag; its automatic route is
the installed exact table, with manual geometry available for new experiments.

## `run`: saved native kernel binaries or model replay

```sh
openassembler run output_gemm --backend modal --budget 300 --job-timeout 120
openassembler run output_softmax --backend modal --dry-run
openassembler run output_repo --source kernels/ops.ptx --kernel relu --spec launch.json --backend user@host
openassembler run examples/native_model --backend modal --verify-application
```

`PATH` accepts a source repository/file or an existing build directory.
`run REPO` reads `REPO/output/manifest.json`; custom builds use
`run REPO --out-dir DIR`. A source path restricts execution to its own saved
records. A configured model is selected through its matching model.json metadata.
It never compiles or recaptures. A saved native kernel directory uses its manifest;
bare run retains the ./output workflow. Legacy native builds are selected only
when their recorded source matches the requested repository. The repository-local
build takes priority; ambiguous single-file builds require an explicit output
directory. Legacy Inductor application bundles reject.

### Kernel and shared execution options

These flags are also accepted by `cnr`, except where its section says otherwise.

| Option | Default for standalone kernels | Meaning |
|---|---|---|
| `--backend MODE` | `auto`, resolved to `local` for kernels | `local`, `modal`, `user@host`, or `ssh` with `--host`. Native models reuse their saved build transport in auto mode. |
| `--host USER@HOST` | Saved application host when using `ssh` | Select or override the SSH destination; SSH config aliases work. |
| `--out-dir DIR` | `REPO/output` | Build root when PATH names sources; bare `run --out-dir DIR` also selects an artifact directory. |
| `--spec FILE` | None | JSON buffer/launch/pipeline specification; replaces guessed auto-harness arguments. |
| `-n N`, `--n N` | `1048576` | Elements per generic auto-harness pointer buffer; positive. Generated GEMM/rowwise contracts supply their own dimensions. |
| `--reps N` | `20` | Timed repetitions; nonnegative. `0` runs correctness without repetition timing. |
| `--gpu TYPE` | `A100-40GB` | Requested Modal GPU type; it does not change compiled architecture or prove that requested memory capacity was returned. |
| `--budget SECONDS` | `300` | Positive finite worker/batch budget. Standalone Modal jobs reject values above 1800. |
| `--job-timeout SECONDS` | `60` | Positive finite watchdog per standalone job. |
| `--dry-run` | Off | Describe jobs/launches/upload size and estimates; do not execute a GPU job. |
| `--save-jobs FILE` | None | Also write serialized standalone GPU jobs to this pickle file. |

`--budget` bounds scheduled GPU-worker work, with additional watchdog/cleanup
grace periods. It is not the complete image-build, queue, transfer and client
wall time, and a dry-run price estimate is not a measurement. Local execution
requires this machine to be Linux with an NVIDIA driver. SSH execution needs a
Linux driver host, Python and GNU `timeout`; it uploads the stdlib-only GPU
runner, not a full CUDA toolkit. Modal execution needs the authenticated
`modal` command on PATH.

### `run`-specific selection and replay flags

| Option | Default | Meaning |
|---|---|---|
| `--kernel NAME` | All selected native entries | Select a kernel. |
| `--source FILE` | All matching sources | Select its source when entry names repeat across compiled files. |
| `--from-results RES.pkl` | None | Re-report an existing GPU results pickle rather than execute jobs. The manifest/job selection must still describe that run. |
| `--application NAME` | Disabled | Retained compatibility spelling; legacy Inductor bundles reject. Cannot combine with `--kernel`, `--source`, `--spec`, `--from-results`, or `--save-jobs`. |
| `--app-arg ARG` | Disabled | Retained compatibility spelling; native model input data is passed to its native call API. A supplied list must exactly match the saved argument list; it does not specialize a new shape. Use `--app-arg=--flag` for a token beginning with `-`. |
| `--verify-application` | Off | Verify a native model replay against its CPU reference, checking outputs, gradients, parameters, workspace guards and profiler ownership. |

For native models, `--backend auto` reuses the saved transport. Override it with
`local`, `modal`, `user@host` or `ssh --host HOST`. `--verify-application` checks
saved CPU references and full native state. Standalone flags `--application` and
`--app-arg` are retired vendor compatibility spellings and reject. `--reps`,
`-n`, `--spec`, `--save-jobs`, `--from-results` and `--job-timeout` also do not
apply to native models; configure steps in `[model]` and use `--budget`.

### What the native auto-harness verifies

Generic pointer arguments are guessed as FP32 buffers of N elements. Integer
arguments named `n/count/size/len/num/length` receive N; other integers receive
16. Other scalar arguments get deterministic test values. The default launch
is one-dimensional, block 256, grid `ceil(N/256)`. This is a convenience for
simple elementwise kernels, not a valid launch for arbitrary user code.

A checked generated GEMM gets its exact declared buffers, tile launch and shared memory, using exact small integers in this smoke test. A checked
rowwise kernel gets zero input and NaN-poisoned output; every result must match
the analytic constant and remain finite. These smoke
checks do not replace full nonzero numerical/canary/paired-performance tests.

Standalone `run` checks successful OA execution, requested output sizes and any
specified analytic smoke conditions. It does not launch a vendor reference or
claim general accuracy from arbitrary input values. Faults, missing outputs and
skipped cases yield nonzero status. Results are saved in `OUTDIR/run.json` and
`OUTDIR/runs/*.json`. Timing covers the specified native launch sequence.

### Describe an actual launch with `--spec`

Save this as `launch.json`, after compiling the matching vector-add example:

```json
{
  "buffers": {
    "a": {"count": 4096, "dtype": "f32", "init": "random"},
    "b": {"count": 4096, "dtype": "f32", "init": "random"},
    "out": {"count": 4096, "dtype": "f32", "init": "zeros"}
  },
  "steps": [{
    "kernel": "vector_add",
    "grid": 16,
    "block": 256,
    "smem": 0,
    "args": [{"buf": "a"}, {"buf": "b"}, {"buf": "out"}, {"i32": 4096}]
  }],
  "compare": ["out"],
  "reps": 20
}
```

Check the actual compiled entry name/argument order in the manifest; use the
repository's `examples/run_spec.json` when running `examples/vector_add.py`.
Buffer `count` is an element count. Types are `f32`, `i32`, `u32`, `f16`, `bf16`;
initialization is `random`, `zeros`, or an explicit numeric list. FP16/BF16
random inputs are exact small integers. An explicit list initializes that
prefix of the buffer; remaining elements stay zero. It does not repeat.
`grid` and `block` may be positive integers or lists of
one to three dimensions. Shared bytes are given by `smem`. Each step may also
specify `source` to disambiguate an entry. Arguments use `buf` plus optional
byte `offset`, or one scalar key: `i32`, `u32`, `f32`, `u64`. Offsets must be
aligned and inside their buffer. Multiple steps describe a pipeline; `compare`
defaults to all buffers. Spec-level `reps` overrides the CLI repetition count.

## `cnr`: compile and run OA artifacts

```sh
openassembler cnr examples/vector_add.py --spec examples/run_spec.json --backend modal --out-dir output_cnr
openassembler cnr examples/vector_add.cu --backend user@host --out-dir output_cnr
openassembler cnr examples/div_by_const.ptx --verify --dry-run --out-dir output_cnr
openassembler cnr examples/native_model --frontend model --verify --backend modal
```

`SRC` is required. `cnr` accepts the compilation-option table and the shared
execution-option table. Its `--kernel` restricts compilation; it runs only the
source/entry pairs successfully built by this invocation, not unrelated stale
manifest entries. It does **not** expose `run`-only `--source`, `--from-results`,
`--application`, `--app-arg`, or `--verify-application` flags.

The complete option spellings are also valid with the command prefix:
`openassembler compile --frontend native` and
`openassembler cnr --frontend torch`.

`--frontend native` selects static kernel discovery and OpenAssembler compilation;
`--frontend model`/`torch` selects the configured native model route.
`--backend` selects the transport for either frontend.
`--build-backend` optionally separates those locations. `--dry-run` plans both
stages without builds, uploads, output writes or GPU jobs, including for native
sources. `--verify` also requests native model replay verification. Application
`cnr` rejects `--spec`, `--save-jobs` and nondefault `-n`/`--reps`. Both native
and application workflows accept SSH target strings directly. Each stage has
its own budget; a `cnr` budget is not a combined end-to-end wall-time cap.

## `verify`: a compiled probe under emulation

```sh
openassembler verify corpus/sm_80/cubin/ENTRY.cubin corpus/sm_80/ptx/ENTRY.ptx --kernel ENTRY --trials 6
```

Both positional paths and `--kernel NAME` are required. `--trials N` defaults
to 6. This command disassembles an existing cubin, emulates it with normal and
poisoned registers, and compares with the PTX interpreter. It requires the
probe corpus's fixed `(out, in, k, kf)` signature; it is not a generic runner
for arbitrary GEMM/application ABIs. For general supported kernel signatures,
use `compile --verify`; for real hardware execution, use `run` or `gpu run`.
Approximate PTX arithmetic is interpreted under the checker's supported
approximation rules. An emulator pass alone is not hardware timing validation.

## Corpus, checker and scalar benchmark builds

```sh
openassembler corpus --arch sm_80
openassembler check --compiler ra --only div --trials 6
openassembler bench --arch sm_80
```

| Command | Options and defaults | Result |
|---|---|---|
| `corpus` | `--arch sm_80` | Build instruction probes with ptxas and disassemble them under `corpus/ARCH/`; wraps `tools/build_corpus.py`. |
| `check` | `--compiler ra` (`ra` or `naive`), `--only ""`, `--trials 6` | Compile and validate the existing SM80 probe corpus; `--only` is a probe-name substring. The register-allocating backend is `ra`; `naive` is the small reference compiler. |
| `bench` | `--arch sm_80` | Build scalar microbenchmark cubins at the optimization levels, checking emulated outputs; wraps `tools/bench_build.py`. It builds, rather than measures, these microbenchmarks. |

All three commands need the toolkit. `check` requires a previously built
`corpus/sm_80/ptx/`. Neither `corpus` nor `bench` silently establishes support
for every architecture string supplied. GPU execution of probes uses the next
command. The public `gpu run` selector is limited to the probe `ptxas`/`ours`
directories; specialized benchmark-directory execution remains a developer
script workflow rather than an extra undocumented public selector.

`check` clears its chosen generated `ours` or `ours_naive` output directory
before rebuilding, even with `--only`; a filtered check does not preserve the
other previously built native probe cubins in that directory.

## `gpu run`: real-hardware probe validation

```sh
openassembler gpu run --which ptxas --only add --backend local --trials 4
openassembler gpu run --which ours --only div --backend user@host --budget 300
openassembler gpu run --which ours --backend modal --gpu A100-40GB --dry-run
```

| Option | Default | Meaning |
|---|---|---|
| `--backend MODE` | `local` | `local`, `modal`, or `user@host`. This differs from application transport selection. |
| `--which COMPILER` | `ptxas` | `ptxas` or `ours`: choose the already-built probe cubins. Build `ours` with `check` first. |
| `--only FILTER` | Empty/all | Probe-name substring, or a comma-separated list of exact probe names. |
| `--trials N` | `4` | Random hardware trials per probe. |
| `--budget SECONDS` | `300` | Total worker budget. |
| `--gpu TYPE` | `A100-40GB` | Modal GPU request; irrelevant to local/SSH execution. |
| `--dry-run` | Off | Construct/report the job selection without executing a GPU workload. |

This command targets the SM80 probe corpus. It compares hardware with supported
emulator/PTX semantics and writes corpus GPU evidence. `run OUTDIR` instead
uses a user build manifest and its ptxas control. The two commands are not
interchangeable. There is no public `gpu build` or `gpu profile` subcommand.
The current Modal probe driver additionally clamps its worker budget at 600
seconds; raising this command's `--budget` beyond that does not extend the
driver's fixed function lifetime.

## `fastmm`: low-rank matrix-multiplication research

```sh
openassembler fastmm build --arch sm_80 --variant opt
openassembler fastmm build --only sgemm,lincomb --variant sched
openassembler fastmm run --n 4096 --schemes strassen_2x2x2,at_2x2x2,naive_2x2x2 --gpu A100-40GB --budget 400 --dry-run
```

### `fastmm build`

| Option | Default | Meaning |
|---|---|---|
| `--arch ARCH` | `sm_80` | Build target. |
| `--variant LEVEL` | `opt` | Scalar compiler policy: `safe`, `mined`, `sched`, `opt`. |
| `--only FILTER` | Empty/all | Comma-separated kernel-name substrings, not a list of factorization scheme names. |

The toolkit-backed builder generates direct GEMM, linear-combination and
fused-product research kernels under `corpus/ARCH/fastmm/`, with ptxas and OA
cubins and emulator checks. A partial build may not supply every kernel needed
by a later plan.

### `fastmm run`

| Option | Default | Meaning |
|---|---|---|
| `--n N` | `4096` | Square matrix dimension; it must meet the selected plans' divisibility/tile requirements. |
| `--schemes LIST` | `strassen_2x2x2,at_2x2x2,naive_2x2x2` | Comma-separated registered schemes. Invalid schemes are rejected. |
| `--gpu TYPE` | `A100-40GB` | Modal GPU request. |
| `--reps N` | `5` | Timed repetitions of a plan. |
| `--rounds N` | `2` | Shuffled plan timing rounds. |
| `--budget SECONDS` | `400` | GPU-worker budget. |
| `--dry-run` | Off | Construct/report plans without requesting their GPU execution. This still invokes the Modal CLI and needs the relevant built kernels. |

This wrapper uses Modal only and compares research plans, including the direct
and cuBLAS controls defined by the driver. Its default path uses FP32 scalar
research kernels; it is **not** the native FP16/BF16 Tensor Core `gemm` command
and does not establish an AlphaTensor training speedup. Numerical checks and
sample coverage are those of the research driver. Extra switches exposed by
`tools/fastmm_run.py` directly, such as its internal variant/seed controls,
are not public `openassembler fastmm run` options.
The current fastmm Modal driver clamps its worker budget at 600 seconds.

## `benchmark`: CLI integration matrix

```sh
openassembler benchmark --out-dir output/cli-benchmark
openassembler benchmark --execute --backend modal --build-backend modal --budget 300
openassembler benchmark --execute --backend local --build-backend local --skip-optional
```

| Option | Default | Meaning |
|---|---|---|
| `--backend MODE` | `modal` | Execution backend for the component run commands. Individual rows enforce their own supported transports. |
| `--build-backend MODE` | `modal` | Application capture transport: `modal`, `local`, `ssh`. |
| `--build-host USER@HOST` | None | Host for application SSH capture. |
| `--out-dir DIR` | `output/cli-benchmark` | Matrix artifacts/report root. |
| `--budget SECONDS` | `300` | Passed to GPU run rows; it is not one aggregate cap for the entire matrix. |
| `--execute` | Off | Actually execute the planned commands. Without it, record a command matrix with rows marked skipped. |
| `--skip-optional` | Off | Skip the optional NN-kernel-study and fastmm dry-plan rows. Primary source-build/probe/application rows remain. |

The matrix exercises interface integration and records command outputs/statuses
under `OUTDIR/runs/cli-benchmark-TIMESTAMP.json`. It is not the whole-transformer
performance study, a cuBLAS tuning search, or proof that every supported CLI
combination is covered. Toolkit, corpus, network and application prerequisites
still apply to the rows being executed.

## `status`, diagnostics and exit behavior

`openassembler status` has no command-specific options. It displays the first
Markdown table in `docs/STATUS.md` and points to the full document. It does not
reprofile the GPU or run a new test suite.

Common exit codes are 0 for success, 1 for a failed build/check/run, 2 for an
argument-usage error or an unverified compiler-check outcome, and 130 for an
interrupted command. External research wrappers may propagate their tool's
exit code. `compile --verify` can produce a `not-verified` result when a
construct is outside the functional checker; that is not a semantic pass.
Unsupported native compilation fails; vendor GPU fallback is disabled.

For ambiguous kernels, inspect the manifest and use `--source`/`--kernel`.
For an incorrect guessed launch, provide `--spec` rather than increasing N at
random. For a different application shape, edit its configuration and recompile.
For architecture, numerical and performance limits, consult the feature dossier
and STATUS; a successful argument parse is not a hardware-support guarantee.
