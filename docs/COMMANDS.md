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

## Find a command

| Command | Purpose |
|---|---|
| `openassembler` | Open the interactive command shell. |
| `openassembler --help` | List the top-level commands. |
| `openassembler --version` | Print the installed OpenAssembler version. |
| `openassembler compile SRC` | Statically discover/compile native kernels; use `--frontend torch`/`triton`/`inductor` to explicitly capture a configured PyTorch application. |
| `openassembler gemm ...` | Generate and compile a checked native A100 FP16/BF16 GEMM. |
| `openassembler rowwise ...` | Generate and compile cooperative native FP32 softmax or LayerNorm. |
| `openassembler run [PATH]` | Execute existing kernel cubins or replay a bundled application. |
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

## `compile`: kernel files, source trees and applications

```sh
openassembler compile examples/div_by_const.ptx -o division --verify
openassembler compile examples/vector_add.py --out-dir output_python
openassembler compile examples/vector_add.cu -I include --nvcc-flag=-DTILE=32 --out-dir output_cuda
openassembler compile my_kernel_repo/ --out-dir output_repo
openassembler compile testcases/test_whole_repo --frontend torch --verify --backend modal
```

`SRC` is required. A standalone `.ptx` file goes to the native backend; a
supported Python kernel uses OpenAssembler's typed `@kernel` subset; CUDA
C/C++ goes through `nvcc --ptx`, then the native backend. Host-only Python/C++
code is not translated into a GPU program. Directory discovery respects Git's
tracked/non-ignored files and recognizes kernel sources; output paths mirror
the source tree. A directory build does not accept a single output name.

A directory with `[application] frontend="torch"` in `openassembler.toml` still
uses the native repository route by default. Use `--frontend torch` (or the
aliases `triton`/`inductor`) to explicitly run the configured TorchDynamo/
AOTAutograd and Inductor/Triton capture. This is not native OpenAssembler SASS
generation for the complete model. A file inside that directory always takes
the standalone native-kernel route.

### Compilation options

These options are also accepted by `cnr`.

| Option | Default | Meaning |
|---|---|---|
| `-o NAME`, `--out NAME` | Source stem; application default name | Name one file's output or an application bundle. Use a name, not a path; use `--out-dir` for its directory. |
| `--out-dir DIR` | `REPO/output` for directory builds; `./output` for files | Build root; use the same override with `run REPO`. |
| `--kernel NAME` | All entries | Compile only the named entry in a standalone source. |
| `--arch ARCH` | `sm_80` | Target architecture. The validated native product target is SM80/A100; accepting an architecture string is not a support guarantee. |
| `--opt-level LEVEL` | `opt` | One of `safe`, `mined`, `sched`, `opt`; see below. |
| `--verify` | Off | Check standalone semantics with the emulator/PTX checker; for an application, compare eager/compiled observable state on CUDA. |
| `--trials N` | `6` | Random standalone semantic-check trials. |
| `--fallback ptxas` | Disabled | Explicitly keep vendor code for an unsupported kernel. The manifest marks it `ptxas-fallback`; it is not an OA code-generation result. |
| `-I DIR`, `--include DIR` | None | Additional CUDA include directory; repeatable. The source file's directory is included; directory builds also add the discovered repository root. |
| `--nvcc-flag FLAG` | None | Additional CUDA compiler flag; repeatable. Use `--nvcc-flag=-DNAME=VALUE` when a value begins with `-`. |
| `--dump-ptx` | Off | Print generated/input PTX. |
| `--no-sass` | Off | Omit the readable `nvdisasm` listing; still produce cubins and metadata. |
| `--backend MODE` | `auto` | Shared build/run destination: `local`, `modal`, `user@host`, or `ssh` with `--host`. Native auto builds locally; applications use configuration. |
| `--host USER@HOST` | None | SSH target or SSH config alias. Ports/keys/ProxyJump come from SSH configuration. |
| `--gpu TYPE` | Application config / `A100-40GB` | Modal GPU for application capture/execution. Native remote compilation is CPU-only. |
| `--budget SECONDS` | Application config / native build `900` | Per-stage budget. Remote native compilation accepts 30–1800 seconds. |
| `--job-timeout SECONDS` | Application config | Application capture watchdog; native compiler work is bounded by its build budget. |
| `--build-backend MODE` | Shared `--backend` / application config | Override only compilation, e.g. build on Modal and execute via SSH. Also accepts legacy `ssh`. |
| `--build-host USER@HOST` | Shared `--host` / application config | Override the SSH compilation target. The application bundle records its capture host. |
| `--build-gpu GPU` | Application configuration | Requested Modal GPU for application capture. Actual returned hardware is recorded separately. |
| `--build-budget SECONDS` | Shared `--budget` / configuration | Override only compilation time budget. |
| `--dry-run` | Off | Plan native compilation or application capture, without writing outputs, uploading sources or running workloads. `cnr` plans both stages. |
| `--frontend MODE` | `auto` | `auto`/`native` statically discover native kernels and use OpenAssembler; `torch`, `triton` and `inductor` explicitly select the configured application capture. `auto` never silently invokes TorchInductor. |

For scalar kernels, `safe` uses conservative in-order scheduling; `mined`
uses dependency-distance stall selection; `sched` adds list scheduling; `opt`
also applies the supported bounded loop unrolling. They are compiler policies,
not promises that increasing the level always improves a workload. Native
tensor kernels use their separate validated tensor policy at every level.

Application capture currently requires `--arch sm_80`. Standalone-only flags
(`--kernel`, `--fallback`, `-I`, `--nvcc-flag`, `--dump-ptx`, and nondefault
`--opt-level`/`--trials`) are rejected for applications. Transport selection
works for both frontends. Native repository discovery is static: host programs,
build systems, imports and launch logic are reported as context but are not
executed or translated into a GPU program. `--backend` applies to both stages of `cnr`; an
explicit `--build-backend` overrides only compilation. Native remote builds
ship the same compiler and encoding data and publish their results locally,
without requiring a local toolkit. Local native compilation has no build
watchdog and rejects explicit build-budget requests; `cnr --budget` still
bounds GPU execution. See [repository setup](REPOSITORY_WORKFLOW.md).

Capture `--build-backend auto` selects local execution on Linux when a `torch`
module is discoverable, otherwise Modal. This selection is not a GPU or pinned
runtime compatibility test; choose a transport explicitly when necessary.

The worker requires PyTorch 2.7.1 and SM80. The pinned Modal image uses CUDA
12.8 and recorded Python 3.11; those versions are not independent universal
constraints on every compatible local capture environment. Replay must match
the bundle's recorded PyTorch, CUDA, Python and device capability. Configuration
also fixes entrypoint arguments, seed, requirements, verification tolerances,
transport and budgets. Different model arguments require reconfiguration and
recompilation.

### Output artifacts

For a standalone source named `NAME`, the output root contains
`cubin/NAME.cubin`, `sass/NAME.sass` unless suppressed, `manifest.json`,
`.build/ptx/NAME.ptx` and `.build/reference/NAME.cubin`. Native SASS is encoded
by OpenAssembler and spliced into a ptxas-built ELF container; the reference
cubin is a separate ptxas compilation. A cubin is launched through the driver,
not executed as a host program.

Application builds instead add source/runtime/cache/graph/report bundles and
Inductor-generated cubins to the manifest. Their runtime dependencies remain
necessary. Builds lock each output directory; use separate output roots for
concurrent builds.

## `gemm`: checked native A100 tensor GEMM

```sh
openassembler gemm --m 1024 --n 4096 --k 1024 --dtype f16 --pointer-mode direct --out-dir output_gemm
openassembler gemm --m 16384 --n 4096 --k 4096 --cta-order morton8 --out-dir output_morton
openassembler gemm --m 16384 --n 4096 --k 4096 --cta-order grouped_n8 --out-dir output_grouped_n
openassembler gemm --m 1024 --n 4096 --k 1024 --fragment-schedule load_next --out-dir output_load_next
openassembler gemm --m 4096 --n 4096 --k 4096 --dtype bf16 --epilogue bias_residual_relu --out-dir output_fused
```

The operation is `C = A @ B.T`: contiguous `A[M,K]` and `B[N,K]` FP16/BF16
inputs; FP32 accumulation and contiguous `C[M,N]` output. M, N and K must be
positive multiples of 128. Inputs need 16-byte alignment; output needs
eight-byte alignment. Arbitrary strides, tails, alpha/beta and alternate
transposed storage are not part of this command.
Each input/output element count must fit the generator's unsigned 32-bit index
contract, and the N tile count must fit CUDA's grid-y limit.

| Option | Default | Meaning |
|---|---|---|
| `--m M`, `--n N`, `--k K` | Required | Matrix dimensions. |
| `--dtype TYPE` | `f16` | Input type: `f16` or `bf16`. |
| `--arch ARCH` | `sm_80` | This checked generator accepts only `sm_80`. |
| `--schedule MODE` | `stagepair` | `stagepair` is the validated alternating-stage plan; `combined` retains its earlier comparison plan. |
| `--fragment-schedule MODE` | `source` | `source` keeps the default source-order fragment stream; opt-in `load_next` is admitted for FP16/BF16 at 1024×4096×1024 and 4096×1024×1024, with stagepair, legacy pointers, late advancement and no epilogue. |
| `--cta-order ORDER` | `m_fast` | `m_fast`, `morton8`, or opt-in `grouped_n8`; Morton requires complete 8×8 output-tile groups, while grouped-N requires N tile count divisible by 8 and changes the launch grid to `(m_tiles×8, n_tiles/8)`. |
| `--epilogue MODE` | `none` | `none`, `bias_residual`, `bias_residual_relu`. Nonempty modes require `stagepair` and append FP32 bias/residual pointers. |
| `--pointer-mode MODE` | `legacy` | `legacy` or explicit `direct` pointer homes. Direct requires `stagepair`, `m_fast`, and `--epilogue none`. |
| `--pointer-advance MODE` | `late` | `late` (default), `early`, or shape-specific `auto` placement for the validated stagepair/m_fast/unfused pointer-update candidate. |
| `--out-dir DIR` | `output` | Output root. |
| `-o NAME`, `--out NAME` | `gemm` | Artifact basename; entry name remains `gemm`. |
| `--no-sass` | Off | Skip the readable listing. |
| `--emit-only` | Off | Write annotated generated PTX without invoking the CUDA tools. |

The fixed launch uses 128 threads and 65,536 dynamic shared-memory bytes per
CTA. The generated source annotation supplies the checked launch to `run`.
Custom callers must use the recorded launch and opt into the shared-memory
size. Fusion computes `(A @ B.T + Bias) + Residual`, optionally followed by
ReLU, with contiguous FP32 `Bias[N]` and `Residual[M,N]`.

Morton and grouped-N orders are logical output-tile traversals, not new MMA
instructions or promises about which SM runs each block. Direct pointers and fusion have
shape-specific wins and regressions. None is automatically selected. There
is no `morton6` CLI option.

The annotated PTX is saved under `.build/generated/NAME.ptx`. Recompiling that
exact source with `compile` retains its plan/pointer policy. Edited or
unannotated PTX loses inferred launch/buffer metadata and uses the default
tensor policy; describe an actual launch with `run --spec`.

## `rowwise`: cooperative FP32 softmax and LayerNorm

```sh
openassembler rowwise --op softmax --rows 1024 --width 512 --out-dir output_softmax
openassembler rowwise --op layernorm --rows 129 --width 127 --epsilon 1e-5 --out-dir output_norm
openassembler rowwise --op softmax --rows 7 --width 33 --emit-only --out-dir generated
```

Both operators read/write contiguous FP32 `[rows,width]`, with four-byte-aligned
buffers. One warp cooperates on each row, four warps per CTA; the launch uses
128 threads, `ceil(rows/4)` blocks and no shared memory. Row and column tails
are supported. Input values and the relevant differences/sums/squares must
remain finite in FP32; NaN/Inf and overflowing intermediate arithmetic are not
covered. The entry ABI is `(u64 X, u64 Y)`; use separate input/output buffers.

| Option | Default | Meaning |
|---|---|---|
| `--op NAME` | Required | `softmax` or `layernorm`; also selects the entry name. |
| `--rows R` | Required | Positive row count below 2³²; total elements must not exceed 2³². |
| `--width W` | Required | Integer width 1–1024. |
| `--epsilon VALUE` | `1e-5` | LayerNorm epsilon. It must round to a positive, normal, finite FP32 value. The plan validates it for both operations. |
| `--arch ARCH` | `sm_80` | Only `sm_80` is supported by this checked generator. |
| `--out-dir DIR` | `output` | Output root. |
| `-o NAME`, `--out NAME` | Operation name | Artifact basename; entry name remains the operation name. |
| `--no-sass` | Off | Skip the readable listing. |
| `--emit-only` | Off | Write annotated PTX without the CUDA tools. |

LayerNorm uses population variance and produces normalized values only: no
affine weight/bias, saved mean/inverse deviation or backward. These explicit
generators do not automatically replace arbitrary Python loops or operators
inside captured PyTorch graphs. Tiny cases can regress; read the dossier's
timing scope before treating the command as a dispatch recommendation.

## `run`: existing kernel binaries or application replay

```sh
openassembler run output_gemm --backend modal --budget 300 --job-timeout 120
openassembler run output_softmax --backend modal --dry-run
openassembler run output_repo --source kernels/ops.ptx --kernel relu --spec launch.json --backend user@host
openassembler run output_app --application test_whole_repo --backend modal --verify-application
```

`PATH` accepts a source repository/file or an existing build directory.
`run REPO` reads `REPO/output/manifest.json`; custom builds use
`run REPO --out-dir DIR`. A source path restricts execution to its own saved
records; a sole matching application is selected automatically. It never
compiles or recaptures. `run DIR` still accepts a directory containing a
manifest, and bare `run` retains the `./output` workflow. Legacy builds in
`./output` are found only when their recorded source matches the supplied
repository. The repository-local build takes priority over that legacy path;
ambiguous single-file builds require `--out-dir`. With an explicit
mixed kernel/application build directory, select `--application NAME`.

### Kernel and shared execution options

These flags are also accepted by `cnr`, except where its section says otherwise.

| Option | Default for standalone kernels | Meaning |
|---|---|---|
| `--backend MODE` | `auto`, resolved to `local` for kernels | `local`, `modal`, `user@host`, or `ssh` with `--host`. Applications reuse their saved capture transport in auto mode. |
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
| `--application NAME` | Auto when unambiguous | Select a saved application bundle. Cannot combine with `--kernel`, `--source`, `--spec`, `--from-results`, or `--save-jobs`. |
| `--app-arg ARG` | Saved arguments | Repeatable application-argument entries. A supplied list must exactly match the saved argument list; it does not specialize a new shape. Use `--app-arg=--flag` for a token beginning with `-`. |
| `--verify-application` | Off | Compare eager/compiled application state on replay using the saved numerical policies. Requires a selected application, inferred from the repository or an unambiguous application-only build. |

For applications, `--backend auto` chooses the saved capture transport. The
accepted replay transports are `local`, `modal`, and SSH targets. Use
`--backend user@host` directly, or `--backend ssh --host user@host`; a saved
SSH host remains usable with `--backend auto` or `--backend ssh`.
Application `--gpu`, `--budget` and `--job-timeout` default to saved values;
budgets must be 30–1800 seconds and the watchdog must be positive and no larger
than that budget. Nondefault `-n` or `--reps` is rejected with explicit application
selection. Model arguments are saved in the bundle. Cache-only replay blocks
new GPU code generation and rejects missing or changed runtime artifacts.
Both `--app-arg` and `--verify-application` accept the application inferred
from a repository path; an explicit `--application NAME` is only needed when
selection is ambiguous.

### What the native auto-harness verifies

Generic pointer arguments are guessed as FP32 buffers of N elements. Integer
arguments named `n/count/size/len/num/length` receive N; other integers receive
16. Other scalar arguments get deterministic test values. The default launch
is one-dimensional, block 256, grid `ceil(N/256)`. This is a convenience for
simple elementwise kernels, not a valid launch for arbitrary user code.

A checked generated GEMM gets its real half-precision buffers, tile launch and
shared memory. It uses exact small integers and compares all FP32 output bits
with ptxas; fused bias/residual inputs are zero in this smoke test. A checked
rowwise kernel gets zero input and NaN-poisoned output; every result must match
the analytic constant, remain finite, and also match ptxas bitwise. These smoke
checks do not replace full nonzero numerical/canary/paired-performance tests.

Standalone `run` compares selected output buffers bit for bit; it does not
silently apply a floating-point tolerance. A numerical difference, missing
output, fault or skipped case yields nonzero status. Results are saved in
`OUTDIR/run.json` and timestamped `OUTDIR/runs/*.json`. Event timing measures
the specified launch sequence; it is not automatically a complete training step.

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

## `cnr`: compile and run

```sh
openassembler cnr examples/vector_add.py --spec examples/run_spec.json --backend modal --out-dir output_cnr
openassembler cnr examples/vector_add.cu --backend user@host --out-dir output_cnr
openassembler cnr examples/div_by_const.ptx --verify --dry-run --out-dir output_cnr
openassembler cnr testcases/test_whole_repo --frontend torch --verify --backend modal
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
`--frontend torch`/`triton`/`inductor` selects the configured application route.
`--backend` selects the transport for either frontend.
`--build-backend` optionally separates those locations. `--dry-run` plans both
stages without builds, uploads, output writes or GPU jobs, including for native
sources. `--verify` also requests application replay verification. Application
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
Unsupported native compilation fails unless explicit ptxas fallback is allowed.

For ambiguous kernels, inspect the manifest and use `--source`/`--kernel`.
For an incorrect guessed launch, provide `--spec` rather than increasing N at
random. For a different application shape, edit its configuration and recompile.
For architecture, numerical and performance limits, consult the feature dossier
and STATUS; a successful argument parse is not a hardware-support guarantee.
