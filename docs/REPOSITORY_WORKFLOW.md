# Build and execute a repository

The same source path works with all three commands:

```sh
openassembler compile /path/to/repo --backend modal --verify
openassembler run /path/to/repo --backend modal
openassembler cnr /path/to/repo --backend modal --verify
```

`compile` creates a build. `run` only executes a saved build. `cnr` compiles
and executes the products of that invocation. `--dry-run` plans work without
building, uploading or submitting jobs. All three accept `local`, `modal` or
an SSH destination such as `user@a100` as their backend.

The default frontend is native static discovery. A repository is scanned for
PTX `.entry` kernels, CUDA/C++ `__global__` kernels and Python `@kernel`
functions. A present `openassembler.toml` does not silently switch the build to
TorchInductor/Triton. Select an application explicitly with `--frontend torch`,
`--frontend triton`, or `--frontend inductor`.

## Choose the repository's frontend

**Native GPU kernels:** a directory without an application configuration is
scanned for `.ptx` entries, CUDA C/C++ `__global__` kernels and typed Python
`@kernel` functions. Paths are mirrored into the output tree. NVIDIA's frontend
handles CUDA C/C++ parsing; OpenAssembler handles supported PTX instruction
selection, allocation, scheduling and encoding. Unsupported forms fail unless
an explicitly labelled `--fallback ptxas` is requested. Host scripts, CMake,
shell scripts and arbitrary Python functions are not inferred GPU kernels.

Native execution needs the program's buffer and launch contract. For anything
beyond a simple elementwise kernel or a checked `gemm`/`rowwise` plan, supply
`--spec launch.json`. Discovery compiles independent sources; it does not infer
how an entire host application should allocate data or call those kernels.
The [command reference](COMMANDS.md#run-existing-kernel-binaries-or-application-replay)
documents the spec format and reference-comparison behavior.

**PyTorch application:** put `openassembler.toml` in the supplied repository
root. For example, adapt this to your actual training entry point and arguments:

```toml
[application]
frontend = "torch"
entrypoint = "train.py"
args = ["--device", "cuda", "--steps", "8"]
seed = 7
torch = "2.7.1"
build_backend = "modal"
gpu = "A100-40GB"
budget = 900
job_timeout = 600
verification = "state"
```

Capture executes that configured workload and saves its observed model
specializations through TorchDynamo/AOTAutograd and TorchInductor/Triton.
GPU kernels become cubins and SASS listings; Python/PyTorch and external CUDA
libraries remain runtime dependencies. A whole repository is not converted to
one standalone host executable, and unobserved shapes/branches are not compiled
in advance. The native optimization research is preserved independently;
native GEMMs and reductions are not silently substituted into application graphs.

```sh
openassembler compile /path/to/repo --frontend torch --backend modal --verify
openassembler run /path/to/repo --backend modal --verify-application
```

The source path automatically selects its saved application. Use
`--application NAME` when a build contains multiple application choices.
`--verify` on `cnr` checks both capture and replay. Replay blocks new GPU
code generation; changed model arguments need a new capture. See the
[application dossier](research/ideas/pytorch-application-capture.md) for runtime,
state-check and numerical limitations. The included
[transformer example](../testcases/test_whole_repo/README.md) needs no dataset download.

## Output location and selection

Directory builds default to the supplied repository's `output/`, independent
of the shell's working directory:

```text
repo/output/
  manifest.json
  cubin/                 GPU modules
  sass/                  corresponding readable disassembly
  .build/                native PTX/references or application sources/caches/reports
  runs/                  execution records after real runs
  run.json               latest execution report
```

Applications also have `applications/NAME/config.json`; their GPU modules and
listings live under `cubin/applications/NAME/` and `sass/applications/NAME/`.
Keep the complete build root for execution; a cubin or SASS directory alone is
insufficient. Cubins are loaded by a CUDA driver/runtime and are not shell
executables. `--no-sass` explicitly omits readable listings.

Single-file builds retain the default `./output`. A custom build root is
relative to the invoking shell, not the source repository:

```sh
openassembler compile /path/to/repo --out-dir /path/to/build --backend modal
openassembler run /path/to/repo --out-dir /path/to/build --backend modal
# An explicit artifact directory also works, including copied bundles:
openassembler run /path/to/build --backend modal
```

`run REPO` checks recorded source ownership and selects only that repository's
products, even when a custom build root contains other sources. It can also
find a legacy `./output` build whose recorded sources match. If both legacy
and repository-local builds match, pass `--out-dir` explicitly. There is no
global build index or arbitrary recursive search. A missing build gives a
compile instruction, rather than triggering compilation.

Native source edits produce stale-source warnings. Application replay uses
its saved source bundle, not newly edited local files. Recompile to execute
new code. Corrupt manifests and changed/missing application artifacts are
rejected. Builds use writer locks and only retire owned artifacts; user files
in the build root are preserved.

## Modal from a workstation

Install OpenAssembler with Python 3.12+ and set up the optional Modal client:

```sh
python3 -m pip install -e .
python3 -m pip install modal
modal setup
openassembler cnr testcases/test_whole_repo --backend modal --verify --dry-run
openassembler cnr testcases/test_whole_repo --backend modal --verify
```

The `modal` executable can belong to a separate environment, but must be on
`PATH` and authenticated. No local CUDA toolkit or GPU is needed for remote
compilation. Native compilation uses a CPU-only CUDA 12.8.1 development image
with Python 3.12; native execution uses the existing bounded A100 runner.
Application capture/replay uses the pinned PyTorch 2.7.1/CUDA 12.8 image.
The default GPU request is `A100-40GB`; returned hardware is recorded in results.

Source/cache archives are bounded to 256 MiB and 20,000 files each. Native
remote builds transfer the repository (or a single source's parent directory)
and any explicit include directories, following source discovery exclusions.
Git-ignored files, symlinks, environments and the selected output root are
excluded; an explicitly requested single source is included. Use `.gitignore`
to exclude datasets/generated files or build locally for larger source trees.
Remote builds transfer the installed compiler/data, rather than installing
an unrelated released compiler version. Structured `-I DIR` includes are staged;
paths embedded in arbitrary `--nvcc-flag` values must make sense on the remote host.

`--budget` applies per stage, with transport/cleanup grace periods; image builds,
queues and uploads add wall time. Native remote builds accept 30–1800 seconds.
Application budgets accept 30–1800 seconds and their positive watchdog cannot
exceed the stage budget. `cnr` can consume separate build and run budgets.
`--build-budget` overrides only compilation. Apps are ephemeral with no retries
and one container per function. Inspect `modal app list` after a hard client
interruption and stop an unfinished app with `modal app stop APP_ID`.

## SSH from your workstation

Configure noninteractive SSH access to your Linux A100 host first. For example:

```sshconfig
Host training-a100
    HostName gpu.example.com
    User your-user
    IdentityFile ~/.ssh/id_ed25519
    Port 22
```

Test the environment through a noninteractive command, since its `PATH` may
differ from an interactive terminal:

```sh
ssh -o BatchMode=yes training-a100 'python3 --version; command -v timeout; nvidia-smi'
# Native remote compilation additionally needs these on PATH:
ssh training-a100 'ptxas --version; nvdisasm --version; nvcc --version'
```

Native compilation needs Python 3.12+, CUDA toolkit tools and a compatible host
C++ compiler for CUDA input. Native execution needs Python 3, GNU `timeout`,
an NVIDIA driver and the matching A100; it does not need a toolkit. Application
capture needs PyTorch 2.7.1 and its CUDA dependencies on the remote `python3`.
Application replay must match the saved Python, torch, CUDA and SM capability;
a bundle captured on Modal is not automatically portable to a differently
versioned SSH environment. Capture and replay on that host when necessary.
The OpenAssembler worker/compiler is uploaded automatically; installing
OpenAssembler on the remote host is not required for workstation dispatch.

Directory native builds compile independent source files with a bounded worker
pool by default (`OPENASSEMBLER_BUILD_JOBS=4`, capped at eight). Candidate order,
manifest ownership and per-source verification remain deterministic. Set
`OPENASSEMBLER_BUILD_JOBS=1` to reproduce the historical serial build or when
the host is CPU constrained. `--dump-ptx` intentionally uses the serial path
so PTX output remains readable and ordered.

```sh
openassembler compile /path/to/repo --backend training-a100 --verify
openassembler run /path/to/repo --backend training-a100
openassembler cnr /path/to/repo --backend training-a100 --verify
# Equivalent explicit spelling, also usable to change an application's saved host:
openassembler run /path/to/repo --backend ssh --host training-a100
```

SSH aliases, `user@host` and bracketed IPv6 destinations are supported. Configure
ports, keys and jump hosts in SSH config. SSH/SCP use bounded noninteractive
requests, unique temporary directories and best-effort cleanup; cleanup errors
do not replace the original compiler/runtime result.

## Inside an SSH terminal on the GPU host

After connecting and installing the CLI in a suitable Python environment,
the repository is local to that host:

```sh
ssh training-a100
cd /path/to/OpenAssembler
openassembler compile /path/to/my_repo --backend local --verify
openassembler run /path/to/my_repo --backend local
openassembler cnr /path/to/my_repo --backend local --verify
```

Local native compilation does not require a GPU, but execution and application
capture do. `--backend auto` builds native sources locally; application auto
uses its configuration (or the existing Linux/torch discovery policy). Choose
`local` explicitly to override a configuration that requests Modal.

Local native compilation runs in the CLI process without a build watchdog;
explicit build-budget requests are rejected rather than silently ignored.
`cnr --budget` still bounds its GPU execution stage. Remote native builds and
application workers provide their documented build/capture timeouts.

For advanced workflows, separate the build and run destinations:

```sh
openassembler cnr /path/to/my_repo --build-backend modal --backend training-a100
```

This still requires compatible application runtime versions across destinations.
Native code generation, optimization levels, tensor schedules, pointer policies
and launch contracts are unaffected by transport selection. See
[COMMANDS.md](COMMANDS.md) for every option and [STATUS.md](STATUS.md) for
measured performance and supported compiler coverage.
