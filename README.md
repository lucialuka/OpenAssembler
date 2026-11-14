
<p align="center"><img src="figs/openassembler_logo.png" width="680" alt="OpenAssembler"></p>

# OpenAssembler
**A native NVIDIA SASS/cubin compiler for SM80/A100, built through black-box reverse engineering and validated kernel optimizations.**

OpenAssembler explores how much more performance existing GPUs can deliver for model serving and training through better native code generation.

OpenAssembler generates native SASS and executable cubins for supported A100/SM80 kernels, with shape-specific kernel and compiler optimizations. Performance depends on the operation and comparison: measured improvements over earlier OA kernels do not imply a win over matched ptxas or cuBLAS, and a whole-native transformer-training speedup has not been established. The CLI supports standalone kernels and repository discovery for supported Python, CUDA C/C++ and PTX inputs. A constrained Python model route captures CPU graphs and emits OA-owned forward/backward/loss/update cubins; Inductor/Triton execution is disabled. Broader architecture support and end-to-end training acceleration remain research goals; see the measured contracts and limitations below.

[Architecture and status](docs/STATUS.md) · [All commands](docs/COMMANDS.md) ·
[Research catalogue](docs/research/README.md) · [Development setup](docs/DEVELOPMENT.md)
· [Repository workflow and SSH/Modal setup](docs/REPOSITORY_WORKFLOW.md)
· [Reverse-engineering results](docs/reverse_eng_results/README.md)

<p align="center">
 <a href="https://www.star-history.com/lucialuka/openassembler">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank" />
   <img alt="Star History Rank" src="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank" />
  </picture>
 </a>
</p>


> ⚠️ **IMPORTANT NOTICE**
> This project is currently under **active heavy development** for kernel optimization and assembly language edge case handling. If you encounter any bugs/compiler inefficiency during deployment, please feel free to open an Issue. Your feedback will help us build a better version!

## Compile a repository

```sh
openassembler inspect /path/to/repo       # static coverage; no user code executes
openassembler /path/to/repo --backend modal --verify
openassembler cnr /path/to/repo --backend user@a100 --verify
openassembler run /path/to/repo --backend user@a100
```

A source path is shorthand for `compile PATH`. `compile` and `cnr` default to the
current directory. `[model]` selects the Python model bridge; `--frontend native`
selects discovered Python `@kernel`, CUDA C/C++/C `__global__`, and PTX `.entry`
kernels. Unsupported code fails with a local `output/compile-error.json` report;
`inspect --json` shows candidates and skipped files. SASS listings are optional
(`--no-sass`) and ignored by Git. See the [codebase audit](docs/CODEBASE_AUDIT.md)
for ownership, cleanup, and remaining compiler boundaries.

## Native Python models

A repository with a `[model]` factory and exact input shapes now compiles supported
Python tensor graphs, their backward pass, MSE loss and SGD updates through OA.
The default FP32 contract and explicit FP16-operand Tensor Core contract have
separate numerical gates. Local, Modal and SSH execution use saved OA cubins.
Unsupported operations fail; vendor GPU fallback is disabled. Explicit FP16 BMM
now covers forward/dX/dW for equal batches through 32 and matrix dimensions
through 512, including tails. Attention remains composed from OA products,
FP32 softmax, layouts and projections; masks/broadcasting and fused attention
are outside this bounded contract. See the [A100 validation](exp/attention_half/RESULTS.md).

```sh
openassembler cnr examples/native_model --backend modal --verify
openassembler cnr examples/native_tensor_model --backend modal --verify
openassembler cnr examples/native_attention_model --backend modal --verify
openassembler run examples/native_model --backend user@a100 --verify-application
```

See the [native model contract](docs/research/ideas/native-model-bridge.md).
This constrained route does not yet compile arbitrary repositories or establish
performance superiority over the NVIDIA toolchain. Earlier Inductor application
studies below remain historical comparator evidence.

## What you can do

- Compile standalone kernels or discover supported kernels in a repository.
- Generate full-tile FP16/BF16 GEMMs with FP32 output. Manual policy choices
  include tile64 copy-address reuse and pointer placement, and optional
  correctness-gated A100 tuning can compare the complete admitted policy set.
  Automatic shape entries require a current compiler identity and independently
  confirmed held-out evidence. The current 42-contract replay installs 34
  tuned native policies and eight verified baseline retentions; unknown/stale
  shapes retain the reference. See the [measured results](exp/dispatch_tuning/RESULTS.md).
- Generate cooperative FP32 softmax and population LayerNorm with row/column tails.
  Five exact A100 contracts automatically select held-out-validated CTA geometry;
  other contracts retain the four-warp reference.
- Inspect cubins and SASS, emulate supported instructions, and run matched GPU
  checks locally, over SSH or on Modal.
- Capture supported Python model graphs, generate OA forward/backward/loss/update
  kernels, and replay their saved native launch plans with state verification.
- Study factorized matrix multiplication, compiler encodings and scheduling
  through the retained research tools and evidence.
- Audit transformer QKV/output/FFN roles and shape/layout coverage before
  selecting a native GEMM policy.

Support and performance are workload-specific. The [status page](docs/STATUS.md)
separates native compiler results, library controls and application-frontend
results. The [research catalogue](docs/research/README.md) includes successful,
rejected, mixed and untested ideas, each with its own methods and reproduction.
The [native NN arithmetic campaign](docs/research/ideas/nn-arithmetic-campaign.md)
adds isolated operators, forward/backward GEMMs and six composed networks;
its new candidates remain experimental and its failed gates remain excluded.

## Quick start

On macOS or Linux with Homebrew, install the complete CLI and its Modal
transport with:

```sh
brew tap lucialuka/openassembler https://github.com/lucialuka/OpenAssembler
brew install openassembler
```

The formula installs Python 3.12, the compiler/data tree, examples,
documentation and the Modal CLI. See [Homebrew installation](docs/HOMEBREW.md)
for the CUDA toolkit and SSH runtime requirements. If the formula is not yet
available from the tap, install the local formula from a checkout with
`brew install --build-from-source ./Formula/openassembler.rb`.

Use Python **3.12 or newer**:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Modal, set up its optional client once. The workstation needs Python and Modal; remote compilation supplies the
CUDA toolkit, and remote execution supplies the GPU driver:

```sh
pip install modal
modal setup
openassembler compile examples/native_model --frontend model --backend modal --verify
openassembler run examples/native_model --backend modal --verify-application
# Or capture and replay in one command:
openassembler cnr examples/native_model --frontend model --backend modal --verify
```

Replace that example with your repository path. A model repository provides
`openassembler.toml` with its CPU factory and exact input shapes; a native-kernel
repository needs no application configuration. Directory builds default to `REPO/output/`: native kernels have `cubin/`,
`sass/` and `manifest.json`; native models have `plan.json`, `native-ledger.json`,
weights and their native kernel tree.
For a custom location, use the same `--out-dir DIR` on `compile` and `run`.
Independent native source files build with up to four bounded workers by
default; set `OPENASSEMBLER_BUILD_JOBS=1` for serial reproducibility.

Use an existing Linux A100 host over SSH with the same commands:

```sh
openassembler compile /path/to/my_repo --backend user@a100 --verify
openassembler run /path/to/my_repo --backend user@a100
openassembler cnr /path/to/my_repo --backend user@a100 --verify
```

SSH needs noninteractive access and the remote toolkit/runtime described in
the [setup guide](docs/REPOSITORY_WORKFLOW.md#ssh-from-your-workstation).
Inside an SSH terminal on the GPU host, use `--backend local` instead.
Native multi-kernel execution needs your actual buffers and launches in
`--spec launch.json`; discovery alone does not infer a training program.

For local native compilation, an installed Linux CUDA toolkit can be used.
On macOS the toolkit runs in a Colima Linux VM:

```sh
# macOS only: install Colima if necessary, then start its VM.
colima start
# Download the pinned tools and check their version.
env/fetch_cuda.sh
env/cuda.sh ptxas --version
```

See [development setup](docs/DEVELOPMENT.md#toolkit-and-host-requirements) for
platform details, optional dependencies and an existing-toolkit installation.

Compile a kernel and inspect the intended launch without sending a GPU job:

```sh
openassembler compile examples/vector_add.py --out-dir output/vector
openassembler run output/vector --dry-run
```

Generate a native tensor kernel or cooperative reduction:

```sh
openassembler gemm --m 1024 --n 1024 --k 1024 --dtype f16 --out-dir output/gemm
openassembler gemm --m 1024 --n 4096 --k 1024 --fragment-schedule load_next --out-dir output/load_next
openassembler gemm --m 16384 --n 4096 --k 4096 --cta-order grouped_n8 --out-dir output/grouped_n
openassembler rowwise --op softmax --rows 1024 --width 512 --out-dir output/softmax
```

A `[model]` configuration selects native graph capture automatically;
`--frontend native` requests static kernel discovery. Inspect work before submission:

```sh
openassembler compile examples/native_model --frontend model --verify --dry-run
openassembler cnr /path/to/my_repo --backend user@a100 --dry-run
```

To execute native outputs on an authenticated Modal GPU:

```sh
openassembler run output/gemm --backend modal --budget 180 --job-timeout 60
```

Run `openassembler` without arguments for the interactive shell, or use
`openassembler --help` and `openassembler COMMAND --help` for one-shot commands.
The [complete command reference](docs/COMMANDS.md) explains every command and
flag. Actual research reproduction belongs to the
[individual idea dossiers](docs/research/README.md), not this introduction.

## Repository map

- `openassembler/`: frontends, native backends, binary support and GPU/application runtime.
- `specs/`, `corpus/`, `probes/`, `tools/`: encoding data, test inputs and research tooling.
- `tests/`, `examples/`, `testcases/`: regression tests and executable examples.
- `docs/`: architecture, command reference, per-idea research and retained evidence.
- `exp/`: isolated experiment drivers and raw records; no production import dependency.

The package is version **0.2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Star History

<a href="https://www.star-history.com/?repos=lucialuka%2Fopenassembler&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&legend=top-left" />
 </picture>
</a>

[🇭🇰 Free Hong Kong](https://en.wikipedia.org/wiki/Liberate_Hong_Kong,_revolution_of_our_times)

