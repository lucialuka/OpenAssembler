<p align="center">
  <img src="figs/openassembler_logo.png" width="820" alt="OpenAssembler">
</p>

<h1 align="center">OpenAssembler</h1>

**Compile customized GPU kernels and Python model graphs to native NVIDIA SASS and cubins.**

OpenAssembler targets **A100 / SM80**. Its native backend selects instructions,
allocates registers, schedules dependencies and encodes GPU machine code.
Production execution loads OpenAssembler-owned cubins through the CUDA driver.

<h4 align="center"> Python 3.12+ · Linux and macOS · Apache 2.0 </h4>
<p align="center">
 <a href="https://www.star-history.com/lucialuka/openassembler">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank" />
   <img alt="Star History Rank" src="https://api.star-history.com/badge?repo=lucialuka/OpenAssembler&type=rank" />
  </picture>
 </a>
</p>

## Quick start

Install with Homebrew and authenticate Modal once:
```bash
brew tap lucialuka/openassembler https://github.com/lucialuka/OpenAssembler
brew install lucialuka/openassembler/openassembler
```
<details>
  <summary>If you are using Modal</summary>
    If you do not have access to real NVIDIA hardware, you can use Modal for A100 and other Nvidia hardware. Just enter:

```bash
modal setup
```
</details>

---

Start OpenAssembler from your terminal:

```bash
openassembler
```

The logo appears each time you open the interactive shell. At its prompt, enter
commands directly. From a source checkout, try the vector-add example on Modal:

```bash
compile examples/vector_add.py --backend modal --out-dir output/first-kernel --verify
run output/first-kernel --spec examples/run_spec.json --backend modal
exit
```
Choose one backend: `modal`, `local`, or an SSH destination such as `user@host`.
Local Linux supports compilation and execution with the required CUDA toolkit,
driver and A100. macOS supports local compilation through Colima; use Modal or
SSH for GPU execution. Type `help`, `help compile`, or press Tab for completion.
Type `exit` or press Ctrl-D to return to your terminal.

If an old Python environment shadows the Homebrew executable, run `deactivate`
in your terminal, then launch `openassembler` again. See
[installation troubleshooting](docs/HOMEBREW.md#an-old-python-environment-shadows-homebrew).

The [first-kernel walkthrough](docs/getting-started/first-kernel.md) explains
the expected artifacts and launch specification. See the
[installation guide](docs/getting-started/installation.md) for local Linux, macOS/Colima and SSH requirements.

## Supported inputs

| Input | Compilation route |
|---|---|
| PTX | OpenAssembler parses admitted PTX and compiles it with the native backend. Best for customized MMA (Matrix Multiplication and Addition) operation |
| Typed Python kernels | **Customized Kernel MUST use @kernel at the start.** The supported @kernel subset lowers to PTX, then uses the native backend. |
| CUDA C/C++ device sources | nvcc emits PTX; OpenAssembler compiles the admitted PTX. Better for routine ML tasks.|
| Python model repositories | Static discovery or an explicit model contract captures supported CPU PyTorch graphs and generates native kernels. |
| Generated GEMM and rowwise kernels | Dedicated generators compile bounded shape and policy contracts. |

These are bounded compiler contracts. Arbitrary Python programs, unsupported
operations, host launch rewriting and architectures beyond SM80 are not
supported. Unsupported GPU operations fail explicitly. The production route
does not use Triton, Inductor or vendor GPU fallback. Read the
[supported-input guide](docs/user-guide/supported-inputs.md) before choosing a
larger workload.

## Build, inspect and run

After starting `openassembler`, use the same repository path across the workflow:

    inspect /path/to/repo --json
    compile /path/to/repo --backend modal --verify
    run /path/to/repo --backend modal

Use cnr to compile and run in one command. The compile step writes a manifest,
cubins, and readable SASS listings by default. Model bundles also save their
graph plan, weights and launch metadata. Run loads saved artifacts and does not recompile.
See the [command reference](docs/COMMANDS.md) for flags, defaults, outputs and
launch specifications.

For Python model setup, use the
[model walkthrough](docs/getting-started/first-model.md), the
[configuration reference](docs/reference/configuration.md), and the
[operation support table](docs/reference/supported-operators.md).

## Architecture and results

PTX, typed Python and CUDA source converge on PTX before OpenAssembler's
SM80 backend. Python model graphs use a separate CPU capture and graph-lowering
route. NVIDIA tools provide CUDA parsing and cubin metadata where needed.
OpenAssembler owns the executable GPU instructions and runtime validates the
saved module and launch contract. The
[compilation pipeline guide](docs/architecture/compilation-pipelines.md)
shows the stages and component ownership.

On A100, the retained 42-contract GEMM replay shows selected policies improving
on OpenAssembler's earlier baseline while same-source ptxas remains faster in
every nominal comparison. The measurements cover specific GEMM contracts, not
whole-model performance. See the
[method and results](exp/dispatch_tuning/RESULTS.md) and the
[current status](docs/STATUS.md).

## Documentation

Start at the [documentation home](docs/README.md) for task-based navigation.

| Reference | Use it for |
|---|---|
| [Commands](docs/COMMANDS.md) | CLI syntax, options and outputs |
| [Repository workflow](docs/REPOSITORY_WORKFLOW.md) | Discovery, configuration, build and replay |
| [Current status](docs/STATUS.md) | Architecture, support and validation state |
| [Compiler contracts](docs/compiler/README.md) | Implemented behavior and known gaps |
| [Research catalogue](docs/research/README.md) | Methods, results, limitations and integration decisions |
| [Reverse engineering](docs/reverse_eng_results/README.md) | Instruction, encoding and hardware observations |
| [Development](docs/DEVELOPMENT.md) | Contributor setup and tests |

The compiler and runtime live in openassembler/. Examples are in examples/,
regressions in tests/, and retained experiments in exp/. See the
[documentation maintenance rules](docs/development/documentation-guidelines.md)
when changing supported behavior or evidence.

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Star History 

<a href="https://www.star-history.com/?repos=lucialuka%2Fopenassembler&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=lucialuka/openassembler&type=date&legend=top-left" />
 </picture>
</a>
