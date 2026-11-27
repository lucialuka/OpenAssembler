# Install OpenAssembler

OpenAssembler targets NVIDIA A100 GPUs (SM80). The CLI needs Python 3.12 or
newer. The supported user installation path is Homebrew on macOS or Linux.

## Install

With Homebrew installed, register the formula repository and install:

    brew tap lucialuka/openassembler https://github.com/lucialuka/OpenAssembler
    brew install lucialuka/openassembler/openassembler
    openassembler --version
    openassembler --help

Then launch the interactive shell:

    openassembler

At its prompt, use `compile PATH --backend modal`, `run PATH --backend modal`,
or `cnr PATH --backend modal --verify`. Use `help` for commands and `exit` to
return to your terminal. Choose a single backend value; square-bracketed lists
are documentation notation and must not be pasted as an argument.

If an active virtual environment selects a broken `.venv/bin/openassembler`,
run `deactivate` in your terminal or use `"$(brew --prefix openassembler)/bin/openassembler"`.
See [PATH troubleshooting](../HOMEBREW.md#an-old-python-environment-shadows-homebrew).

The formula currently installs a moving repository branch, not an immutable
release. The version command reports the installed package version, not its
source commit. For reproducible builds or bug reports, record the formula and
repository revisions separately. See the [Homebrew guide](../HOMEBREW.md).

## Choose where work runs

- **Modal:** the formula includes the Modal CLI. Run modal setup once to
  authenticate. The workstation needs neither an NVIDIA GPU nor CUDA toolkit.
- **SSH:** configure noninteractive access to a Linux A100 with compatible
  tools and runtime. See the [SSH workflow](../REPOSITORY_WORKFLOW.md#ssh-from-your-workstation).
- **Local Linux:** native compilation needs CUDA toolkit tools. Execution also
  needs a matching NVIDIA driver and GPU.
- **macOS local compilation:** NVIDIA's toolkit runs in a Linux VM. Install
  Colima, then fetch and check the toolkit:

      brew install colima
      colima start
      openassembler-fetch-cuda
      openassembler-cuda ptxas --version

  The repository path must be inside Colima's mounted home directory.
  macOS cannot execute an NVIDIA cubin locally; use Modal or SSH for execution.

Installation does not build a kernel, authenticate Modal, configure an SSH
host, or provide a local GPU. Native PTX packaging needs NVIDIA tools, but the
build can run remotely. See [requirements by task](../DEVELOPMENT.md#toolkit-and-host-requirements)
and the [command reference](../COMMANDS.md).
