# Homebrew installation

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Homebrew is the user installation path on macOS and Linux. With Homebrew already
installed, copy this single command:

```sh
brew tap lucialuka/openassembler https://github.com/lucialuka/OpenAssembler && brew install lucialuka/openassembler/openassembler
```

It registers the project's formula repository and installs OpenAssembler with
Python 3.12, compiler data, examples, documentation and the Modal client. A source
checkout, Python virtual environment and manual `pip install` are unnecessary.
The explicit formula name also avoids ambiguity if older OpenAssembler taps are
already registered.

## Why the tap is needed

`brew install openassembler` works only when Homebrew can already resolve that
name. OpenAssembler is not currently listed in `homebrew/core`, and Homebrew
does not search arbitrary GitHub repositories for unknown formula names. The
command above performs the one-time registration as part of installation.

The explicit URL uses the existing OpenAssembler repository. Homebrew's usual
automatic tap lookup would require a separate `homebrew-openassembler` repository.
See [Homebrew's tap discovery rules](https://docs.brew.sh/Taps) and the
[maintainer publication notes](DEVELOPMENT.md#homebrew-distribution).

## Check the installation

```sh
openassembler --version
openassembler --help
```

The current checked-in formula downloads the repository's `main` branch and
labels it version 0.2.0; it does not pin a released source archive or the current
local checkout. Check `Formula/openassembler.rb` before relying on a particular
compiler/table revision. Local changes take effect for new installations after
they are published to that repository.

Start the interactive shell by typing `openassembler` in your terminal. Its
logo appears on every interactive launch, followed by a workspace and command
guide. Enter `compile`, `run` or `cnr` at the prompt, and `exit` to leave.
Installation, direct commands, help, version and piped output show no logo.
An old `welcome-shown` marker has no effect and can be removed.

## An old Python environment shadows Homebrew

If `openassembler` raises `ModuleNotFoundError` from `.venv/bin/openassembler`,
your terminal selected an older Python launcher before the Homebrew executable.
The installation log may explicitly warn that the command is shadowed. In your
regular terminal, leave that environment and check the command path:

```sh
deactivate
hash -r
command -v openassembler
openassembler
```

You can also launch Homebrew's installation directly without leaving the environment:

```sh
"$(brew --prefix openassembler)/bin/openassembler"
```

For a source checkout's development environment, repair its editable installation
with `uv pip install --python .venv/bin/python -e .` from the repository root.
On macOS, Python also ignores editable-install `.pth` files marked with the
filesystem's hidden flag. If that flag is being reapplied by a local tool,
reinstalling alone will not fix imports; use the Homebrew launcher directly or
run `PYTHONPATH="$PWD" .venv/bin/openassembler` from the checkout.
Setting `HOMEBREW_NO_PATH_SHADOW_CHECK` only hides the warning; it does not change
which executable your shell selects.

The formula does not install the repository's full `exp/` evidence archive or
`testcases/` tree. Documentation links to those paths describe research in a
source checkout; use that checkout for the retained raw records and experiment
reproduction rather than expecting them in the formula installation.

## Modal and SSH

Modal is installed by the formula. Authenticate it once, then use the same
repository command for native build, execution and combined execution:

```sh
modal setup
openassembler
compile /path/to/repo --backend modal --verify
run /path/to/repo --backend modal
cnr /path/to/repo --backend modal --verify
```

For an A100 over SSH:

Inside the OpenAssembler shell:

```sh
compile /path/to/repo --backend user@a100 --verify
run /path/to/repo --backend user@a100
cnr /path/to/repo --backend user@a100 --verify
```

The remote host still needs its compatible Linux Python/CUDA/PyTorch runtime as
described in [REPOSITORY_WORKFLOW.md](REPOSITORY_WORKFLOW.md). SSH keys, ports
and jump hosts belong in `~/.ssh/config`.

Native models are selected by `[model]` configuration or automatic source
discovery. See [automatic model discovery](REPOSITORY_WORKFLOW.md#automatic-model-discovery)
for configuration-free builds and one-command overrides. `--frontend native`
selects standalone kernels explicitly. Legacy Inductor applications remain disabled.

## Native CUDA tools on macOS

NVIDIA's toolkit is Linux-only. The formula exposes the repository's verified
helper scripts as `openassembler-cuda` and `openassembler-fetch-cuda`:

```sh
brew install colima
colima start
openassembler-fetch-cuda
openassembler-cuda ptxas --version
openassembler-cuda nvdisasm --version
```

The CUDA redistribution remains governed by NVIDIA's license and is fetched by
the user. The Homebrew formula does not redistribute NVIDIA binaries. On Linux,
an existing toolkit on `PATH` or `OPENASSEMBLER_CUDA_HOME` is also supported.

## Optional PyTorch application runtime

Application capture through Modal uses the pinned PyTorch image and does not
require local Torch. SSH capture requires PyTorch 2.7.1 and CUDA 12.8 on the
remote host. Contributors running local CPU model tests should use the isolated
[development environment](DEVELOPMENT.md); this is not another user installation
path or a prerequisite for Modal.

Modal, SSH and local native paths are selected explicitly by `--backend`.
