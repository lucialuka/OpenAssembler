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

The SM80 encoder table is a required compiler input, included as a verified
formula resource. `brew test lucialuka/openassembler/openassembler` checks that
the installed encoder loads and encodes an instruction without needing CUDA.

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

The formula pins a source commit and verifies its SHA-256 checksum. This avoids
reusing an older cached `main` archive when the formula changes. Check
`Formula/openassembler.rb` for the installed compiler revision. Maintainers must
update the commit, checksum and package version together when publishing an update.
Local working-tree changes reach other users after publication and a formula update.

Start the interactive shell by typing `openassembler` in your terminal. Its
logo appears on every interactive launch, followed by a workspace and command
guide. Enter `compile`, `run` or `cnr` at the prompt, and `exit` to leave.
Installation, direct commands, help, version and piped output show no logo.
An old `welcome-shown` marker has no effect and can be removed.

## Use your own ML repository

Open a terminal in your repository and type:

```sh
openassembler
```

The shell keeps that directory as its workspace. It uses Homebrew's Python and
bundled compiler data, so no checkout of OpenAssembler, editable install, or
activation of an OpenAssembler Python environment is needed. Your ML project's
virtual environment can stay active. At the OpenAssembler prompt:

```text
inspect --json
compile --backend modal --verify
run --backend modal
exit
```

Omitting the source path selects the current repository. Standalone kernels may
need a launch specification for `run`; model repositories retain their launch
plan. Modal authentication (`modal setup`) is a one-time setup in your regular
terminal. SSH and local execution retain their documented host requirements.

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

If you keep multiple Python installations with their own `openassembler` command,
you can make the Homebrew command take precedence without deactivating your ML
environment. Add this once to your shell startup file (`~/.zshrc` for zsh or
`~/.bashrc` for Bash), replacing the prefix for your Homebrew installation:

```sh
alias openassembler='/opt/homebrew/opt/openassembler/bin/openassembler'
```

Apple Silicon Homebrew normally uses `/opt/homebrew`; Intel macOS uses
`/usr/local`, and Linux commonly uses `/home/linuxbrew/.linuxbrew`. The stable
`opt` path continues to resolve after Homebrew upgrades.

If `brew` reports the formula in both the old `qingyang-lucas-fang/openassembler`
tap and `lucialuka/openassembler`, and your installed package belongs to the
latter, remove the unused tap with `brew untap qingyang-lucas-fang/openassembler`.
Use `brew info lucialuka/openassembler/openassembler` to check its source.

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

The formula includes Modal's optional CONNECT/SOCKS proxy helper, so its
subprocess can use macOS system proxies or standard proxy environment variables.
Application capture through Modal uses the pinned PyTorch image and does not
require local Torch. SSH capture requires PyTorch 2.7.1 and CUDA 12.8 on the
remote host. Contributors running local CPU model tests should use the isolated
[development environment](DEVELOPMENT.md); this is not another user installation
path or a prerequisite for Modal.

Modal, SSH and local native paths are selected explicitly by `--backend`.
