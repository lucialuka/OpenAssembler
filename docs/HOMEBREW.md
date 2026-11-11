# Homebrew installation

The Homebrew formula installs the complete OpenAssembler source/data tree and
the `openassembler` CLI. It also installs Homebrew's Modal client, so the
documented remote workflow does not require a second Python environment.

From a checkout, install the maintained local formula with:

```sh
brew install --build-from-source ./Formula/openassembler.rb
```

Alternatively, add the repository as a tap and use its formula:

```sh
brew tap lucialuka/openassembler \
  https://github.com/lucialuka/OpenAssembler
brew install openassembler
```

The current checked-in formula downloads the repository's `CLI` branch and
labels it version 0.2.0; it does not pin a released source archive or the current
local checkout. Check `Formula/openassembler.rb` before relying on a particular
compiler/table revision. A documentation update does not publish a new formula
or benchmark the installation.

Check the installation and see every command:

```sh
openassembler --version
openassembler --help
openassembler compile --help
openassembler run --help
openassembler cnr --help
```

The formula uses Homebrew's Python 3.12 and installs the package as a
relocatable source tree. That preserves the paths used by the native encoder,
PTX parser, SASS emulator, compiler data, Modal/SSH workers, research tools and
documentation. It does not silently replace the existing optimization backend
or generated code.

The formula does not install the repository's full `exp/` evidence archive or
`testcases/` tree. Documentation links to those paths describe research in a
source checkout; use that checkout for the retained raw records and experiment
reproduction rather than expecting them in the formula installation.

## Modal and SSH

Modal is installed by the formula. Authenticate it once, then use the same
repository command for native build, execution and combined execution:

```sh
modal setup
openassembler compile /path/to/repo --backend modal --verify
openassembler run /path/to/repo --backend modal
openassembler cnr /path/to/repo --backend modal --verify
```

For an A100 over SSH:

```sh
openassembler compile /path/to/repo --backend user@a100 --verify
openassembler run /path/to/repo --backend user@a100
openassembler cnr /path/to/repo --backend user@a100 --verify
```

The remote host still needs its compatible Linux Python/CUDA/PyTorch runtime as
described in [REPOSITORY_WORKFLOW.md](REPOSITORY_WORKFLOW.md). SSH keys, ports
and jump hosts belong in `~/.ssh/config`.

For a configured PyTorch application, add `--frontend torch` to `compile` or
`cnr`; application replay selects its saved build. Native discovery remains
the default even when the repository has an application configuration.

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
remote host. A local development environment can install the explicit Python
extra separately:

```sh
python3.12 -m pip install 'openassembler[torch]'
```

The formula's `openassembler` command itself remains dependency-light; Modal,
SSH and local native paths are selected explicitly by `--backend`.
