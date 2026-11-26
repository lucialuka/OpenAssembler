# Dependencies and host requirements

Requirements depend on the operation. The core package has no mandatory
third-party Python runtime dependency, but native compilation and GPU execution
use NVIDIA tools and runtime components.

| Task | Environment | GPU required? |
|---|---|---|
| Install or inspect CLI | Python 3.12+ from supported Homebrew formula | No |
| Inspect source or dry-run | Supported Python runtime | No |
| Compile PTX or typed Python locally | ptxas and nvdisasm through CUDA toolkit | No |
| Compile CUDA C/C++ locally | Above plus nvcc, toolkit headers and compatible host C++ compiler | No |
| Compile through Modal or SSH | Remote native tools; Modal account or SSH access | No for native build |
| Execute standalone cubin | Linux NVIDIA driver compatible with SM80 cubin | Yes |
| Capture native model | CPU PyTorch in selected worker; Modal uses configured image, SSH/local need documented runtime | No for capture |
| Execute or verify native model | Compatible PyTorch reference environment and NVIDIA GPU/driver | Yes |
| Run repository test suite | pytest; toolkit for toolkit-dependent scopes | Hardware tests only |

On macOS, CUDA tools run in the Colima Linux VM; macOS cannot load an NVIDIA
cubin. On Linux, tools may come from PATH or OPENASSEMBLER_CUDA_HOME. Setup is
documented in [DEVELOPMENT.md](../DEVELOPMENT.md) and
[HOMEBREW.md](../HOMEBREW.md); transport setup is in
[REPOSITORY_WORKFLOW.md](../REPOSITORY_WORKFLOW.md).

Installing PyTorch or CUDA does not make unsupported graph or instruction
contracts admissible.
