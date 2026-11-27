# Troubleshoot builds and execution

CLI examples below run inside the shell started with `openassembler`; enter
commands such as `compile`, `run` and `cnr` without repeating the executable name.
Homebrew, Python, Modal and SSH setup commands run in your regular terminal.

Start by inspecting the source repository and selected frontend:

    inspect /path/to/repo --json

Inspection reads source but does not import user code or prove that each
operation will compile.

| Symptom | What to check |
|---|---|
| ModuleNotFoundError after brew install | A stale Python launcher may shadow Homebrew. In your regular terminal, use deactivate, then launch openassembler again. See [Homebrew PATH troubleshooting](../HOMEBREW.md#an-old-python-environment-shadows-homebrew). |
| zsh: bad pattern with --backend | Pass one value such as --backend modal; do not paste a bracketed list. Start openassembler, then enter compile/run/cnr without the executable prefix. |
| Unsupported Python/PTX operation | Read the source location in the diagnostic. Discovery or parser acceptance does not promise native lowering. Check the [compiler contracts](../compiler/README.md); there is no vendor fallback. |
| ptxas, nvdisasm or nvcc not found | Local builds need toolkit tools; CUDA C/C++ also needs nvcc. On macOS, start Colima and use the installed CUDA wrapper. For remote builds, test tool paths on the selected host. See [dependencies](../architecture/dependencies.md). |
| Modal command fails before build | Run modal setup and check the authenticated account. A dry-run inspects a plan; it does not validate credentials or submit a build. |
| SSH works interactively but build fails | Test noninteractive access and PATH with ssh -o BatchMode=yes. Check Python, toolkit tools, driver and model runtime on that same host. See [SSH setup](../getting-started/modal-setup.md#ssh-to-an-a100). |
| run says build is missing | run loads existing artifacts and never compiles. Compile first and pass the same --out-dir to build and run when using a custom path. |
| run warns that source changed | Recompile to use edited native source. A saved model replay uses its bundle; it does not recapture edits from the local tree. |
| Kernel argument or launch mismatch | Compare entry names, parameter order, dtypes and launch dimensions with manifest.json. Check buffer sizes, argument bindings, grid and block in the run spec. See [launch specifications](../COMMANDS.md#describe-an-actual-launch-with---spec). |
| Build fails and more detail is needed | Preserve the output directory. Read compile-error.json, compiler output and any build-errors.json there; include the source revision and a minimal reproducer in a bug report. |

Use [command semantics](../COMMANDS.md) for command-specific exit codes,
verification behavior and output locations. Use the linked feature contract for
the exact supported shape or instruction boundary.
