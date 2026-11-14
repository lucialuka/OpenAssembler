# Repository discovery, manifests and safe artifact publication

Status: implemented

The CLI builds discovered standalone GPU kernels or an explicitly selected
native Python model while preserving an explicit record of what was compiled
and by whom. It does not turn arbitrary host code into a binary.

## Idea and behavior

A directory build should remain understandable: discover genuine GPU sources,
keep their relative paths, avoid output collisions and refuse stale/partial
artifacts. In a Git checkout, discovery follows tracked and non-ignored files;
otherwise it walks supported files. Hidden/generated trees, environments,
symlinks and host-only files are not blindly compiled.

Standalone discovery recognizes PTX entries, CUDA `__global__` kernels and the
Python frontend's `@kernel` functions. Same-stem source collisions get stable
distinct output names. Directory builds mirror source paths and do not accept
a single-file `-o` name. A root `[model]` configuration automatically selects
[native OA model capture](native-model-bridge.md); `--frontend native` forces
kernel discovery. `model` and the compatibility alias `torch` select OA model
capture. `--frontend triton`, `--frontend inductor`, and legacy `[application]`
GPU execution are disabled.

The output manifest records source identity, kernels, backend attribution,
paths, resource counts and verification state. Each output directory has a
writer lock. Publication uses atomic file replacements, and corrupt manifests
are rejected before overwriting recorded outputs. A failed recompilation does
not silently authorize old successful binaries as current. Deleted/disappeared
sources have their manifest-owned artifacts retired by normal build logic.
Unlisted user files are preserved. Cleanup of no-longer-discovered sources occurs
during a subsequent nonempty directory build. Individual atomic replacements
and a writer lock do not make a multi-file transaction under power loss;
rebuild after interrupted publication before trusting the output.

Explicit files always use their source frontend, even inside an application
directory. Missing files do not accidentally trigger training capture.
Application-only and kernel-only options are checked before execution, and
source/output ownership collisions or unsafe output overlap fail early.

Repository commands now share the same source path: `compile REPO`, `run REPO`
and `cnr REPO`. The conventional build root is `REPO/output`; explicit build
roots and legacy working-directory outputs remain supported. Source-based
execution checks provenance and restricts selection to that repository.
Compilation and execution both accept local, SSH and Modal destinations. Native
remote builds stage the installed compiler/data and rebase returned manifest
sources to the local tree; they retain the existing code-generation pipeline.
See [repository workflow](../../REPOSITORY_WORKFLOW.md) for setup and migration.

## Methodology and findings

Tests exercise nested paths, extension/case collisions, source selection,
ignored/generated directories, locking, interruption/publication, manifest
corruption and stale-output refusal. The full integration matrix covers
language routes and actual CLI invocation. These are correctness/workflow
features; path mirroring does not itself accelerate GPU instructions.

## Reproduce

Run from the repository root with an installed Python 3.12+ CLI and CUDA 12.8
toolkit. Repository-native compilation needs ptxas/nvdisasm; CUDA files also
need nvcc and its host compiler. Actual execution needs a GPU.

```sh
openassembler compile examples/mlp --out-dir output_repository
openassembler run output_repository --backend modal --dry-run
openassembler inspect examples/native_attention_model --json
openassembler cnr examples/native_attention_model --backend modal --verify --no-sass
.venv/bin/python -m pytest -q tests/test_cli_compile.py tests/test_runner.py \
  tests/test_integration.py tests/test_torchfront.py tests/test_repo_workflow.py
openassembler benchmark --help
```

The first `run --dry-run` inspects proposed launches; it does not execute or
validate the whole MLP. Real multi-kernel workloads need a correct `--spec`
or configured application. `benchmark --help` exposes the separate integration
matrix rather than guessing its options; see [COMMANDS.md](../../COMMANDS.md).

## Source and evidence

- [Discovery](../../../openassembler/frontend.py),
  [paths/manifests/locking](../../../openassembler/project.py),
  [build/publication](../../../openassembler/build.py),
  [run validation](../../../openassembler/runner.py),
  [source/build resolution](../../../openassembler/workflow.py),
  [native remote builds](../../../openassembler/remote_build.py).
- [CLI build regressions](../../../tests/test_cli_compile.py),
  [integration tests](../../../tests/test_integration.py),
  [integration evidence](../../evidence/integration-20261104.json).

Source checksum changes and new workload arguments require recompilation or
recapture. A readable `.sass` belongs to the corresponding cubin, not to an
independently trusted source listing. Exact output layout and all selectors
are maintained in the single command reference.
