# Repository compiler and codebase audit

Reviewed November 14, 2026. OpenAssembler's core is an SM80/A100 compiler from
supported Python model/kernel, CUDA C/C++/C, and PTX sources to OA-owned native
instructions in executable cubins. Transport changes where work runs; it does
not authorize Triton, Inductor, cuBLAS, or donor cubins as production compute.

## Source to executable map

| Boundary | Owner and behavior |
|---|---|
| CLI and repository coverage | `cli.py`, `repository.py`, `diagnostics.py`: static inspection, automatic `[model]` selection or native kernel discovery, path shorthand, local error reports |
| Typed Python GPU kernels | `pyfront/`: the supported `@kernel` language lowers to PTX |
| CUDA C/C++/C GPU kernels | `cppfront.py`: nvcc parses device sources into PTX; host-only C/C++ is not inferred GPU code |
| Python models | `modelfront/graph.py`: CPU FX capture, exact shapes, symbolic backward; `kernels*.py` lower supported operators into OA PTX recipes |
| Native compilation | `backend/`, `opt/`, `isa/`: instruction selection, allocation, scheduling, native encoding; NVIDIA tools supply oracle/reference information and ELF containers |
| Packaging and provenance | `container/`, `build.py`, `native_ownership.py`, model ledgers: executable text is checked against OA output |
| Reusable execution | `modelfront/runtime.py`, `replay.py`: fixed storage, guarded tapes/workspaces, OA-only driver launches, full training replay |
| Local / SSH / Modal | `transport.py`, `remote_build.py`, model workers: the same compiler/data/artifact contracts are staged and checked |

Discovery is not whole-program linking. A configured model repo can contain
standalone native kernel sources, which are compiled separately; arbitrary custom
operators are not automatically inserted into its tensor graph. General host
launch rewriting, CMake linking, dynamic Python control flow, arbitrary framework
operators, all dtypes, and architectures beyond SM80 remain outside coverage.
Failures should identify the missing operation or source rather than silently
execute a different compiler's output. General superiority to NVIDIA is a goal,
not an established property of these bounded implementations.

## FP16 BMM and attention

The [experiment record](../exp/attention_half/RESULTS.md) covers both matrix
orientations in backward, batch/tail shapes, exact operand rounding, changing
inputs, poisoned workspaces, guards, profiler ownership and complete training.
Equal flattened batch counts through 32 and positive M/N/K through 512 are
supported. Larger products use dual padded FP16 layouts and CTA-z strides;
products with all three dimensions at most 64 use explicit FP16-rounded FP32
workspaces and scalar BMM. Softmax, layouts, master weights and updates remain
FP32. This is composed unmasked attention. Broadcasting, masks and fused
attention remain unsupported.

## Code and artifact cleanup

- Removed the unused legacy `_historical_execute_program`, `_historical_process`
  and child-process implementation from `torchfront/worker.py`. Kept its disabled
  compatibility guards and CPU state comparison utilities, which still have tests.
- Removed unused `torchfront/instrument.py` and `torchfront/modal_app.py` after
  checking source imports, CLI dispatch, research tools, tests and documentation.
  The historical application dossier now labels those results as retired.
- Kept archive/source-bundling helpers and compatibility APIs used by the native
  frontend and tests. Unselected experiments are evidence, not automatically dead.
- Ignored all generated `*.sass` files and removed 274 listings from the Git
  index with `git rm --cached`; their local files were preserved. Listings can
  be regenerated from retained cubins with `nvdisasm`. PTX input programs and
  recorded cubins, logs, hashes and experiment archives remain evidence.
- Made normal native-model CLI output concise. Full verification, launch ledgers,
  profiling and timings remain in `model-build-report.json` and `model-run.json`.
- Audited active documentation links, command flags, catalogue statuses and
  contradictory frontend/precision statements. Historical numerical results are
  preserved with their original attribution and failed gates.

## Reproduce or report a failure

```sh
openassembler inspect /path/to/repo --json
openassembler /path/to/repo --backend modal --verify --no-sass
openassembler cnr /path/to/repo --backend user@a100 --verify
openassembler run /path/to/repo --backend user@a100
```

`compile` and `cnr` default to the current directory. On a failed build, inspect
`OUT/compile-error.json`, compiler output and any `build-errors.json`; include
these and a minimal reproducer when filing a bug. No external issue is submitted.
Dry runs, inspection and disabled vendor-frontend rejection create no build
outputs. Reports collect version/platform, file names/coverage and error text,
not source files or environment values; error text can include source excerpts.

Run the regression suite and documentation checker as described in
[DEVELOPMENT.md](DEVELOPMENT.md). Hardware results and local tests have separate
scopes. Installed GEMM/rowwise dispatch identities are refreshed only after the
retained instruction words, controls and register counts pass the continuity
audit; a diagnostic-only compiler edit does not imply new performance evidence.
