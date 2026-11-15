# Source completeness, parameter ABI and launch constraints

Status: implemented

This is a bounded source-discovery/parser/integrity repair. It does not implement
whole-program linking or arbitrary Python application capture.

## Failure and candidate selection

The [initial baseline](../../compiler_exp/repository_coverage/baseline.json)
found PTX accepted by CUDA 12.8 ptxas but omitted by OA's old entry regex:
bare `.entry`, `.weak .entry`, and entries with `.maxntid` before their body.
A mixed module returned only its visible entry. The
[oracle control](../../compiler_exp/repository_coverage/oracle-controls.json)
also confirmed that nvcc accepts a `.cu` kernel whose `__global__` qualifier is
supplied by a local header macro, while old static discovery misses it.

The [parser candidate experiment](../../compiler_exp/repository_coverage/parser_candidates.py)
compared the existing parser, a wider regex and a balanced header scanner using
nine positive/negative fixtures. The regex extension still mistakes a quoted
filename for an entry. The selected scanner skips strings/comments, balances
the parameter list, distinguishes prototypes from definitions, and captures
the complete header before parsing the body. Unknown headers reject rather
than silently removing the entry. Before/after outcomes are retained in
[before](../../compiler_exp/repository_coverage/parser-candidates-before.json)
and [after](../../compiler_exp/repository_coverage/parser-candidates-after.json).

The supported parameter ABI is explicit scalar 32/64-bit integer/bit/FP
parameters. Aggregate arrays, pointer annotations and other parameter grammar
outside this subset reject before code generation. Duplicate definitions and
parameter names reject. Frontend/parsed entry sets must agree before lowering;
the existing executable ownership check covers every emitted function.

## Retained directives and runtime checks

The parser records `.maxntid`, `.reqntid`, `.maxnreg`, `.minnctapersm` and
`.maxnctapersm` in each kernel. Original PTX goes to ptxas intact, so entry
metadata is not obtained by deleting launch directives. The manifest records
`ptx_directives`; the native ownership inventory binds these values alongside
the executable bytes, registers and parameter ABI. Existing manifests without
directives keep their existing identity representation.

- `.reqntid` requires the exact three block dimensions. Explicit launch specs
  are checked before submission. Automatic runs use required one-dimensional
  blocks; multidimensional requirements need an explicit spec.
- `.maxntid` limits the product of block dimensions, as defined by PTX. It does
  not impose an independent limit on each axis. Automatic launches obey it.
- `.maxnreg` is checked against the final OA register declaration. Kernels
  needing more registers reject; this does not implement spills.
- CTA residency directives remain recorded tuning hints. They do not promise
  OA achieves ptxas's register allocation, occupancy or speed. No new schedule,
  shared-memory staging policy or asynchronous dependency is introduced here.

Generic functional verification uses the required block geometry or maximum
thread bound. CUDA device limits and other validity checks still apply through
the normal build/launch paths. The relevant production code is
[parser](../../openassembler/ptx/parse.py), [build](../../openassembler/build.py),
[ownership](../../openassembler/native_ownership.py) and
[launch planning](../../openassembler/runner.py).

## CUDA discovery tradeoff

Always forwarding all CUDA-looking files would compile host-only files and
qualifier headers as separate translation units. Invoking preprocessing during
`inspect` would change its tool-free, nonexecuting contract. The selected bounded
static candidate pass follows quoted local includes looking for a device
qualifier. Macro-only `.cuh` definitions are not independent kernels.

The traversal stays inside the inspected tree, skips symlinks, tracks cycles,
and is capped at 128 files/8 MiB per source. It resolves relative includes and
root-relative quoted includes. It does not evaluate conditionals, system
include paths, command-line definitions or generated build graphs. It can
overselect a source whose included qualifier is unused; nvcc remains the
authority, and compilation fails explicitly if no kernel is produced.
Sources beyond this bounded discovery need explicit source/include/flag
selection. Build-command-based discovery is a separate roadmap item.

## Validation and reproduction

[Regression tests](../../tests/test_ptx_entries.py) cover definitions/prototypes,
quoted fake entries, `$` symbols, unsupported ABI/directives, duplicates,
inventory mismatch before codegen, normal CLI compilation, header cycles,
tree boundaries, native metadata integrity, automatic block geometry and
explicit-spec rejection. Toolkit tests build actual mixed-entry cubins and
disassemble them through the ordinary compiler.

The A100 entry regression builds OA-native bare/weak/required-block kernels,
checks full integer outputs on three CTAs and surrounding guard bytes with
three poison patterns, and records actual device resources. Raw results and
the experiment runner are retained under `compiler_exp/repository_coverage/`
and `compiler_exp/latency_hiding/`. This establishes those entry/launch cases,
not arbitrary GPU operations or a performance improvement.

The [A100 result](../../compiler_exp/repository_coverage/a100-entries-20261115T051000Z.json)
passes all nine checks and 2,304 integer output elements, with guards intact.
The actual device is an A100-SXM4-80GB. The
[continuity audit](../../exp/native_indexing/dispatch-continuity-20261115T051928Z.json)
also verifies unchanged native words, scheduling controls and register counts
for all 500 original GEMM candidate contracts across 1,000 retained session
checks, 40 rowwise checks and six fixed tensor goldens. It refreshes dispatch
provenance only after checking the retained binaries; it is not new timing.

The [CUDA discovery comparison](../../compiler_exp/repository_coverage/discovery-candidates.json)
confirms the selected static closure discovers the macro fixture without a tool
invocation; both forwarding all sources and the closure can overselect unused
macros. nvcc-based discovery distinguishes all four fixtures but incurs actual
frontend work and would violate static inspection's nonexecuting contract.

```sh
.venv/bin/python -m pytest -q tests/test_ptx_entries.py
.venv/bin/python compiler_exp/repository_coverage/parser_candidates.py
modal run compiler_exp/latency_hiding/run_modal.py --action entries
```

Use the recorded compiler/test environment and CUDA toolkit as described in
[development](../DEVELOPMENT.md). Header fixtures on macOS must be visible to
the Colima toolkit. Original baseline probes used local Torch 2.15.0, outside
the production 2.7.1 model contract; this parser repair makes no new model claim.

Primary specification: [PTX entry performance directives](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives).
