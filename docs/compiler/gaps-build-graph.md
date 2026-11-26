# Recorded NVCC build commands and dependency identities

Status: implemented

G42 identified that discovering source names does not preserve original build
settings or generated inputs. The selected route consumes an existing
`compile_commands.json`, using its recorded per-translation-unit settings and
working directory, without executing the original host command or generator.
Source-root databases are selected automatically for native builds;
`--compile-commands FILE` selects another recorded configuration. This is a
concrete build integration route, with the remaining original G42 exit gates
kept explicit.

## Contract and original-command comparison

The [database resolver](../../openassembler/compilation_database.py) reads the
[Clang compilation database format](https://clang.llvm.org/docs/JSONCompilationDatabase.html):
file, directory and an argument vector, or a plain command string. Argument
vectors are preferred. The optional output field does not authorize executing
host output/link stages. Commands must select NVCC and SM80; conflicting variants
for one source reject. Non-CUDA host compiler records are ignored, while CUDA
commands from another compiler reject. Shell commands/expansions reject;
literal dollar characters remain possible in an explicit argument vector.

The resolver expands bounded response/`--options-file` inputs and fingerprints
them. It preserves per-unit definitions, undefinitions, precision/standard and
compiler options, include/forced-header order, original working directory and
exact source-argument spelling. OA replaces the object output stage with
`nvcc --ptx`; its native backend then compiles every emitted entry. The original
source, arguments, normalized device flags, database/response hashes, toolchain
identity and actual preprocessed dependency hashes are retained in each
manifest entry's `build_contract`.

`nvcc -M` resolves the actual dependency graph using the same settings. Hashing
runs in the tool environment: system headers inside Colima must not accidentally
be hashed from macOS. Missing generated inputs reject with a direct diagnostic.
Recorded generated translation units participate even when ignored by Git;
inspection and dry runs remain static and execute neither NVCC nor host code.

The original NVCC command is independently compiled in regression controls by
replacing only `-c` and its output path with the PTX stage. Complete parsed PTX
instruction bodies match OA's path. `__FILE__` controls also check argument
spelling, relative includes and remote relocation, rather than assuming an
absolute-path normalization is semantically harmless.

## Remote relocation

The [native transport](../../openassembler/remote_build.py) stages generated
translation units, response sources, explicit include trees and recursively
resolved quoted sibling headers. Output/virtual-environment directories are
excluded; existing bundle file/count limits apply. Forced headers and working
directories retain distinct per-unit bindings. Database paths are rebased in
the staged tree, while `submitted_origin` retains the user's recorded contract.
NVCC resolves/hashes the actual dependency graph again in the pinned worker.

When relocating source/include spellings changes file macros, tested compiler
file-prefix maps restore original spellings. Specific source maps override
broader include maps. Conflicting aliases, uncomposable existing prefix maps,
unrepresentable map characters and remote host-compiler overrides reject.
Source aliases must be materialized for remote relocation. Arbitrary path-bearing
options outside the normalized vocabulary can still fail in the worker; they
are never silently dropped or converted into a fallback build.

## Research cycle and retained failures

The [candidate](../../compiler_exp/gap_closure/build_graph/candidate.py) defined
per-source settings/path preservation before production integration. Its initial
macOS `/tmp` spelling assertion failure and corrected result remain retained.
Actual NVCC tests then caught system dependency hashing in the wrong host
filesystem; the selected implementation hashes inside the toolkit environment.

Independent review retained failures for an ignored generated source's omitted
quoted header, shell-variable command text, normalization of `./` in source
arguments, and overlapping source/include file-prefix mappings. These are
preserved as `reviewer-*.json` in the
[experiment directory](../../compiler_exp/gap_closure/build_graph/). Production
uses raw source/path tokens, explicit expansion rejection, quoted dependency
closure and ordered prefix maps; dedicated tests cover each counterexample.
No failed record is relabeled as a successful build.

The maintained [regressions](../../tests/test_compilation_database.py) include
actual NVCC original-command/PTX controls, generated discovery and relocation,
response hashes, conflicting build variants, mutation-free dry planning,
implicit headers, shell controls and file-macro equivalence. The dossier's final
integration evidence is recorded beside those source-defined controls.

## Reproduction and remaining work

```sh
python -m pytest -q tests/test_compilation_database.py tests/test_cppfront.py tests/test_repo_workflow.py
openassembler inspect /path/to/repository --json
openassembler compile /path/to/repository --compile-commands /path/to/build/compile_commands.json --verify
modal run compiler_exp/gap_closure/build_graph/run_modal.py
```

CMake/setuptools/source generators must first produce their recorded database and
inputs. General build-command execution, host/native library linkage, device
linking/relocations, arbitrary compiler wrappers, dependency installation and
complete package/environment reproduction remain open. A source may compile
correctly yet still need explicit host launch bindings or a model custom-op
contract. This work does not execute arbitrary host orchestration or establish
whole-application equivalence/performance.

## Explicit scalar device source libraries

The [source-library route](gaps-device-functions.md#explicit-ptx-and-cuda-source-libraries)
adds repeatable `--device-library` PTX/CUDA inputs. Each CUDA translation unit
keeps its selected compilation-database flags, source spelling/working directory
and before/after tool-environment dependency hashes. Remote transport bundles
external library roots and their selected command/include inputs. Root entries
and selected helpers resolve through exact signature/visibility rules before
static OA inlining; kernel ownership records the link contract. This is not an
original host build/link executor or a native object/archive linker.

Original separately compiled PTX/CUDA objects are linked independently with
nvlink for the **576-launch A100 control gate**. Normal CLI and remote-worker
builds pass complete buffers/resources; an additional ordinary Modal CLI run
checks client publication of an external library. General library/package/build
resolution and native relocations remain open under the original G42 gate.

## Static Python helper source graphs

The [local-package helper route](gaps-python.md#static-local-package-device-helpers)
adds a concrete Python source/dependency build graph. Entry/helper/package source
hashes, defining-module annotations, body/signature/call records and original
diagnostics flow through ordinary local/Modal/SSH builds. Reachable ignored
helpers are included in remote bundles; explicit or inferred package roots and
returned entry paths stay consistent. Changes during compilation prevent artifact
publication. Native ownership snapshots the graph, with source-exact A100, worker,
public CLI and standalone regression evidence in the
[promotion cycle](../../compiler_exp/gap_closure/python_module_helpers/promotion/README.md).
General host build orchestration, installed/runtime import hooks, generated Python
sources and mutable captures remain separate G42/G04 boundaries.

The [explicit static-value extension](gaps-static-values.md) resolves Final
numeric/tuple declarations through that same original graph. Dependency hashes
remain original-source hashes, and exact used values/type bits/per-kernel uses
are included in ownership. Constant-only kernels also retain source identity.
The compiler does not replace an original graph with a normalized-file identity
or infer arbitrary mutable host bindings as static values.
