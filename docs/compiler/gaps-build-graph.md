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
