# Repository compilation: diagnostic baseline and proposed research contract

Status: original diagnostic baseline retained; bounded entry/ABI/discovery repairs
are now implemented. Current contracts and validation are maintained in
[docs/compiler](../../docs/compiler/README.md).

Discussion date: November 15, 2026. The user accepts CPU host execution plus
OpenAssembler GPU computation. The user subsequently confirmed OA-only as the
default. The text below retains the original baseline and proposal context. Experiments for this
work belong under `compiler_exp/`, as explicitly requested by the user.
This named experiment directory is now retained in Git; generated hardware
build trees remain ignored and selected artifacts are archived separately.

## Review scope and retained evidence

The repo-wide AST inventory covers 646 Python files and 87,879 lines across
compiler, tests, tools, examples and historical experiments. All 104 compiler
modules (22,968 lines) are inventoried with hashes, module descriptions, imports
and definitions. This is a structural inventory, not a claim to have manually
reviewed every line of all experiments, generated data, binaries or dependencies.
Detailed review followed CLI selection, repository discovery, both Python
frontends, CUDA-to-PTX lowering, PTX parsing/selection/allocation, packaging,
ownership checks, CPU capture, execution, caching and transport, alongside
workflow rules, architecture/audit documents and related tests/evidence.

- `source-inventory.json`: module/file identities and structural inventory.
- `audit.py`, `baseline.json`, `baseline.log`: reproducible CPU diagnostic probes.
- `oracle_controls.py`, `oracle-controls.json`, `oracle-controls.log`: NVIDIA
  acceptance controls for missed PTX and CUDA source constructs.
- `validation.json`: the focused existing-suite command and observed outcome.

The source checkout already had uncommitted compiler, discovery, CLI,
documentation and UI changes. They were preserved and their current hashes,
HEAD and Git status are recorded. No production implementation was changed.

## Observed limits

The diagnostic harness has 18 cases: four controls accepted at their stated
stage and 14 rejected/mismatched cases. Rejected outcomes include deliberately
unsupported features, not only defects. No rate over this constructed set is a
real-world repository success-rate estimate.

| Boundary | Observation | Consequence |
|---|---|---|
| CUDA discovery | A header defines `OA_GLOBAL` as `__global__`; the `.cu` file uses it. nvcc accepts the kernel; OA classifies the source as host-only. | Textual discovery does not implement preprocessing or actual build configuration. |
| PTX entry parsing | Bare `.entry`, `.weak .entry`, and `.maxntid` forms pass ptxas but produce no OA parsed kernels. A mixed visible/bare module returns only the visible entry. | Entry completeness needs a cross-stage check; valid kernel syntax needs parser support. This probe does not assert a silently successful full build. |
| Model operations | Sigmoid, Conv2d and tensor bias broadcasting reject in CPU capture. ReLU control succeeds. | Operator vocabulary and tensor semantics remain narrow. |
| Python control flow | A branch on a tensor reduction fails FX symbolic tracing. | Static symbolic tracing alone cannot capture this application behavior. |
| Device lowering | Global atomic add, device-function call and FP64 add reject. | Accepting PTX text is separate from implementing its semantics/ABI in SASS. |
| Allocation | A real PTX load/store sequence with 260 simultaneously live scalar values rejects as out of registers. | This path has no general spill solution; legal kernels can exceed its allocation capacity. |
| CPU packages | A scikit-learn host program is identified through its import but has no compilation/execution route in the repository CLI. | CPU host preservation needs an application runner and dependency environment, not a guessed GPU translation. scikit-learn was not installed or executed by this probe. |
| Build/dependency integration | Repository code records build marker and import/include names. CUDA files compile as independent `nvcc --ptx` units. | No resolved build/link graph or general host-launch rewriting exists. |
| Models/training | Model configuration requires FP32 tensor inputs, exact shapes, MSE and plain SGD; mixed-repo kernels are compiled separately from graph operations. | Capturing a model forward does not preserve an arbitrary training program or its custom extensions. |
| Worker environment | Native model Modal image installs pinned Torch 2.7.1. | It does not resolve and reproduce an arbitrary repository's package/native dependency closure. |

The CPU Torch probes ran with locally available Torch **2.15.0**, which differs
from the pinned production 2.7.1 contract. They illustrate current source checks;
future changes need validation in the pinned worker and intended dependency
matrix. The focused suite used compiler-only Python 3.14.5 and reported
**100 passed, 2 skipped**. The two skipped cases require Torch in that interpreter.
CUDA/ptxas oracle controls used CUDA 12.8 tools through the existing wrapper.
There was no A100 execution, scheduling validation or performance measurement.

## Target behavior

`openassembler compile PATH` should build an executable application bundle:
CPU host program/runtime, resolved host dependencies, OA device cubins, launch
bindings, state/checkpoint assets, contracts and provenance. A cubin is the
device-code component, not the executable representation of host Python and I/O.

Discover an entrypoint and workload configuration when unambiguous; otherwise
request explicit flags. A path alone cannot uniquely determine which of several
training/evaluation scripts, datasets or model configurations the user intends.
Coverage must name the exact workload, phases, dependencies and guarded input
domain. Capturing observed paths does not establish coverage of every future
input or dynamic import.

For third-party packages, separately identify CPU code retained on CPU, tensor
operations lowered to OA, source-backed extensions needing compilation/linking,
and GPU binary/library calls without an OA implementation. CPU execution should
be visible in the bundle contract and included in end-to-end timing.

Proposed GPU policies, pending discussion:

| Policy | Benefit | Cost |
|---|---|---|
| OA-only (recommended default) | Every workload GPU function has an OA ownership record. | Missing GPU library/extension semantics block that workload until implemented. |
| Explicit mixed backend | Reuses available vendor kernels to broaden compatibility and retain strong existing library performance. | The resulting GPU program is partly vendor-generated and cannot be called fully OA-compiled. |

A mixed mode, if chosen, needs separate execution ledgers and reporting; it must
not weaken the existing OA-native admission checks or make a missing OA operator
look implemented. No such mode has been enabled by this discussion.

## Candidate experiments and order

Each issue gets a fixed workload contract, minimal reproducer, plausible candidate
set, negative controls, and explicit acceptance criteria before performance work.

1. **Source/entry completeness.** Compare a bounded grammar-aware PTX parser
   extension with a token-based parser; add discovered/parsed/emitted entry
   reconciliation. For CUDA, compare conservative candidate forwarding,
   preprocessing-aware discovery and build-command-driven discovery. Measure
   both false negatives and unnecessary compilation. Preserve launch metadata;
   do not just strip `.maxntid` to make the parser accept the text.
2. **Dependency/build and host execution.** Compare imported build descriptions
   (CMake File API/compilation databases), captured build commands and adapters
   for Python extension builds. Preserve include paths, macros, template
   instantiations, generated files, library identities, host/device linkage and
   launch ABI. Prefer declared metadata first and execution-backed observation
   where metadata is insufficient. Record the exact dependency closure in an
   isolated environment; editable sources/checkpoints remain intact.
3. **Framework semantic coverage.** Compare extending FX with a normalized
   exported ATen graph and a framework dispatch route. Evaluate functionalization,
   decompositions and AOTAutograd for alias/effect handling and derivatives.
   Capture method, training phase, dtype, shape/stride domain, RNG, saved tensors
   and optimizer state are explicit contracts. Generic legal OA lowerings provide
   correctness coverage; specialized recipes optimize admitted cases. Export
   tooling still needs OA implementations and does not erase control-flow limits.
4. **Custom operators and device compilation.** Connect extension registrations
   to schemas, launch parameters, shape/dtype rules, alias/mutation effects and
   backward definitions. Extend device calls/linkage, atomics/memory ordering,
   local storage/spills, types and missing instruction families based on actual
   blocker frequency. Resolve source/module symbols before native lowering;
   retained donor-container relocations are not a general OA device linker.
5. **Dynamic execution.** Compare guarded ahead-of-time shape variants, native
   conditional/loop plans and lazy OA specialization. Host Python can execute
   control flow while its GPU operations dispatch to OA. Any lazy mode must be
   explicit; strict AOT never pretends to cover unseen dynamic code. Track guard
   misses, recompilations, code size, compile latency and steady-state cost.

Do not attempt every conceivable proposal or endlessly restart an unconstrained
search. Test the documented credible candidates, refine them using evidence,
and distinguish a solved contract from unsupported inputs, a blocked experiment,
a finite unsuccessful search and a fundamental impossibility. An unavailable GPU
or a losing benchmark does not prove impossibility.

## Corpus and promotion

Start with reproducible small repositories for each boundary, then pin independent
real repositories spanning tabular CPU+GPU pipelines, CNN inference/training,
transformers with masking/dynamic lengths, and a source-backed custom extension
with cross-file device calls. Include dependency-heavy src/package layouts and
generated builds. No real repository has yet been selected or claimed supported.

For each frozen candidate:

- Compare original and compiled program outputs, input mutation/alias behavior,
  RNG progression, gradients, parameters, buffers, optimizer state and checkpoint
  resume. Keep the original precision, optimizer and numerical policy. Bitwise
  equivalence is a separate stronger contract from fixed numerical tolerances.
- Check complete source/build identities, PTX/SASS inventory, ABI, disassembly,
  A100 execution, guard regions, synchronization and GPU ownership. CPU/emulator
  passes are separate from hardware correctness.
- Measure paired original-repository and OA-incumbent baselines on the same A100.
  Include host work, transfers, packing, launches and complete training steps;
  report compilation/cold-start and steady-state separately. Use repetitions,
  uncertainty and held-out workloads. Document regression thresholds in advance.
- Promote into ordinary compiler/runtime/CLI paths only after the relevant gates
  pass. Production imports no experimental monkeypatches. Preserve failing
  candidates and reject performance claims outside the measured contract.

Universal "any repository, every input, complete AOT, and never slower" is not
a deliverable guarantee. Unknown runtime-generated code cannot be exhaustively
enumerated in a finite AOT artifact, and coverage expansion does not guarantee
competitive kernel or application performance. Exact supported-workload
contracts, explicit JIT where selected, preserved semantics and measured
regression gates are concrete deliverables.

The first proposed implementation milestone is source/entry completeness plus a
resolved application build/launch contract. This prevents adding disconnected
operators while the whole-program execution boundary remains missing.

## Reproduction

```sh
.env/bin/python compiler_exp/repository_coverage/audit.py
.venv/bin/python compiler_exp/repository_coverage/oracle_controls.py
.venv/bin/python -m pytest -q \
  tests/test_repository_frontend_selection.py tests/test_repo_workflow.py \
  tests/test_model_discovery.py tests/test_cppfront.py tests/test_native_ownership.py
```

`.env` is a local CPU Torch environment, not a portable dependency declaration.
Without Torch, the audit skips its optional model probes and records that fact.
NVIDIA oracle controls need the toolkit and, on macOS, the existing Colima setup.

Primary background: [CUDA compilation and separate device linking](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda),
[scikit-learn GPU/Array API scope](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).
