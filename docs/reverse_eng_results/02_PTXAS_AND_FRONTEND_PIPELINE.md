# ptxas and frontend pipeline

## Native compilation boundary

The native path is:

```text
PTX / CUDA C++ / typed Python @kernel
        ↓
frontend to PTX
        ↓
PTX parser and contract checks
        ↓
scalar or SM80 tensor selector
        ↓
virtual values, DCE, copy coalescing and allocation
        ↓
SASS text plus control fields
        ↓
OpenAssembler encoding
        ↓
ptxas-built ELF container + selected text replacement
        ↓
cubin, SASS listing and manifest
```

ptxas remains necessary for the reference cubin/container; `nvcc` supplies CUDA
C/C++ parsing and PTX generation. The reference is built from the same PTX that OpenAssembler
receives. This keeps comparisons focused on backend code generation rather than
different frontend semantics.

The November 23 foundation dossiers add
[bounded atomic ordering/shared operations](../compiler/gaps-atomics.md) and
[full-warp indexed/up/down shuffles](../compiler/gaps-collectives.md) using
already recovered templates. Source-exact A100 byte/state gates and separate
native continuity audits precede admission. Unsupported modifiers, participation
and operand domains still reject; successful encoding alone is not hardware
semantic evidence. [Explicit decode specialization](../compiler/gaps-decode.md)
uses the same ordinary OA build/runtime ownership route for new exact lengths.
The [native local-storage route](../compiler/gaps-storage.md) additionally
patches OA-owned per-function frame reservations after donor optimization.
Actual A100 local-resource attributes and complete per-thread outputs validate
that layout; an optimized donor's zero frame cannot be reused implicitly.

The [floating literal route](../compiler/gaps-wide.md#decimal-and-raw-floating-literal-contexts)
uses binary64 decimal representation before narrowing at scalar use. Local
ptxas V12.8.61 contexts and independent CUDA 12.8.1/A100 byte gates separate
decimal rounding from raw-bit width behavior, including NaN payloads and
zero-extended raw32 f64 moves/stores. The parser changes no recovered templates.

The [shared-width extension](../compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory)
recovers four exact narrow-store forms through the existing learner and loads
them as maintained oracle data. Varied disassembly, failed and repaired barrier
exchanges, and complete A100 output/resource gates precede normal admission.

The [direct scalar device-function route](../compiler/gaps-device-functions.md)
resolves module-local calls through static specialization, preserving lexical
slots, copied arguments, returns and local resources. The final original-source
A100 gate passes 960 launches, including ordinary CUDA noinline input. Native
call stacks, aggregate/register ABI and cross-module linking remain open.

The [explicit source-library extension](../compiler/gaps-device-functions.md#explicit-ptx-and-cuda-source-libraries)
adds deterministic scalar PTX/CUDA symbol resolution and per-unit dependency
identity through normal CLI and remote builds. Its 576-launch A100 gate keeps
original separately linked controls distinct from merged-source controls.
Native objects/archives, dynamic stacks and complete host linking remain open.

## Checked generated-kernel dispatch

The `gemm` and `rowwise` commands validate a declared operation before emitting
PTX. Their automatic selectors use exact measured contracts and source
identities: the current GEMM table contains 34 tuned choices and eight measured
baseline retentions, while the FP32 softmax/LayerNorm table contains five
geometry choices. They select OpenAssembler-generated policies, not a hidden
ptxas/cuBLAS fallback. An unknown, ambiguous or stale contract keeps its
representable reference plan. Explicit implementation options bypass measured
dispatch, and `gemm --autotune local|modal` is the opt-in path that performs
GPU search. Ordinary compilation does not launch a tuning job.

The selected annotated PTX and manifest retain the actual tile, pointer/CTA
policy, launch geometry and numerical contract so recompilation and execution
do not silently change the policy. See [GEMM dispatch](../research/ideas/shape-aware-native-gemm-dispatch.md)
and [rowwise dispatch](../research/ideas/shape-aware-rowwise-dispatch.md).

## Frontend selection

Repository inspection is static. A configured `[model]` selects the OA-native
CPU graph frontend; otherwise automatic selection discovers standalone kernels.
`--frontend native` forces kernel discovery. `model` and `torch` select native
model capture; Triton/Inductor GPU frontends and old application replay reject.
The [repository audit](../CODEBASE_AUDIT.md) records current boundaries.

This boundary matters for reverse engineering: a Triton or Inductor cubin is
evidence about that frontend/runtime, not evidence that OpenAssembler generated
the kernel.

## Fallback policy

Unsupported PTX forms fail with a source-level explanation and a build failure
report. The retired `--fallback ptxas` flag rejects. Ptxas remains an ELF donor
and explicit research reference; production launches only OA-owned executable
text. Historical vendor comparisons do not count as OA optimization results.

## Reference implementation

The main code paths are:

* `openassembler/frontend.py` and `openassembler/pyfront/` — source to PTX.
* `openassembler/ptx/parse.py` and `openassembler/ptx/interp.py` — parsing and
  reference semantics.
* `openassembler/backend/ra.py` — scalar/tensor selection, DCE, allocation and
  lowering.
* `openassembler/backend/compile.py` and `openassembler/backend/tensor.py` —
  dependency controls and tensor scheduling.
* `openassembler/build.py` — source compilation, ptxas reference build,
  splicing, disassembly and manifest publication.

The production build now compiles independent repository sources with bounded
parallelism while preserving candidate order and output determinism. That is a
host build optimization; it does not alter generated SASS.


## Local-package Python helper source graphs

The [helper source-graph dossier](../compiler/gaps-python.md#static-local-package-device-helpers)
records an extension at the Python frontend boundary. Helpers retain their
original module annotation/intrinsic bindings and source diagnostics, then use
existing pure scalar specialization. No new ISA field or device-call ABI is
inferred. Entry/helper/package file hashes, body/signature/call identities,
ignored reachable dependencies and explicit package roots flow through the normal
CLI, worker and native ownership. Independent original-tree A100 gates retain
600 launches, 5,529,600 complete bytes and sixty exact native outputs, with
separate failed integration and corrected frontend-routing records. This is
source and correctness coverage, with no general package/host-runtime claim.

## Explicit static numeric and tuple source values

The [static-value dossier](../compiler/gaps-static-values.md) extends the original
Python source graph with bounded Final declarations and exact source/value/use
identity. Finite arithmetic, tuple indices/slices/len and defining-module imports
specialize before ordinary device type/effect checks. No new ISA encoding is
inferred. The original-source A100 gate passes 1,440 launches, 9,338,880 complete
bytes and 144 native outputs; the earlier normalized prototype and its rebinding
counterexample remain distinct. These checks do not establish arbitrary Python
host/metaprogramming semantics or a performance improvement.

## Finite numeric tuple iteration

The [tuple-loop dossier](../compiler/gaps-static-loops.md) records an original-source
control extension. A runtime counter selects finite scalar values, executes one
emitted body and uses explicit continue/exhaustion/break labels. Empty tuples,
observable targets, nested loops and loop else retain their type/control/effect
contracts. No backend encoding template changed. The original-source A100 gate
passes 2,400 launches, 16,465,920 independent bytes and 240 exact native outputs;
first candidate type and fixture/harness failures remain retained. This is not a
performance result or arbitrary runtime iterable implementation.

The [pure helper return-path cycle](../compiler/gaps-helper-returns.md) now preserves
original branch/loop returns, caller snapshots, live-path initialization and exact
source/native ownership. Its final gate passes 2,400 A100 launches; broader
Python effects remain open; the subsequent runtime loop-control cycle below
now admits range/while else under its separate validated contract.

The [finite runtime loop-control cycle](../compiler/gaps-range-control.md#original-source-production-extension)
now preserves range/while else and exhausts finite i32 ranges before counter wrap.
Its original-source gate passes 2,400 A100 launches; dynamic steps, wider bounds
and general control/effects retain separate open contracts.

The [typed condition-chain cycle](../compiler/gaps-comparison-chains.md) preserves
single-evaluation/short-circuit memory and atomic effects, pairwise exact literal
types, special-value truth and original expression/source/native ownership.
Its final gate passes 2,400 A100 launches; general Boolean values and arbitrary
Python effects remain open.

The [Boolean-value cycle](../compiler/gaps-boolean-values.md) preserves lazy operand
selection, contextual literal types, explicit truth/conversion and original value
ownership. A100 loop failures led to a native predicate-path readiness repair; the
final gate passes 2,400 launches. Boolean ABI and arbitrary type unions remain open.

The [logical Boolean helper extension](../compiler/gaps-boolean-helpers.md) now admits exact predicate
parameters/returns in pure static expansion, with original source/native signatures
and a 2,400-launch independent A100 gate. Byte-backed launch/storage semantics
remain a separate physical ABI experiment.

The [narrow scalar entry ABI](../compiler/gaps-narrow-parameters.md) adds exact byte/halfword
slots and extension with coordinated parser/selector/interpreter/verifier and
public explicit-launch bindings. Independent original-PTX/A100 and explicit
run-spec gates pass; logical Python Boolean launch/storage is a separate contract.

The [logical Boolean kernel parameter cycle](../compiler/gaps-boolean-parameters.md) preserves predicate
body semantics, canonical byte host values and exact source/manifest/native roles.
Its original-source A100 gate passes 9,600 launches plus 960 public Boolean
run-spec launches; Array/Buffer/shared Boolean storage remains separate research.
