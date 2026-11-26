# Research and feature catalogue

This is the complete entry point for the repository's researched features and
ideas. Each item links to **one canonical dossier** with plain-language context,
implementation/tests, positive and negative findings, methodology, prerequisites,
reproduction commands, raw evidence and remaining questions.

**Correctness, performance and deployment are separate questions.** A technique
can be correct but slower, implemented only as an example, useful only on one
shape, or experimentally valid without being integrated. “Not selected” below
is not a claim of universal mathematical invalidity. Source-only proposals and
hardware-unavailable ideas have their own list instead of being called failed
experiments. Several dossiers can use the same experiment; their counts are
not independent replications.

For architecture and newest measurements read [STATUS](../STATUS.md). For all
CLI syntax use [COMMANDS](../COMMANDS.md). Shared setup, evidence conventions and
a short glossary are in [DEVELOPMENT](../DEVELOPMENT.md). The complete
cross-cutting [whole-stack study](WHOLE_STACK_RESULTS.md) and
[primary references](REFERENCES.md) remain essential companion documents.

The cross-cutting black-box compiler and GPU reverse-engineering record is
centralized in the [reverse-engineering results hub](../reverse_eng_results/README.md).
Per-idea dossiers remain the authoritative status and reproduction records for
individual features.

The [compiler gap register](../compiler/gap-register.md) links the November 23
foundation additions and their canonical dossiers: typed wide values/buffers,
[Python atomic effects](../compiler/gaps-python-primitives.md),
[native ordering/shared atomics](../compiler/gaps-atomics.md), recorded builds,
optimizer policies and complete owned-state checkpoints. Its
[integration evidence](../../compiler_exp/gap_closure/integration_20261123/README.md)
keeps source tests, hardware correctness, binary continuity and open exit gates
separate; this does not change the historical idea-count inventory below.
The [floating literal cycle](../compiler/gaps-wide.md#decimal-and-raw-floating-literal-contexts)
adds decimal/exponent parsing, use-type rounding and raw payload/width rules,
with independent rational, toolkit and A100 controls. This is a numerical/source
contract repair with no speed claim.
The subsequent [typed-expression cycle](../compiler/gaps-wide.md#typed-scalar-operand-constant-expressions)
adds integer precedence/casts and per-operation binary64 rounding. It retains
failed cross-toolchain controls and unresolved signedness/generated-NaN cases
separately from the final A100 and full-suite gates.
The [shared-width cycle](../compiler/gaps-memory.md#scalar-shared-byte-halfword-and-64-bit-memory)
adds scalar narrow/64-bit exchanges and four exact recovered store templates.
A failed 64-bit hardware gate led to a bounded barrier schedule repair before
the final 1,216-launch A100 admission. Shared race/initialization and broader
width/order contracts remain open.
The subsequent [bounded decode specialization](../compiler/gaps-decode.md)
adds explicit unseen-length builds with full cache/native-launch validation;
paging and complete serving remain open.
The [full-warp shuffle dossier](../compiler/gaps-collectives.md) adds exact
indexed/up/down contracts with 1,792 A100 launches and preserved binary policies;
general participation and cooperative synchronization remain open.
The [native local-storage dossier](../compiler/gaps-storage.md) now adds static
per-thread frames and scalar32 accesses with independent A100 frame/resource
and complete-state validation. The subsequent
[spilling route](../compiler/gaps-spilling.md) preserves scalar, aligned
pair/quad and predicate values under explicit limits and automatic pressure.
Generic module storage, device calls, slot reuse and pressure-aware policy
selection remain open.

The [typed Python shared-array extension](../compiler/gaps-python-types.md#wide-shared-arrays-and-native-cta-reconvergence)
adds i64/u64 storage, mixed padding and eight-byte index scaling. A hardware
conditional-writer failure led to a separate bounded CTA convergence pass.
The expanded gate passes 1,280 A100 launches; cyclic control flow and general
shared-memory safety remain open.

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

The [virtual pressure scheduling cycle](../compiler/gaps-spilling.md#pressure-aware-virtual-scheduling)
adds independent dependency/resource checks, paired A100 kernel comparisons
and maximum-frame recovery. Its explicit compiler-cost measurements remain
separate from resident GPU gains. Tensor and occupancy policies retain their
own admission gates.

The November 26 [model kernel study](../../exp/model_perf/RESULTS.md) measures
reuse of repeated 128×128 forward operand packs and allocator interval-scan
cost. The [allocator dossier](ideas/scalar-register-allocation.md) records exact
binary continuity and compile-time measurements alongside the workload results.

The role/shape coverage audit is under
[`exp/shape_coverage`](../../exp/shape_coverage/README.md). That study remains a
contract and measurement matrix. The separate completed
[November 11 production replay](../../exp/dispatch_tuning/RESULTS.md) supplies 42
exact native GEMM dispatch entries: 34 tuned policies and eight reference
retentions. Unknown or stale contracts keep the reference; the five rowwise
entries have their own independent evidence.

Catalogue reviewed **November 14, 2026**. There are **82 dossiers**: **40 implemented**, **9 validated but not integrated**, **22 tested but not selected**, and **11 open or hardware unavailable**. The earlier November 9 revalidation ledger is in [`exp/revalidation/`](../../exp/revalidation/README.md); the new role/tile evidence is inventoried in [`exp/shape_coverage/`](../../exp/shape_coverage/README.md).

The [native convolution cycle](../compiler/gaps-convolution.md) adds FP32 NCHW
Conv2d and input/weight/bias gradients for groups=1. Its two-layer CNN passes
public CLI compilation and saved training replay on A100. The
[experiment](../../compiler_exp/gap_closure/conv2d/README.md) retains the original
rejection, independent CPU/native/hardware results and exact source identities.

## 1. Validated and implemented

These are available in the current code under their stated contracts. Explicit tuning options and shipped examples are not automatically selected or universally faster.

| Feature / idea | What is established or limited |
|---|---|
| [Exact runtime indexing and F2I source collection](ideas/runtime-integer-indexing.md) | Repairs a real large-grid conversion-source hazard with permanent A100 regression and binary continuity evidence. |
| [Native Python model graphs and launch plans](ideas/native-model-bridge.md) | CPU capture and symbolic backward lower bounded broadcast/attention/indexed graphs to OA-only inference, MSE/CE and SGD/Adam/AdamW; exact cache sessions and numerical contracts remain explicit. |
| [Recovered arithmetic instruction sequences and rounding policy](ideas/arithmetic-instruction-selection.md) | The standard backend imports retained CUDA 12.8 arithmetic fragments and generates its own words. |
| [Constant integer division and remainder lowering](ideas/constant-division.md) | The implemented 32-bit lowering removes a reciprocal-based path from direct PTX division/remainder sites. |
| [Cooperative population LayerNorm](ideas/cooperative-layernorm.md) | Shifted warp-per-row population LayerNorm has measured exact CTA geometry; affine parameters, saved statistics and backward remain outside the production operator. |
| [Cooperative FP32 row softmax](ideas/cooperative-softmax.md) | Warp-per-row FP32 softmax has measured exact CTA geometry; earlier serial-mapping gains and current W4-relative geometry gains are separate, with no whole-model claim. |
| [Shape-aware rowwise CTA geometry](ideas/shape-aware-rowwise-dispatch.md) | Five exact heldout A100 contracts select 2/8 warps per CTA; unknown or stale shapes keep four warps. |
| [Loadable cubins and metadata-safe code replacement](ideas/cubin-metadata-splicing.md) | The production container code parses/serializes CUDA ELF and .nv.info records, replaces kernel text, fixes exit offsets and declares the generated register count. |
| [CUDA C/C++ frontend through nvcc PTX](ideas/cuda-cpp-frontend.md) | nvcc supplies CUDA parsing/PTX, while OpenAssembler supplies supported native SASS generation. |
| [Dead-code removal, copy coalescing and local folding](ideas/dead-code-and-copy-elimination.md) | These passes are in the regular scalar and tensor compilation pipelines before allocation. |
| [Direct pointer homes for asynchronous GEMM copies](ideas/direct-pointer-homes.md) | Direct homes remove pointer copies; exact current dispatch selects direct/early combinations while tall regressions retain other policies. |
| [Earlier pointer advancement in tensor mainloops](ideas/early-gemm-pointer-advance.md) | Explicit pointer-advance late/early/auto rules coexist with exact measured dispatch; current complete plans select 25 early policies while tall contracts keep legacy/late Morton. |
| [Fused Adam-like scalar update example](ideas/fused-adam-like-update.md) | Fused example passes tolerance and beats the chosen eager expression; it is not complete AdamW or a tuned-optimizer comparison. |
| [Fused FP32 bias, residual and ReLU epilogues](ideas/fused-gemm-epilogues.md) | The explicit `bias_residual` and `bias_residual_relu` plans compute `(A@B.T + Bias) + Residual`, then optional finite-value ReLU, with FP32 inputs to the epilogue and FP32 output. |
| [GPU execution, explicit launch specifications and fault isolation](ideas/gpu-execution-and-launch-specs.md) | Typed launch planning, fault-isolated local/Modal/SSH transports implemented; Modal hardware is validated and latest real SSH evidence is absent. |
| [Four-way counted-loop unrolling](ideas/loop-unrolling.md) | The `opt` scalar path applies four-way unrolling to the supported counted-loop form. |
| [Exact AlphaTensor/Strassen factors and FP32 research plans](ideas/low-rank-factorization-library.md) | Exact factors and explicit FP32 research plans exist; sampled/interrupted timing records do not prove a general win. |
| [Morton8 logical CTA traversal](ideas/morton-cta-order.md) | Corrected 8×8 CTA traversal is selected for the two independently validated 16384×4096×4096 contracts; it is not an MMA storage layout. |
| [Native tanh-approximation GELU example](ideas/native-gelu.md) | Tanh-GELU passes a predeclared FP64 tolerance, while the old exact gate failure remains and gains depend on size. |
| [Native scalar ReLU example](ideas/native-relu.md) | Finite-input exact example passes; measured large-size PyTorch is faster, with no universal speedup. |
| [Native SM80 tensor GEMM compiler and mainloop](ideas/native-tensor-gemm.md) | Contiguous FP16/BF16-to-FP32 GEMM has checked 128×128 and 64×64 CTA families, protected tensor dependencies and 44 exact measured dispatch entries; this is not a complete native transformer compiler. |
| [CUTLASS-style load-next / compute-current fragment schedule](ideas/cutlass-load-next-compute-current.md) | Explicit load-next is admitted for two exact FP16/BF16 shapes; no load-next policy wins the current 42-contract frozen replay, while earlier gains remain valid against their recorded controls. |
| [CUTLASS-style grouped-N CTA traversal](ideas/grouped-n-cta-swizzle.md) | Explicit grouped_n8 traversal and checked launch mapping are integrated; no grouped-N policy wins the current 42-contract frozen replay, while older tall gains remain scoped evidence. |
| [Operand reuse hints and ordering](ideas/operand-reuse-hints.md) | The tensor scheduler emits reuse hints for supported neighboring HMMA operands. |
| [Paired B-fragment LDSM loads and register-group use](ideas/paired-fragment-loads.md) | The selected native kernel uses paired `LDSM.x4` B loads. |
| [Native PTX input and compilation boundary](ideas/ptx-native-frontend.md) | Supported SM80 PTX routes to the native scalar/tensor backends with explicit verification limits. |
| [Typed Python kernel frontend](ideas/python-kernel-frontend.md) | Typed numeric Python source compiles without executing imports; fixed GPU numeric semantics are explicit. |
| [Repository discovery, manifests and safe artifact publication](ideas/repository-builds.md) | Static inspect/JSON coverage, path shorthand, local failure reports, safe discovery, path mirroring, locking and publication are implemented and tested. |
| [SASS execution emulator and fitted tensor arithmetic](ideas/sass-emulator-and-tensor-numerics.md) | Functional execution and fitted tensor arithmetic pass the recorded cases; they do not prove real timing or all special values. |
| [SM80 SASS instruction encoding](ideas/sass-encoding.md) | The production encoder emits the supported SM80 instructions from generated templates and field maps. |
| [Scalar register dependencies, scoreboards and synchronized shuffles](ideas/scalar-dependency-correctness.md) | Fresh binding, reader sets, scoreboard-generation retirement, source WAR and actual WARPSYNC repair hardware failures. |
| [Register allocation, aligned groups and guarded definitions](ideas/scalar-register-allocation.md) | The scalar compiler uses virtual values, liveness, stable homes for repeated/guarded definitions and aligned register-group allocation. |
| [Measured scalar instruction scheduling](ideas/scalar-scheduling.md) | Corrected fixed-pipe probes found four cycles for tested same-pipe ALU/FMA pairs, five across tested classes, and 13 from a predicate definition to tested guarded instructions/branches. |
| [Shared-memory swizzling and bank conflicts](ideas/shared-layout-swizzling.md) | The selected GEMM uses the tested symmetric 7/7 XOR swizzle. |
| [Two-tile stage-pair specialization and peeled drain](ideas/stage-pair-specialization.md) | All 44 installed GEMM plans retain the explicit two-stage stagepair protocol; combined remains an admissible comparison but wins no frozen row. |
| [Complete application-state verification diagnostics](ideas/state-verification-diagnostics.md) | Checks all state groups under unchanged gates and bounded failure reports; exposes both BF16 parameter failures. |
| [Carry-linked pointer updates and folded shared addresses](ideas/tensor-address-generation.md) | The earlier address work reduced one selected kernel from 1,898 to 1,812 static non-NOP instructions; six same-worker cases improved about 0.9–5.4%. |
| [Vectorized FP32 result stores](ideas/vectorized-fp32-epilogue.md) | The native GEMM epilogue groups FP32 stores into pairs and passes an existing aligned register pair directly when possible. |
| [Shape-aware native GEMM dispatch and on-device tuning](ideas/shape-aware-native-gemm-dispatch.md) | The original 42-contract replay plus two independently admitted FP16 contracts install 36 tuned policies and eight baseline retentions; exact source/contract checks and optional local/Modal tuning preserve shape-specific alternatives. Missing/stale data falls back to the reference. |
| [Smaller CTA tiles for short transformer GEMMs](ideas/small-grid-tensor-cta-tiles.md) | Twenty-four current contracts select checked tile64 variants after the original 42-contract replay and two-contract extension; shape regressions and same-source ptxas gaps remain. |

## 2. Validated, not integrated as a production optimization

These capabilities, diagnostics and research prototypes have bounded validation. The dossier states whether correctness, measurement or performance was established; none should be mistaken for an automatic compiler pass.

| Feature / idea | What is established or limited |
|---|---|
| [Four-edge A100 tensor dependency scheduler](ideas/a100-tensor-dependency-scheduler.md) | A100 probes validate bounded dependency measurements; the generalized scheduler remains experimental, with no reordered stream promoted. |
| [cuBLAS dispatch observation and cuBLASLt controls](ideas/cublas-observation-and-baselines.md) | Vendor dispatch and matched controls guide independent kernels; no vendor kernel was extracted into the compiler. |
| [Structured-sparse MMA capability probes](ideas/sparse-mma-capability.md) | Limited ptxas-built sparse probes ran on A100; native OA sparse code generation and speedup are not established. |
| [Hardware-counter-guided tensor analysis](ideas/tensor-counter-profiling.md) | Retained matched profiles show equal tensor work but different overhead and stalls. |
| [Clocked tensor probes and synchronization sweeps](ideas/tensor-timing-and-synchronization.md) | Corrected exact-SASS probes measured an approximately eight-cycle-per-operation slope for the tested dependent HMMA sequence and about 41 cycles per iteration for LDSM→LOP3→IADD3. |
| [Whole-transformer training and inference evaluation](ideas/whole-transformer-evaluation.md) | Nine of ten configurations pass both rounds; gains use Inductor/Triton and BF16 training remains excluded. |
| [Native NN arithmetic and composed-network campaign](ideas/nn-arithmetic-campaign.md) | 26 operator families and six custom NN configurations expose conditional gains and failed training gates; no general CUDA superiority. |
| [Fused local dot products for backward row reductions](ideas/fused-backward-reductions.md) | Explicit FMA improves selected native backward reductions; smaller cases can lose and same-source ptxas remains faster. |
| [Stable analytic GELU backward](ideas/stable-gelu-backward.md) | A stable exponential derivative passes the unchanged accuracy gate after the direct-tanh form fails; no ptxas speed win. |

## 3. Tested and rejected or not selected

This list distinguishes failed numerical gates, correct-but-slower candidates, and candidates that added no separate benefit. Some example source remains for testing; presence in the tree does not override a failed gate.

| Feature / idea | What is established or limited |
|---|---|
| [Asymmetric shared-memory address layouts](ideas/asymmetric-shared-banking.md) | The historical shared-layout sweep tested asymmetric 7/3 and 3/7 swizzles against symmetric 7/7. |
| [Preserving intermediate casts as a BF16 repair attempt](ideas/bf16-cast-preservation.md) | Keeping eager-like casts still fails the original BF16 parameter gate; the setting was not promoted. |
| [Compile-time rank-seven versus rank-eight native MMA](ideas/compile-time-low-rank-mma.md) | Compile-time rank-seven tests did not establish a competitive precision-matched end-to-end tensor mainloop. |
| [Deeper copy pipelines and larger tiles](ideas/deeper-pipelines-and-tile-shapes.md) | For the tested 64-wide K tile, two stages beat three/four: approximately 1.003 ms versus 1.178/1.180 ms in the archived run. |
| [Hilbert, grouped-M and alternative CTA orders](ideas/hilbert-and-grouped-cta-order.md) | The phase-five study tested N-fast, grouped-M, Morton8 and Hilbert8 while retaining the physical HMMA/LDSM/LDGSTS words and controls. |
| [Persisting-L2 cache allocation policies](ideas/l2-persistence.md) | The controlled cache-policy study performed 224 output checks across square/tall FP16/BF16 and policy settings, with graph-node policy readback. |
| [Copy prefetch hints and bounded fragment windows](ideas/prefetch-and-fragment-windows.md) | The production follow-up tested cp.async L2-prefetch modifiers, B-fragment windows of two/four and bounded prefetch. |
| [Software producer/consumer warp specialization](ideas/producer-consumer-warps.md) | The tested role-split variants passed four large-case A100 correctness checks but measured about 1.019–1.943 ms and lost to the unified-warp schedule. |
| [Runtime-factor Triton low-rank lowering](ideas/runtime-factor-low-rank-triton.md) | The early A10040GB prototype was far slower: at 4096³ FP16, cuBLAS was about 0.511 ms, direct Triton 1.325 ms, Strassen 11.611 ms and AlphaTensor 13.041 ms. |
| [PyTorch application capture and saved replay](ideas/pytorch-application-capture.md) | Historical Inductor/Triton capture/replay; disabled and superseded by the OA-native bridge, with original results retained. |
| [Naive scalar attention-score baseline](ideas/scalar-attention-score-baseline.md) | The illustrative score kernel fails its retained exact gate; it is not a fused attention implementation. |
| [Serial scalar cross-entropy baseline](ideas/scalar-cross-entropy-baseline.md) | The example exists, but its retained exact hardware gate fails; no speedup is admitted. |
| [Naive scalar matrix-multiplication baseline](ideas/scalar-matmul-baseline.md) | The scalar example fails its retained exact gate and is not the validated tensor-core GEMM. |
| [Software fragment caching in registers](ideas/software-register-caching.md) | Bounded B-fragment windows of two/four and prefetch variants passed correctness but were slower across seven paired rounds on the tested workloads. |
| [No-op address instruction removal](ideas/non-tensor-address-overhead.md) | Removing eight proven no-op address adds passed full output checks and reduced static SASS, but did not improve paired A100 timing. |
| [Adam epsilon sensitivity of near-invariant BF16 key bias](ideas/adam-epsilon-sensitivity.md) | The current diagnostic passes only after changing the optimizer policy; it is not a compiler repair or speedup. |
| [Immutable inference weight-packing cache](ideas/inference-weight-packing-cache.md) | Application cache correctness passes, but an equally cached ptxas control remains faster; no OA compiler gain is admitted. |
| [Joint direct-pointer, traversal and epilogue options](ideas/joint-tensor-options.md) | Full checks pass, but combinations are mixed or slower and add no selected production policy. |
| [Two-slot asynchronous barrier overlap](ideas/mbarrier-overlap.md) | A retired tiny one-CTA probe passes, but no current dense GEMM integration or application gain exists. |
| [Native GEMMs inside a complete hybrid residual MLP](ideas/native-hybrid-mlp.md) | Whole-step checks are valid but current native OA combinations are mixed or slower. |
| [Padded forward/backward integration prototype](ideas/padded-training-hook.md) | The bounded hook is state-correct but slower and its original runner is retired. |
| [Post-allocation pointer-copy forwarding](ideas/post-allocation-pointer-forwarding.md) | The rewrite is byte-identical to existing direct-pointer mode and adds no distinct optimization. |

## 4. Open, incomplete or unavailable on A100

These are source-inspection conclusions, implementation proposals, missing hardware features or unprofiled host changes. They have no admitted optimization result.

| Feature / idea | What is established or limited |
|---|---|
| [A100 feasibility: TMA, clusters, FP8, DPX and virtual grouping](ideas/a100-hardware-limits.md) | Missing hardware capabilities cannot be enabled by a compiler flag; software substitutes remain distinct proposals. |
| [Adaptive shared-memory budgeting](ideas/adaptive-shared-memory-budgeting.md) | Two fixed 32/64 KiB GEMM families are shape-selected; independent adaptive stage/operand/epilogue budgeting remains unimplemented. |
| [Cross-architecture encoding and cubin observations](ideas/architecture-portability.md) | Historical CUDA 12.8 probes examined common control fields on SM75/86/90/100/120 and parameter/ELF metadata changes. |
| [Automatic equivalent SASS sequence search](ideas/automatic-sass-sequence-search.md) | Historical candidate searches explored several epilogue orders, fragment windows, pre-addressing, operand orders and scheduler variants; selected changes contributed to the current GEMM. |
| [Python versus a compiled host implementation](ideas/compiler-language-and-host-latency.md) | No profile shows that rewriting the host compiler improves GPU instruction speed; profile individual host passes first. |
| [Control-flow recovery and a SASS-to-PTX lifter](ideas/control-flow-lifting.md) | Useful instruction and reconvergence patterns were observed; a complete structured lifter is not shipped. |
| [CUDA-L2 project source audit](ideas/cuda-l2-source-audit.md) | Pinned public source and contracts were inspected; a faithful port and performance reproduction remain open. |
| [Joint shared-layout and register-fragment search](ideas/joint-layout-register-search.md) | Existing swizzle experiments largely held register organization fixed; fragment-window and tile experiments explored other dimensions separately. |
| [Fuse low-rank transforms into operand staging](ideas/low-rank-transform-fusion.md) | The existing runtime-factor and tiny compile-time studies identify this as promising future work. |
| [Alternative multi-precision accumulation paths](ideas/multi-precision-accumulation.md) | The implemented tensor GEMM already uses mixed precision in the ordinary sense: FP16/BF16 inputs with FP32 accumulation and FP32 output. |
| [Public CUTLASS and cuBLASLt techniques for A100](ideas/cutlass-cublas-public-techniques.md) | Tile, traversal, load-next and exact dispatch have separate implemented dossiers; generalized pipeline, staged-epilogue, split-K and joint cost-model transfers remain proposals. |

November 20: the [native model dossier](ideas/native-model-bridge.md) links the
new [model semantics and state evidence](../compiler/model-semantics.md).

The November 24 [scalar wide-selection addition](../compiler/gaps-wide.md#scalar-64-bit-conditional-selection)
adds exact integer/FP64 bit selection after a 1,480-launch original-source A100
gate. Earlier invalid FP64 fixtures and their corrected typed controls remain
separate. General arithmetic/type/modifier coverage is still incomplete.

The November 24 [low64 multiplication cycle](../compiler/gaps-wide.md#low-64-bit-integer-multiplication)
adds PTX/CUDA low products and typed Python `*`/`*=` after 1,560 independent
original-source A100 launches. Full product bits, wrapping, aliases/guards,
helpers/loops/casts and squares have explicit source/byte/native gates. Broader
G03/G11 coverage remains incomplete.

The November 24 [local-package helper cycle](../compiler/gaps-python.md#static-local-package-device-helpers)
adds defining-module scalar specialization, source/diagnostic/dependency custody
and normal local/remote builds after a 600-launch original-source A100 gate.
Aliases, relative imports, static re-exports and explicit roots have independent
source/byte/native controls. Arbitrary runtime imports and global captures remain
open; the static-value experiment is a separate hypothesis.

The November 24 [explicit static-value cycle](../compiler/gaps-static-values.md)
adds original-source Final numeric/tuple specialization after 1,440 A100 launches,
2,592 functional executions and 2,048 independent integer expression trees. Source,
value, diagnostics and ownership remain bound to original files. General objects,
runtime containers, metaprogramming and arbitrary host state remain open G05 work.

The November 24 [finite tuple iteration cycle](../compiler/gaps-static-loops.md)
adds original-source tuple control semantics after 2,400 A100 launches and 4,320
complete functional executions. Empty/nested loops, selected target state,
break/continue and exhausted else have independent source/byte/native gates.
General runtime iteration/control effects remain open G05/G06 work.

## How to use and maintain this catalogue

Start with the problem you want to solve, then read that dossier's contract and
reproduction section before running its commands. Historical drivers that were
retired are identified explicitly; a current smoke command is not advertised as
recreating an old performance experiment. Unsupported/proposed work does not
have fabricated validation commands.

The machine-readable index is [catalogue.json](catalogue.json). It and the
literal dossier status tags are checked by `python3 tools/check_docs.py` from
the repository root. The checker also detects orphan dossiers, broken local
links and command-reference drift without launching GPU work.

Duplicate chronological reports were consolidated into these dossiers. Their
exact original text is available only as [documentation provenance](../evidence/documentation-history/README.md),
not a competing active guide. Raw JSON, source snapshots, oracle inputs and
hashes remain at their recorded evidence locations.

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
