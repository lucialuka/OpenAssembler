# Coverage evidence and next implementation priorities

The [supplied matrix](../../compiler_exp/coverage_matrix/input-coverage.json) is
retained byte-for-byte from `closed_exp/coverage.json`. Its
[audit](../../compiler_exp/coverage_matrix/audit.json) confirms **39 cases: 27
accepted and 12 unsupported, zero unexpected admission decisions**, with seven
raw samples for each accepted case in each of three CPU phases: capture/FX and
autodiff, storage planning, and PTX preparation. There are 189 samples per phase,
567 total. Unsupported cases have no performance samples and are not averaged
as slow implementations.

The recorded environment is macOS ARM CPU, Python 3.13.11, Torch 2.15.0, one CPU
thread. Cold first samples remain visible. These records do not measure SASS
execution, kernel throughput, or an end-to-end GPU training step. The JSON does
not contain the generator/test source or model factories, nor enough state-check
records to independently establish every reported full-output/dX/dW/four-SGD
comparison or all four view/reuse combinations. Those stronger checks are user
reported context, not newly reproduced evidence from this file alone.

The new, fully source-defined [boundary replay](../../compiler_exp/coverage_matrix/replay.py)
reconstructs representative workloads from those shape/type metadata, with its
own visible factories. The [successful run](../../compiler_exp/coverage_matrix/replay-20261119T152043Z.json)
also records 27 acceptances and 12 expected rejections, seven raw repetitions
and all four view/reuse combinations for every accepted case, while verifying
the input plan is unchanged. Its planning/preparation samples include **all four
combinations**, so they are not timing-equivalent to the original samples.
It is explicitly a new corpus, not a reproduction of absent original factories,
plan hashes, full tensor execution or training-state checks. The initial replay
fixture used varargs that FX does not capture in this contract; that failure
and the corrected fixed-arity workload source are retained.

## Confirmed boundary

Accepted cases cover bounded Linear/activations/reductions/layouts/concat and
explicitly half-rounded products. Rejections include unsupported sigmoid,
general sum, slicing, division, exact GELU/dropout; FP16/noncontiguous/NaN
examples; half BMM batch 33 or dimension 513; and half dual-linear dimension
513. Current half BMM still requires matching batch shapes, batch count at most
32 and positive M/N/K at most 512. No resource limit is raised merely to turn
an expected rejection into an acceptance.

The [November 19 independent review](../../compiler_exp/semantic_acceptance/input-review.txt)
also demonstrates accepted-but-wrong custom scalar PTX. The completed
[semantic acceptance repair](semantic-acceptance.md) fixes saturation and FTZ,
adds explicit modifier/type/operand admission and improves source diagnostics.
Capability and correctness counts remain separate; a larger test count is not
a coverage percentage for arbitrary ML repositories.

## Priorities and complete acceptance criteria

| Priority | Work and candidate experiments | Required completion evidence |
|---|---|---|
| P0 acceptance reliability | Saturation/FTZ repair and scalar admission inventory; negative modifiers, shape/type/arity cases | Implemented bounded repair, independent math + ptxas + A100 outputs, guards, poisoned registers and unchanged incumbent policy binaries |
| P0 verification-model correctness | Existing one-ULP exact-division emulator/interpreter mismatch reproduced on unchanged baseline; see [regression limits](semantic-acceptance.md#final-regression-limits) | Keep the fixed counterexample and determine MUFU/FCHK/rounding behavior against independent hardware before changing the emulator or verification status; do not widen tolerance |
| P1 custom device kernels | Extend pointer-annotated/scalar ABI, vector memory width coverage, atomics/order, then calls/linking and local storage/spills; use independent nvcc-generated inputs | Complete modifier/type/operand/memory-order contracts, emitted native provenance, independent outputs and concurrency/resource hardware tests; unsupported forms still reject |
| P1 normalized model semantics | Broadcasting with reduction-correct backward; integer indices/gather/embedding; masked/causal attention and exact slicing/layout effects | Shape/stride/dtype/alias contracts, full forward/dX/dW, all storage combinations, changing inputs and mask edge cases; no silent type or numerical policy changes |
| P1 training state | Cross-entropy and Adam/AdamW with exact optimizer options, RNG/effects and checkpoint resume | Independent full loss/gradient/parameter/buffer/optimizer state across successive updates and resume; preserve epsilon/bias correction/weight-decay policy |
| P1 inference state/dynamics | KV-cache storage ownership, growing lengths, guarded shape variants and explicit lazy OA specialization | Prefill and token-by-token equivalence, cache mutation/position contracts, shape-miss behavior, complete memory/output checks |
| P1 repository execution | Resolve package/build/generated-source/extension/launch graph while preserving CPU orchestration | Pin a real custom-extension/CNN/transformer workload; build and run through the normal CLI with complete state/dependency and ownership evidence |
| P2 efficient general lowerings | Valid generic lowering first, then coalescing/vector tails, reuse/fusion, register-pressure-aware overlap | Matched whole-step timing and held-out shapes; compare current OA plus original application; include packing/transfers/host/optimizer work |
| Separate target expansion | New ISA/ABI/resource/scheduling models beyond SM80 | Their own encoder conformance and hardware evidence; flags alone do not provide support |

The listed P1 items remain **open**. This change does not claim masked attention,
integer embedding/cross-entropy, Adam/AdamW, arbitrary model repositories or a
whole-native character-Transformer row. A passing host Transformer test module
is not an OA execution or performance result. BMM resource expansion needs its
own design and hardware evidence, rather than deleting admission bounds.

For each workstream, stage credible candidate implementations under
`compiler_exp/`, freeze contracts before measurement, and retain failed gates.
Use the existing [memory/latency evidence](memory-latency.md) to decide whether
larger tiles, more buffering or overlap actually help. Current correctness fixes
change only the previously incorrect semantic contracts; existing kernels retain
their native instructions. Unsupported workloads stay outside timing aggregates.

## Distribution and reproduction

The review found maintained documents linked to Git-ignored experiment files
and an ignored roadmap. The named compiler experiment directories and roadmap
are now included in the source distribution, alongside their inputs, sources,
archives and reports. A check of a Git-exported tree is required in addition to
the live documentation check. Private `closed_exp/`, local environments and
generated hardware build directories remain excluded; the supplied matrix is
explicitly retained in the public research directory.

```sh
.venv/bin/python compiler_exp/coverage_matrix/audit.py
.venv/bin/python tools/check_docs.py
```
