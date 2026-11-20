# Reliable scalar acceptance and source diagnostics

Status: implemented

The November 19 independent review reported accepted PTX that produced wrong
A100 results. This repair takes priority over additional tuned GEMM contracts.
An accepted operation must preserve its complete modifier/type/operand semantics;
valid SASS encoding alone is insufficient.

## Reproduction and implementation

The [frozen review](../../compiler_exp/semantic_acceptance/input-review.txt)
reports four signed saturation failures and two FTZ comparison failures confirmed
on A100. The local [before experiment](../../compiler_exp/semantic_acceptance/before-valid-bit-registers.json)
independently reproduces 984 clean/poisoned-register mismatches across four
optimization levels. Non-saturating integer arithmetic and non-FTZ comparisons
are passing controls. Its expectations use Python integer clamping and raw FP32
bit decoding, independently of both OA interpreters.

This baseline replays the corrected bit-register fixture in an isolated export
of revision `91e3f94`, before the production repair. The earlier CPU-only record
used integer register declarations for the FP32 comparison; its raw outcome is
retained, but the corrected replay and ptxas-accepted A100 fixture are the
semantic evidence used here.

| Contract | Previous behavior | Implemented lowering |
|---|---|---|
| `add.sat.s32` / `sub.sat.s32` | Dropped `.sat`, wrapped at overflow | Compute the wrapped result, detect signed overflow from operand/result signs, and select the appropriate signed bound. |
| FP32 `setp` with `.ftz` | Dropped `.ftz`, preserved subnormal operands | Flush only zero/subnormal predicate inputs before the existing floating comparison. Normal values, infinities and NaN bit patterns remain unchanged. |
| Unknown modifiers/types/widths | Several selector branches matched only part of the instruction | Explicit scalar form and operand admission before selection; unsupported semantic combinations reject with a PTX source line. |

The saturation sequence uses the established IADD3, LOP3, ISETP, SHF and SEL
encodings; it does not guess a hardware saturation field. For add, the overflow
sign is `((a ^ result) & (b ^ result))`; for sub it is
`((a ^ b) & (a ^ result))`. The clamp is selected from the sign of `a`.
Aliased destinations and predicated execution retain the original value when
the guard is false.

The bundled encoder does not have a general `FSETP.EQ.FTZ.AND` template. The
selected software input-normalization path uses existing validated instructions:
mask magnitude, compare it with the minimum normal magnitude, and select zero
for smaller magnitudes. Both signed zeros have identical predicate comparison
results, so this preserves the predicate, including unordered NaN behavior.
This adds work only for the newly repaired contract. No speedup is claimed;
a direct native FTZ template remains a future oracle-recovery optimization.

[Scalar contracts](../../openassembler/backend/contracts.py) enumerate admitted
mnemonic/modifier/type/arity forms and validate memory/vector operand shapes.
The [machine-readable inventory](../../compiler_exp/semantic_acceptance/scalar-contracts.json)
lists 426 forms. This is an admission schema, not 426 proven universal
capabilities: instruction-specific constraints, register allocation, resource
limits and encoding can still reject. Tensor MMA/LDSM/async-copy paths retain
their separate exact checks. Legacy comparison selectors reject saturation/FTZ
instead of silently miscompiling them; the production backend implements them.

## Diagnostics and explicit bug rejection

PTX instructions now retain actual source line numbers rather than semicolon
segment indices. Production selection errors include that PTX line; CLI/build
reports also identify the source file. CPU FX capture records user call stacks,
so unsupported model nodes and later shape/semantic rejection messages identify
the user's forward file and line. The node and reason remain in the diagnostic.

The Python kernel frontend now rejects shared-memory reads with **no possible
writer anywhere in the kernel**, including a constant slot for which every
potential write has a different constant index. A dynamic store index remains
a possible producer; the check does not infer thread synchronization or temporal
initialization. Existing literal negative-index/shared-bounds checks, literal
zero-division checks and divergent-barrier rejection remain in place.

Color detection now resolves the current stdout at call time and tolerates a
closed stream. An import-time captured stdout must not break later CLI embedding,
test capture or error reporting.

| Bug/condition | Actual guarantee |
|---|---|
| Python syntax/unsupported constructs | Static frontend rejects with file/line; user modules are not executed. |
| Constant out-of-bounds shared index | Already rejected; global buffer lengths are not supplied by `Array(TYPE)`. |
| Shared read with no possible producer | Newly rejected statically, with file/line. |
| Shared read before a later write, races or divergent memory access | No general static proof is provided. A potential producer is not proof of correct ordering. |
| Runtime out-of-bounds global/shared access | No general bounds instrumentation is added. Tested canaries are finite checks, not a proof for all inputs. |
| Fixed-width integer overflow | Wrapping is the declared default language semantics; explicit PTX saturation now clamps correctly. |
| Runtime zero division/NaN/Inf | Device numeric semantics remain explicit; no automatic Python exception semantics are invented. |
| Wrong application math/host code | Finite reference checks can detect tested mismatches; arbitrary host program correctness is not inferred. |

## Validation and failure custody

[After probes](../../compiler_exp/semantic_acceptance/after-final.json) pass all
four optimization levels with clean and poisoned registers. Tests include signed
limits, random bit patterns, true/false predicates, aliased operands, both signs
of subnormals/zero, minimum normal values, infinities and NaNs, plus explicit
rejections for carry modifiers, invalid saturation types, memory-order variants,
unknown modifiers and incompatible vector shapes.

The [A100 confirmation](../../compiler_exp/semantic_acceptance/a100-20261119T150205Z.json)
passes **144 launches and 25,104 checked output elements**, with zero mismatches,
complete input immutability and intact guards. It covers saturation and FTZ plus
non-saturating/non-FTZ controls at safe/mined/sched/opt, both OA and same-source
ptxas, under three output poison patterns. The actual device is **NVIDIA A100
80GB PCIe**, CC 8.0, driver 13030, Torch 2.7.1+cu128, CUDA 12.8. These are
correctness results; no candidate was timed and no SXM/PCIe times are pooled.
The source, native/reference cubins, SASS, manifests and exact original harness
snapshots are retained beside the report.

Two harness failures remain recorded. The first remote import required a
test-only helper absent from its payload; it was replaced with the bundled
tool helper. The next fixture declared FP32 comparison operands as integer
registers and ptxas correctly rejected it. The corrected bit-register fixture
passes ptxas and OA. Neither failure was counted as compiler correctness evidence.

The [pinned CPU/compiler regression](../../compiler_exp/semantic_acceptance/pinned-tests-20261119T152336Z.json)
passes 100 targeted tests with Torch 2.7.1. This is separate from GPU execution;
the hardware record above covers the actual device checks.

Compiler identity changes require a complete retained-policy binary audit before
refreshing dispatch provenance. The audit must preserve the original measured
GEMM/rowwise selections and encoded words, controls and register counts. Newly
correct scalar contracts do not authorize retuning unrelated model kernels.

```sh
.venv/bin/python -m pytest -q tests/test_backend_contracts.py tests/test_source_diagnostics.py
.venv/bin/python compiler_exp/semantic_acceptance/probe.py NEW_REPORT.json
modal run compiler_exp/semantic_acceptance/run_modal.py --action hardware
modal run compiler_exp/semantic_acceptance/run_modal.py --action tests
```

Hardware results apply to the recorded compiler hashes. The final validation
record and policy continuity check separately establish final-source agreement.
Primary reference: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
integer add/sub saturation and predicate `.ftz` semantics.

## Final regression limits

The [machine-readable validation record](../../compiler_exp/semantic_acceptance/validation.json)
binds the checks, raw failures, source/hardware records and explicit limits.

The targeted source/semantic/entry/welcome suite passes **63 tests, two skips**.
The pinned Torch 2.7.1 run passes **100 tests**. The full final local suite is
**1,061 passed, one failed, seven skipped**: it is not reported as green.
Its failure is an existing exact-division SASS-emulator discrepancy, reproduced
on an isolated unmodified `91e3f94` export with the same corpus and inputs.
The [baseline](../../compiler_exp/semantic_acceptance/emulator-limit-baseline.json)
and [current](../../compiler_exp/semantic_acceptance/emulator-limit-current.json)
records are identical: numerator `0x8147ec48`, denominator `0x21dea823`, emulator
result `0x9ee5dc9c`, PTX interpreter result `0x9ee5dc9d`. The corpus inputs and
listing are retained in the adjacent emulator-limit artifact archive.

The unchanged regression uses a process-randomized hash seed; an isolated
136-test emulator run passes with different sampled inputs, which does not
repair or supersede the retained fixed counterexample. MUFU/FCHK are approximate
or partial emulator models. The exact cause of this one-ULP result has not been
established; no tolerance, test or floating policy is changed to admit it.
It remains a P0 verification-model issue, and emulator success must not be
substituted for hardware correctness. The saturation/FTZ hardware checks and
their complete final-source reencoding are separate, passing evidence.

The [final policy continuity audit](../../exp/native_indexing/dispatch-continuity-20261119T150753Z.json)
checks 522 GEMM candidate contracts in 1,044 retained session comparisons,
40 rowwise comparisons and six fixed tensor goldens without changing policy
choices or encoded words. Stack recording and contract checks add compilation
work; compile-time overhead was not benchmarked here. No existing GPU instruction
stream or runtime throughput change is claimed.
