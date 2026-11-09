# Critical review of the transformer procedure report

The pasted report is a useful workload checklist, but several percentages and
targets need stricter evidence labels. This review compares it with the current
whole-stack results, the November 9 GEMM role experiments, and current compiler
contracts. No percentages below should be read as a generic transformer mix.

## Claims that are correctly bounded

The report correctly says the eager-versus-compiled whole-model column is
TorchInductor/Triton, not OpenAssembler-generated SASS. Its resident-data stage
numbers (forward/loss, backward, optimizer) are tied to a small synthetic
decoder study and exclude H2D, loader, startup, checkpoint and distributed
communication costs. Those stage medians are non-additive.

The first-priority state gate is also justified. The current-source BF16
diagnostic records a parameter failure at the step-29 checkpoint under epsilon
`1e-8`: the failed groups are the two QKV bias tensors, while the component diagnostics
show the key-bias slices have roughly `1.4e-6` to `1.9e-6` gradient L2 and many
sign disagreements. Under epsilon `1e-4`, the separately matched policy passes
the parameter gate. That is a policy sensitivity result; changing epsilon
cannot be presented as a compiler repair. Only steps 1 and 29 were checked in
that record: it does not locate the first failing update or the first causal
operator. This is eager versus Inductor/Triton evidence, not a native OA failure.

The [30-update trajectory replay](../whole_stack/runs/bf16-first-observed-divergence-20261109T111158.926633Z.json)
now fills that sampling gap without relaxing the gate. All 151 required
boundaries were collected. The first observed full-logit gate failure is before
update 3 (two of 524,288 elements); the first parameter-gate failure is after
update 22 (two QKV bias tensors). Raw/clipped gradients, Adam tensors and scalar
settings remain within their declared gates at all checkpoints. This locates
the first observed failing boundaries, not the operation that caused them.

## Percentages that must not be added

The GEMM, fused-attention, pointwise, reduction and optimizer figures are
recognized-kernel-family profiler sums. They overlap across streams and fused
operators; fused attention includes QK/AV products and softmax. They are not
fractions of the 5.796 ms or 4.032 ms end-to-end step. The appropriate whole-step
shares remain the separately reported resident stage table, with its non-additive
timing caveat.

The report's “expected gain” values (for example, 5–15% whole step, 2–10% from
HMMA scheduling or 1–4% from clipping) are hypotheses. The experiments below do
not validate those numbers. They should be treated as test targets with explicit
cost boundaries, not expected performance.

## Current GEMM evidence by role

The static matrix at `role-coverage-manifest.json` covers six model geometries,
15 forward/backward/attention roles and FP16/BF16: 180 rows. It separates a
dimension-compatible `GemmPlan` from a role whose prepared operands can run as
one kernel. There are 44 direct RCR routes, 84 rows whose dimensions fit but
whose transpose/packing or batched-attention semantics block direct routing,
and 52 rows rejected by the
full-tile dimension contract.
These counts are coverage cases, not workload frequencies or an automatic
native PyTorch route. Square dX/dW dimensions still require the right storage.

Current A100 runs cover medium QKV, output projection, FFN-up/down roles and
held-out small/large instances. All complete-output, three-pattern, canary,
input-immutability, matched-ptxas bitwise, poisoned-graph, disassembly and
resource checks pass: 156 checks for the repeated medium cohort, 54 for
held-out FFN-up, and 84 each for held-out QKV, output and FFN-down (462 total).
Same-source ptxas remains faster in all measured roles.
At `M=1024,H=1024`, OA/ptxas is about 1.15x for QKV, 1.48x for output, 1.40x
for FFN-up and 1.43x for FFN-down. These are isolated GEMMs; packing and whole
training are not included.

The role-specific policy sweep shows why exact dimensions and roles matter:

| Role and workload | Candidate versus OA m-fast | Decision |
|---|---:|---|
| FFN-up, FP16/BF16, 1024×4096×1024 | load-next about 9–10% faster | Keep exact policy opt-in; ptxas remains faster |
| FFN-up, FP16, 128×2048×512 | grouped-N did not beat m-fast materially; no load-next contract | Do not generalize the medium result |
| FFN-up, FP16, 4096×8192×2048 | load-next is not admitted; current supported traversal choices do not beat ptxas | Keep baseline fallback |
| QKV, output, FFN-down, small/medium/large held-outs | Morton/grouped-N mostly tie or regress OA; ptxas is faster | No role-wide traversal dispatch |

The source-hashed raw runs are in `exp/shape_coverage/a100-role-gemm_*.json`.
The small-grid experiment below starts a separate tile family; the earlier
traversal study alone did not establish a winning tile size.

### Pointer sweep invalidation and replay

The initial pointer-policy files (`082832`, `083147`, `083359` UTC stamps)
renamed the entry and emitted unannotated PTX. Pointer-home selection is carried
by the checked source annotation and `gemm` entry, not by `generate(plan)` alone.
Consequently their `pointer_direct` labels did not exercise direct lowering.
The raw files are retained, but those labels are invalid evidence and no
pointer dispatch decision may use them. This is a harness failure, not a
devalidation of the separately tested production direct-pointer implementation.

The revised schema-2 runner preserves the checked source and verifies each
requested policy against the actual native build manifest. It adds a matched
cuBLAS GemmEx FP32-output control and records poisoned-graph results explicitly.
Corrected replays below replace the invalid pointer comparison. A
client-disconnected attempt produced no result artifact.

The corrected [FP16 replay](a100-role-gemm_20261109T110806.491968Z.json)
completed on A100-SXM4-40GB: 45 method/input and 15 poisoned-graph checks pass.
The [FP16/BF16 fragment study](a100-role-gemm_20261109T111245.728446Z.json)
passes another 126 method/input and 42 graph checks on A100 80GB PCIe. Its
three-round direct+early OA runtime is about 12–13% below source-order OA on
1024×4096×1024, but still behind same-source ptxas. BF16 direct+early is about
5.6–5.7% below the matched default GemmEx library control on that worker;
this is not a ptxas win, cuBLASLt autotuning result, or whole-step speedup.
Absolute times across these device types are not pooled.

The [eight-case role/shape sweep](a100-role-gemm_20261109T112003.101015Z.json)
adds base-report, small and large FFN-up cases in both dtypes: all 456
method/input and 152 graph checks pass. It finds important counterexamples:
split-2/4/6 regress the 1024×2048×512 base-report case by roughly 6–8% and the
4096×8192×2048 large case by roughly 3–5% in the last round. Direct+early wins
on the medium case, but its large-case gain is only about 1–2%. A role-wide
rule is not supported. The local source guard is false because experiment
harness files changed during this run; it is retained as bounded numerical/
timing evidence and excluded from strict dispatch promotion.

Split-2/4/6 fragment schedules retain every accumulator's K order and all copy,
visibility, address and observer envelopes. Encoded OA HMMA controls remain
8 and post-run NOP drains remain 15. All three pass hardware checks. Their
register counts are 244/240/236; split-6 does not beat split-4 and split-2 is
slower on this role. Fewer registers alone are not a selection criterion.

The [matched six-method counter replay](a100-role-gemm_20261109T111551.304398Z.json)
collects all 20 requested metrics. Native OA baseline executes 4.896M dynamic
instructions, direct+early 4.371M and same-source ptxas 3.231M, all with exactly
2.097M tensor instructions. Direct+early therefore still has about 35.3% more
total instructions than ptxas. Shared-bank counters improve under load-next,
but tensor utilization and wait/scoreboard behavior do not support treating
that counter alone as the cause of the speedup. cuBLAS executes 5.418M total
instructions and is faster than OA FP16: instruction count alone is not a
performance model. See the [counter interpretation](../../docs/reverse_eng_results/11_COUNTERS_PERFORMANCE_AND_VALIDATION.md).

### Small-grid tile family

The first [64×64 CTA hardware study](a100-role-gemm_20261109T112122.548949Z.json)
targets output projection at 128×512×512 FP16. All 15 method/input and five
poisoned-graph checks pass, including exact equality to baseline and same-policy
ptxas. Four 32×32 warp tiles use 32 accumulator values per thread, 88 allocated
registers and 32 KiB shared memory. This is an explicit experiment contract,
not an inferred production GEMM annotation.

OA time falls from 21.25–21.38 µs to 11.01 µs across three rounds. The 64-tile
OA candidate also beats the old 128-tile ptxas control (14.85 µs), but matched
64-tile ptxas is faster at 8.45 µs and cuBLAS takes 7.68 µs. This separates a
kernel-geometry improvement from backend/compiler quality. Larger grids may
lose because smaller tiles repeat more operand traffic and scalar work. The
candidate therefore needs role/shape validation before any dispatch decision.

The [completed 32-case sweep](a100-role-gemm_20261109T113850.802215Z.json)
now supplies that broader comparison: all 1,296 method/input and 432 graph
checks pass with unchanged source identity. Tile64 wins all three rounds on
18 cases and regresses 14. Small-token roles improve roughly 45–54%; medium
output and FFN-down improve roughly 21–28%. Medium QKV regresses roughly
33–35%, and large cases regress roughly 24–49%. The best measured ptxas
option remains faster than the best OA option in every case.

The [predeclared independent small-output replay](a100-role-gemm_20261109T113154.637725Z.json)
also passes 33 method/input and 11 graph checks, agrees with the frozen source
hashes and retains roughly 49% lower OA runtime. The detailed positive and
negative findings are centralized in the
[small-tile dossier](../../docs/research/ideas/small-grid-tensor-cta-tiles.md).

## Scheduler and coverage boundaries

The A100 clock probe measured about 24.023 cycles/HMMA for one dependent chain,
12.012 for two independent chains and 8.008 for three. This motivates separate
RAW and initiation constraints, but does not validate a reordered production
GEMM. HMMA8/drain15 remains the correctness fallback until a full encoded-stream
candidate passes numerical, resource, held-out timing and counter checks.

The current lowered-adapter audit also exposes a safety gap in the earlier
model: none of the 45 audited actual-GEMM regions can safely be reordered with
its present visibility, predicate and implicit-operand coverage. They now fail
closed. A provenance-labelled 24/8 model profile is available for the measured
accumulator/initiation classes; other distances remain uncalibrated. Model
cycle estimates are not hardware timings and no general permutation is encoded.

The production contract still excludes tails, arbitrary strides, transposed
storage, TF32, arbitrary alpha/beta, batched-head execution, fused attention
backward, affine/saved-stat normalization, and a complete native optimizer
route. The role manifest's `evidence_id` field is a proposed dispatch key; no
automatic role/shape policy should be selected until its exact key has paired
OA/ptxas/library evidence and a held-out case.

## Experiment sequence

1. Capture first divergent output/gradient/state at every supported optimizer
   step, preserving the original precision and optimizer policy.
2. Extend native routes for aligned QKV/output/FFN cases and explicitly reject
   unsupported tails, layouts and batched attention until each contract exists.
3. Compare source-order, load-next, grouped-N, Morton, direct-pointer and early
   advancement for each exact role/shape, including packing cost for dX/dW.
4. Use NCU on the selected role winners and same-source ptxas before exploring
   smaller tiles or relaxing HMMA dependency controls.
5. Only after standalone gates pass, substitute one native operator at a time
   in a captured graph and compare full outputs, gradients, parameters and
   optimizer state before timing a whole step.

Reproduce the current studies with the commands and case keys in
[`README.md`](README.md). Static support is not correctness evidence, and a
kernel timing is not a whole-step result.
